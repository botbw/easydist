# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, NamedTuple, cast, List, Optional, Tuple

import torch
import torch.distributed as dist

from torch.distributed._tensor.placement_types import DTensorSpec, _Partial, Shard


class TensorMeta(NamedTuple):
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    shape: torch.Size
    stride: Tuple[int, ...]
    dtype: torch.dtype

    @staticmethod
    def from_global_tensor(tensor: torch.Tensor) -> "TensorMeta":
        return TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)


@dataclass
class Partition:
    '''
        Tensor partition from a logical view, and the rank that holds it.
    '''
    start_coord: Tuple[int, ...]  # start coord of this partition
    end_coord: Tuple[int, ...]  # end coord of this partition
    rank: int  # rank that hold this partition
    partial_coord: Tuple[int, ...]  # partial coord of this partition

    def __repr__(self) -> str:
        return f"{{{self.start_coord}->{self.end_coord} partial {self.partial_coord} on rank{self.rank}}}"

    def shard(self, tensor_dim: int, shard_num: int, shard_idx: int) -> "Partition":
        if (self.end_coord[tensor_dim] - self.start_coord[tensor_dim]) % shard_num != 0:
            raise ValueError(f"shard_num must be a factor of the partition size, found {shard_num=} {self.end_coord[tensor_dim]=} {self.start_coord[tensor_dim]=}")

        block_size = (self.end_coord[tensor_dim] - self.start_coord[tensor_dim]) // shard_num
        return Partition(
            self.start_coord[:tensor_dim] + (self.start_coord[tensor_dim] + block_size * shard_idx,) + self.start_coord[tensor_dim+1:],
            self.end_coord[:tensor_dim] + (self.start_coord[tensor_dim] + block_size * (shard_idx + 1),) + self.end_coord[tensor_dim+1:],
            self.rank,
            self.partial_coord
        )

    def partial(self, new_partial_idx: int) -> "Partition":
        return Partition(self.start_coord, self.end_coord, self.rank, self.partial_coord + (new_partial_idx,))

    @staticmethod
    def from_tensor_spec(spec: DTensorSpec) -> Tuple["Partition"]:
        if spec.tensor_meta is None:
            raise ValueError("tensor_meta is not set")

        tensor_mesh = spec.mesh.mesh
        global_shape, _, _ = spec.tensor_meta
        placements = spec.placements

        unravel = torch.unravel_index(torch.arange(tensor_mesh.numel()), tensor_mesh.shape)
        rank_to_coord = []
        for i in range(unravel[0].shape[0]):
            rank_to_coord.append(
                tuple(unravel[j][i].item() for j in range(tensor_mesh.ndim))
            )
        partitions = [
            Partition((0,) * len(global_shape), tuple(global_shape), rank, tuple()) for rank in tensor_mesh.flatten().tolist()
        ]

        for mesh_dim, placement in enumerate(placements):
            if placement.is_shard():
                shard = cast(Shard, placement)
                tensor_dim = shard.dim
                shard_num = tensor_mesh.size(mesh_dim)
                for partition in partitions:
                    partitions[partition.rank] = partition.shard(tensor_dim, shard_num, rank_to_coord[partition.rank][mesh_dim])
            elif placement.is_partial():
                partial = cast(_Partial, placement)
                for partition in partitions:
                    partitions[partition.rank] = partition.partial(rank_to_coord[partition.rank][mesh_dim])
        return tuple(partitions)

    def from_src(self, src_partition: "Partition") -> Optional["Partition"]:

        start_coord = tuple(max(s1, s2) for s1, s2 in zip(self.start_coord, src_partition.start_coord))
        end_coord = tuple(min(e1, e2) for e1, e2 in zip(self.end_coord, src_partition.end_coord))
        if any(s >= e for s, e in zip(start_coord, end_coord)) or src_partition.partial_coord != self.partial_coord:
            return None

        return Partition(start_coord, end_coord, src_partition.rank, src_partition.partial_coord)

    @staticmethod
    def gen_p2p_op(rank: int, src_tensor: torch.Tensor, src_partitions: Tuple["Partition"], tgt_partitions: Tuple["Partition"]) -> Tuple[List[dist.P2POp], List[dist.P2POp], torch.Tensor, List[Tuple[slice]]]:
        local_src_partition = src_partitions[rank]
        local_dst_partition = tgt_partitions[rank]

        recv_info = defaultdict(list)
        for tgt in tgt_partitions:
            cache = defaultdict(int)
            # TODO better load balance strategy
            for src in sorted(src_partitions, key=lambda x: abs(x.rank - tgt.rank)):
                intersection = tgt.from_src(src)
                if intersection is None:
                    continue
                cache_str = f"{intersection.start_coord}{intersection.end_coord}"
                if cache_str not in cache:
                    recv_info[tgt.rank].append(intersection)
                cache[cache_str] += 1

        send_ops = []
        recv_ops = []
        recv_slices = []
        buffer_shape = tuple(e - s for s, e in zip(local_dst_partition.start_coord, local_dst_partition.end_coord))
        recv_buffer = torch.empty(buffer_shape, dtype=src_tensor.dtype, device=src_tensor.device) * -10086
        for recv_rank, intersections in recv_info.items():
            for intersection in intersections:
                send_rank = intersection.rank
                src_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_src_partition.start_coord, intersection.start_coord, intersection.end_coord))
                tgt_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_dst_partition.start_coord, intersection.start_coord, intersection.end_coord))

                if rank == send_rank == recv_rank:
                    # assign the static intersection to the buffer
                    recv_buffer[tgt_slice] = src_tensor[src_slice]
                    continue

                if rank == send_rank:
                    src_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_src_partition.start_coord, intersection.start_coord, intersection.end_coord))
                    send_ops.append(dist.P2POp(dist.isend, src_tensor[src_slice].contiguous(), recv_rank))

                if rank == recv_rank:
                    tgt_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_dst_partition.start_coord, intersection.start_coord, intersection.end_coord))
                    recv_ops.append(dist.P2POp(dist.irecv, recv_buffer[tgt_slice].contiguous(), send_rank))
                    recv_slices.append(tgt_slice)

        # TODO: handle the case where there is no communication, currently sending a dummy tensor
        if len(send_ops + recv_ops) == 0:
            # NOTE: recv_buffer, recv_slices will still be empty
            dummy_tensor = torch.zeros(1, dtype=src_tensor.dtype, device=src_tensor.device)
            send_ops.append(dist.P2POp(dist.isend, dummy_tensor, rank))
            recv_ops.append(dist.P2POp(dist.irecv, dummy_tensor, rank))

        return send_ops, recv_ops, recv_buffer, recv_slices

    @staticmethod
    def fill_recv_buffer(recv_ops: List[dist.P2POp], recv_buffer: torch.Tensor, recv_slices: List[Tuple[slice]]) -> torch.Tensor:
        for op, slice in zip(recv_ops, recv_slices):
            if op.op is not dist.irecv:
                raise ValueError("recv_ops must be irecv")
            recv_buffer[slice] = op.tensor

        return recv_buffer
