# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, NamedTuple, cast, List, Optional, Tuple

import torch
import torch.distributed as dist

from torch.distributed._tensor import Shard
from torch.distributed._tensor.placement_types import DTensorSpec


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
    global_shape: torch.Size  # logical shape of tensor
    start_coord: Tuple[int, ...]  # start coord of this partition
    end_coord: Tuple[int, ...]  # end coord of this partition
    rank: int  # rank that hold this partition

    def __post_init__(self):
        if len(self.global_shape) != len(self.start_coord) or len(self.global_shape) != len(self.end_coord):
            raise ValueError("global_shape, start_coord and end_coord must have the same length")

    def __repr__(self) -> str:
        return f"[{self.start_coord}->{self.end_coord} {self.rank}]"

    def shard(self, tensor_dim: int, shard_num: int, shard_idx: int) -> "Partition":
        if (self.end_coord[tensor_dim] - self.start_coord[tensor_dim]) % shard_num != 0:
            raise ValueError(f"shard_num must be a factor of the partition size, found {shard_num=} {self.end_coord[tensor_dim]=} {self.start_coord[tensor_dim]=}")

        block_size = (self.end_coord[tensor_dim] - self.start_coord[tensor_dim]) // shard_num
        return Partition(
            self.global_shape,
            self.start_coord[:tensor_dim] + (block_size * shard_idx,) + self.start_coord[tensor_dim+1:],
            self.end_coord[:tensor_dim] + (block_size * (shard_idx + 1),) + self.end_coord[tensor_dim+1:],
            self.rank
        )

    @staticmethod
    def from_tensor_spec(spec: DTensorSpec) -> Tuple["Partition"]:
        if spec.tensor_meta is None:
            raise ValueError("tensor_meta is not set")

        tensor_mesh = spec.mesh.mesh
        global_shape, _, _ = spec.tensor_meta
        placements = spec.placements

        if any(p.is_partial() for p in placements):
            raise ValueError("Partition does not support Partial placement")

        unravel = torch.unravel_index(torch.arange(tensor_mesh.numel()), tensor_mesh.shape)
        rank_to_coord = []
        for i in range(unravel[0].shape[0]):
            rank_to_coord.append(
                tuple(unravel[j][i].item() for j in range(tensor_mesh.ndim))
            )
        partitions = [
            Partition(global_shape, (0,) * len(global_shape), global_shape, rank) for rank in tensor_mesh.flatten().tolist()
        ]

        for mesh_dim, placement in enumerate(placements):
            if not placement.is_shard():
                continue
            shard = cast(Shard, placement)
            tensor_dim = shard.dim
            shard_num = tensor_mesh.size(mesh_dim)
            for partition in partitions:
                partitions[partition.rank] = partition.shard(tensor_dim, shard_num, rank_to_coord[partition.rank][mesh_dim])

        return partitions

    def from_src(self, src_partition: "Partition") -> Optional["Partition"]:

        start_coord = tuple(max(s1, s2) for s1, s2 in zip(self.start_coord, src_partition.start_coord))
        end_coord = tuple(min(e1, e2) for e1, e2 in zip(self.end_coord, src_partition.end_coord))
        if any(s >= e for s, e in zip(start_coord, end_coord)):
            return None

        return Partition(self.global_shape, start_coord, end_coord, src_partition.rank)

    @staticmethod
    def gen_p2p_info(src_partitions: Tuple["Partition"], tgt_partitions: Tuple["Partition"]) -> Dict[int, List["Partition"]]:
        recv = defaultdict(list)
        for tgt in tgt_partitions:
            for src in src_partitions:
                intersection = tgt.from_src(src)
                if intersection is not None:
                    recv[tgt.rank].append(intersection)

        return recv

    @staticmethod
    def gen_p2p_op(this_rank: int, recv_info: Dict[int, List["Partition"]], tensor: torch.Tensor) -> Tuple[List[dist.P2POp], List[dist.P2POp]]:
        send_ops = []
        recv_ops = []

        for recv_rank, intersections in recv_info.items():
            for intersection in intersections:
                send_rank = intersection.rank
                if this_rank == send_rank == recv_rank:
                    continue

                slices = tuple(slice(start, end) for start, end in zip(intersection.start_coord, intersection.end_coord))
                if this_rank == send_rank:
                    send_ops.append(dist.P2POp(dist.isend, tensor[slices], recv_rank))

                if this_rank == recv_rank:  # TODO create tensor recv buffer
                    recv_ops.append(dist.P2POp(dist.irecv, tensor[slices], send_rank))

        return send_ops, recv_ops