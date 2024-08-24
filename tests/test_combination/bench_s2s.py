from collections import defaultdict
from time import time
from typing import Tuple, Dict, List, Optional, cast
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import distribute_tensor
from torch.distributed._tensor.placement_types import Shard, Replicate, DTensorSpec

@dataclass
class Partition:
    '''
        Tensor partition from a logical view, and the rank that holds it.
    '''
    start_coord: Tuple[int, ...]  # start coord of this partition
    end_coord: Tuple[int, ...]  # end coord of this partition
    rank: int  # rank that hold this partition

    def __repr__(self) -> str:
        return f"{{{self.start_coord}->{self.end_coord} on rank{self.rank}}}"

    def shard(self, tensor_dim: int, shard_num: int, shard_idx: int) -> "Partition":
        if (self.end_coord[tensor_dim] - self.start_coord[tensor_dim]) % shard_num != 0:
            raise NotImplementedError(f"Padding not supported, found {shard_num=} {self.end_coord[tensor_dim]=} {self.start_coord[tensor_dim]=}")

        block_size = (self.end_coord[tensor_dim] - self.start_coord[tensor_dim]) // shard_num
        return Partition(
            self.start_coord[:tensor_dim] + (self.start_coord[tensor_dim] + block_size * shard_idx,) + self.start_coord[tensor_dim+1:],
            self.end_coord[:tensor_dim] + (self.start_coord[tensor_dim] + block_size * (shard_idx + 1),) + self.end_coord[tensor_dim+1:],
            self.rank
        )

    @staticmethod
    def from_tensor_spec(spec: DTensorSpec) -> Tuple["Partition"]:
        if spec.tensor_meta is None:
            raise ValueError("tensor_meta is not set")

        if any(p.is_partial() for p in spec.placements):
            raise NotImplementedError("Partial placement is not supported")

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
            Partition((0,) * len(global_shape), tuple(global_shape), rank) for rank in tensor_mesh.flatten().tolist()
        ]
        for mesh_dim, placement in enumerate(placements):
            if not placement.is_shard():
                continue
            shard = cast(Shard, placement)
            tensor_dim = shard.dim
            shard_num = tensor_mesh.size(mesh_dim)
            for partition in partitions:
                partitions[partition.rank] = partition.shard(tensor_dim, shard_num, rank_to_coord[partition.rank][mesh_dim])

        return tuple(partitions)

    def from_src(self, src_partition: "Partition") -> Optional["Partition"]:

        start_coord = tuple(max(s1, s2) for s1, s2 in zip(self.start_coord, src_partition.start_coord))
        end_coord = tuple(min(e1, e2) for e1, e2 in zip(self.end_coord, src_partition.end_coord))
        if any(s >= e for s, e in zip(start_coord, end_coord)):
            return None

        return Partition(start_coord, end_coord, src_partition.rank)

    @staticmethod
    def gen_recv_meta(src_partitions: Tuple["Partition"], tgt_partitions: Tuple["Partition"]) -> Dict[int, List["Partition"]]:
        recv_info = defaultdict(list)
        for tgt in tgt_partitions:
            cache = defaultdict(int)
            # TODO better load balance
            for src in sorted(src_partitions, key=lambda x: abs(x.rank - tgt.rank)):
                intersection = tgt.from_src(src)
                if intersection is None:
                    continue
                cache_str = f"{intersection.start_coord}{intersection.end_coord}"
                if cache_str not in cache:
                    recv_info[tgt.rank].append(intersection)
                cache[cache_str] += 1

        return recv_info

    @staticmethod
    def gen_p2p_utils(rank: int, src_tensor: torch.Tensor, local_src_partition: "Partition", local_dst_partition: "Partition", recv_meta: Dict[int, List["Partition"]]) -> Tuple[List[dist.P2POp], List[dist.P2POp], torch.Tensor, List[Tuple[slice]]]:
        send_ops = []
        recv_ops = []
        recv_slices = []
        buffer_shape = tuple(e - s for s, e in zip(local_dst_partition.start_coord, local_dst_partition.end_coord))
        recv_buffer = torch.empty(buffer_shape, dtype=src_tensor.dtype, device=src_tensor.device)
        for recv_rank, intersections in recv_meta.items():
            for intersection in intersections:
                send_rank = intersection.rank
                src_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_src_partition.start_coord, intersection.start_coord, intersection.end_coord))
                tgt_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_dst_partition.start_coord, intersection.start_coord, intersection.end_coord))

                if rank == send_rank == recv_rank:
                    # directly assign the intersection to the buffer
                    recv_buffer[tgt_slice] = src_tensor[src_slice]
                    continue

                if rank == send_rank:
                    src_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_src_partition.start_coord, intersection.start_coord, intersection.end_coord))
                    send_ops.append(dist.P2POp(dist.isend, src_tensor[src_slice].contiguous(), recv_rank))

                if rank == recv_rank:
                    tgt_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_dst_partition.start_coord, intersection.start_coord, intersection.end_coord))
                    recv_ops.append(dist.P2POp(dist.irecv, recv_buffer[tgt_slice].contiguous(), send_rank))
                    recv_slices.append(tgt_slice)

        # TODO: handle the case when there is no communication
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
            # TODO: avoid redundant buffer and mem copy
            recv_buffer[slice] = op.tensor

        return recv_buffer


device_mesh = DeviceMesh('cuda', torch.arange(4).reshape(2, 2))
global_tensor = torch.randn(16, 8192, 8192).cuda()
rank = dist.get_rank()
runs = 100

def bench(src_placement, tgt_placement):
    src_dtensor = distribute_tensor(global_tensor, device_mesh) # replicate from rank 0
    src_dtensor = src_dtensor.redistribute(device_mesh, src_placement)

    start_t = time()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    for _ in range(runs):
        tgt_dtensor = src_dtensor.redistribute(device_mesh, tgt_placement)
    torch.cuda.synchronize()
    dist.barrier()
    rule_time = time() - start_t
    rule_mem = torch.cuda.max_memory_allocated()

    src_partitions = Partition.from_tensor_spec(src_dtensor._spec)
    tgt_partitions = Partition.from_tensor_spec(tgt_dtensor._spec)
    recv_info = Partition.gen_recv_meta(src_partitions, tgt_partitions)

    start_t = time()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    for _ in range(runs):
        send_ops, recv_ops, recv_buffer, recv_slices = Partition.gen_p2p_utils(rank, src_dtensor._local_tensor, src_partitions[rank], tgt_partitions[rank], recv_info)
        works = dist.batch_isend_irecv(send_ops + recv_ops)
        for work in works:
            work.wait()
        recv_buffer = Partition.fill_recv_buffer(recv_ops, recv_buffer, recv_slices)
    torch.cuda.synchronize()
    dist.barrier()
    p2p_time = time() - start_t
    p2p_mem = torch.cuda.max_memory_allocated()

    assert torch.allclose(recv_buffer, tgt_dtensor._local_tensor), "Mismatch"
    if rank == 0:
        print(f"============================{src_dtensor._spec} -> {tgt_dtensor._spec}================================")
        print(f"Rule based time: {rule_time}, max memory: {rule_mem / 1024**2} MB")
        print(f"P2P based time: {p2p_time} max memory: {p2p_mem / 1024**2} MB")
        print("Is P2P better? time: " + ("✅" if p2p_time < rule_time else "❌") + " mem: " + ("✅" if p2p_mem < rule_mem else "❌"))

if __name__ == "__main__":
    # choices = [Replicate(), Shard(0), Shard(1), Shard(2)]
    # for a in choices:
    #     for b in choices:
    #         for c in choices:
    #             for d in choices:
    #                 bench([a, b], [c, d])
    # exit()

    bench([Shard(0), Shard(0)], [Replicate(), Replicate()])
    bench([Replicate(), Replicate()], [Shard(0), Shard(0)])

    bench([Shard(0), Replicate()], [Shard(1), Replicate()])
    bench([Shard(1), Replicate()], [Shard(0), Replicate()])

    bench([Shard(1), Replicate()], [Shard(2), Replicate()])
    bench([Shard(2), Replicate()], [Shard(1), Replicate()])

    bench([Shard(0), Shard(1)], [Shard(2), Shard(1)])
    bench([Shard(2), Shard(1)], [Shard(0), Shard(1)])

    bench([Shard(0), Shard(1)], [Replicate(), Replicate()])
    bench([Replicate(), Replicate()], [Shard(0), Shard(1)])

    bench([Shard(0), Shard(1)], [Shard(1), Shard(0)])
    bench([Shard(1), Shard(0)], [Shard(0), Shard(1)])

    bench([Shard(2), Shard(1)], [Shard(1), Shard(2)])
    bench([Shard(1), Shard(2)], [Shard(2), Shard(1)])

    bench([Shard(0), Shard(0)], [Shard(0), Shard(1)])
    bench([Shard(0), Shard(1)], [Shard(0), Shard(0)])

    bench([Shard(0), Shard(0)], [Shard(1), Shard(1)])
    bench([Shard(1), Shard(1)], [Shard(0), Shard(0)])

    bench([Shard(0), Shard(1)], [Shard(1), Shard(2)])
    bench([Shard(1), Shard(2)], [Shard(0), Shard(1)])

