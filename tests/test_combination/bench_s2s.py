import torch
import torch.distributed as dist
from torch.distributed._tensor.placement_types import Shard, Replicate

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import Placement, Replicate, Shard, DTensor, distribute_tensor
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec

from easydist.torch.placement_types import Partition, TensorMeta
from time import time

device_mesh = DeviceMesh('cuda', torch.arange(4).reshape(2, 2))
global_tensor = torch.randn(4096, 8192, 4).cuda()

rank = dist.get_rank()

runs = 1000

def bench(src_placement, tgt_placement):
    src_dtensor = distribute_tensor(global_tensor, device_mesh, src_placement)

    start_t = time()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    for _ in range(runs):
        tgt_dtensor = src_dtensor.redistribute(device_mesh, tgt_placement)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"=====================================================")
        print(f"redistribute time: {time() - start_t}, max memory: {torch.cuda.max_memory_allocated() / 1024**2} MB")

    src_partitions = Partition.from_tensor_spec(src_dtensor._spec)
    tgt_partitions = Partition.from_tensor_spec(tgt_dtensor._spec)

    send_ops, recv_ops, recv_buffer, recv_slices = Partition.gen_p2p_op(rank, src_dtensor._local_tensor, src_partitions, tgt_partitions)

    start_t = time()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    for _ in range(runs):
        works = dist.batch_isend_irecv(send_ops + recv_ops)
        for work in works:
            work.wait()
        recv_buffer = Partition.fill_recv_buffer(recv_ops, recv_buffer, recv_slices)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"p2p time: {time() - start_t} max memory: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    assert torch.allclose(recv_buffer, tgt_dtensor._local_tensor)

if __name__ == "__main__":
    bench([Shard(0), Shard(1)], [Replicate(), Replicate()])
    bench([Replicate(), Replicate()], [Shard(0), Shard(1)])
    bench([Shard(0), Shard(1)], [Shard(1), Shard(0)])
    bench([Shard(0), Shard(1)], [Shard(0), Shard(0)])
    bench([Shard(0), Shard(0)], [Shard(0), Shard(1)])
    bench([Shard(0), Shard(2)], [Shard(1), Shard(1)])
    bench([Shard(2), Shard(2)], [Shard(1), Shard(1)])
    bench([Shard(1), Shard(2)], [Shard(0), Shard(1)])