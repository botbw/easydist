import torch
import torch.distributed as dist
from torch.distributed._tensor.placement_types import Shard, Replicate

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import Placement, Replicate, Shard, DTensor, distribute_tensor
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec

from easydist.torch.placement_types import Partition, TensorMeta

device_mesh = DeviceMesh('cuda', torch.arange(4).reshape(2, 2))
# global_tensor = torch.arange(64).reshape(8, 8).cuda()
global_tensor = torch.randn(8, 16).cuda()

rank = dist.get_rank()

def sleep():
    from time import sleep
    dist.barrier()
    sleep(0.01 * rank)

def test(src_placement, tgt_placement):
    src_dtensor = distribute_tensor(global_tensor, device_mesh, src_placement)
    tgt_dtensor = src_dtensor.redistribute(device_mesh, tgt_placement)

    src_partitions = Partition.from_tensor_spec(src_dtensor._spec)
    tgt_partitions = Partition.from_tensor_spec(tgt_dtensor._spec)

    send_ops, recv_ops, recv_buffer, recv_slices = Partition.gen_p2p_op(rank, src_dtensor._local_tensor, src_partitions, tgt_partitions)

    works = dist.batch_isend_irecv(send_ops + recv_ops)
    for work in works:
        work.wait()

    local_tensor = Partition.fill_recv_buffer(recv_ops, recv_buffer, recv_slices)
    sleep()
    print(f"{rank} local_tensor: {local_tensor}")

    assert torch.allclose(local_tensor, tgt_dtensor._local_tensor)

if __name__ == "__main__":
    test([Shard(1), Shard(0)], [Shard(0), Shard(1)])
    test([Shard(0), Shard(0)], [Shard(1), Shard(1)])
    test([Shard(0), Replicate()], [Replicate(), Shard(1)])
