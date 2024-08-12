from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch.distributed._tensor.placement_types import Shard, Replicate
from torch.distributed._tensor.api import DTensorSpec
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import Placement, Replicate, Shard
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec

from easydist.torch.placement_types import Partition, TensorMeta

@dataclass
class FakeDeviceMesh:
    mesh: torch.Tensor

    def size(self, mesh_dim: Optional[int] = None) -> int:
        return self.mesh.numel() if mesh_dim is None else self.mesh.size(mesh_dim)

class TestPartition(TestCase):
    def test_partition(self):
        global_tensor = torch.arange(72).reshape(6, 12)
        device_mesh = FakeDeviceMesh(torch.arange(6).reshape(2, 3))

        src_spec = DTensorSpec(mesh=device_mesh, placements=[Shard(0), Shard(1)], tensor_meta=TensorMeta.from_global_tensor(global_tensor))
        tgt_spec = DTensorSpec(mesh=device_mesh, placements=[Shard(1), Shard(0)], tensor_meta=TensorMeta.from_global_tensor(global_tensor))

        src_partitions = Partition.from_tensor_spec(src_spec)
        tgt_partitions = Partition.from_tensor_spec(tgt_spec)

        print(src_partitions)
        print(tgt_partitions)

        recv_infos = Partition.gen_p2p_info(src_partitions, tgt_partitions)
        for rank, pip in recv_infos.items():
            print(rank, pip)

        send_ops, recv_ops = Partition.gen_p2p_op(1, recv_infos, global_tensor)

        for op in send_ops:
            print(op.peer, op.tensor)

        for op in recv_ops:
            print(op.peer, op.tensor)

if __name__ == "__main__":
    run_tests()
