# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# command:
# torchrun --nproc_per_node 8 --master_port 26543 ./mult_mesh1.py

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import stateless
from torch.fx.experimental.proxy_tensor import make_fx
from torch.distributed._tensor.device_mesh import init_device_mesh

from megatron.core import mpu, tensor_parallel
from torch.distributed._composable.fsdp import fully_shard


#from easydist.torch.scope_auto.scope_marker import scope_marker
#from easydist.torch.scope_auto.build_scope_modules import build_scope_modules
from easydist.torch.utils import _rematerialize_optimizer
from easydist.torch.experimental.pp.split_utils import get_updated_params_states

marker_aux_vars = []

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor, DTensor

_world_size = int(os.environ["WORLD_SIZE"])
assert (
    _world_size == 8
), f"require 8 GPUs, but got {_world_size} gpus"

device_mesh1 = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
device_mesh2 = init_device_mesh(device_type="cuda", mesh_shape=(int(_world_size//4), 4,), mesh_dim_names=("dp", "tp"))

_rank = torch.distributed.get_rank()
device_id = _rank % torch.cuda.device_count()

#print(f"[{_rank}] mesh1: {repr(device_mesh1)}, ndim: {device_mesh1.ndim}")
#print(f"[{_rank}] mesh2: {repr(device_mesh2)}, ndim: {device_mesh2.ndim}")
#print(f"[{_rank}] mesh2 dp: {device_mesh2['dp']}, mesh2 tp: {device_mesh2['tp']}")

def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    **kwargs,
):
    mpu.destroy_model_parallel()
    mpu.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank,
        **kwargs,
    )

class CombinedImageTextNet(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(CombinedImageTextNet, self).__init__()

        # combined feature
        self.fc_combined = nn.Linear(hidden_size * 2, output_size)

    def forward(self, combined_features):
        output = self.fc_combined(combined_features)

        return output

class ImageTextNet(nn.Module):
    def __init__(self, input_size_image, input_size_text, hidden_size, output_size):
        super(ImageTextNet, self).__init__()

        # fc for image
        self.fc_image = nn.Linear(input_size_image, hidden_size)
        # fc for text
        self.fc_text = nn.Linear(input_size_text, hidden_size)

        # combined feature
        self.combined_net = CombinedImageTextNet(hidden_size, output_size)

    def forward(self, image, text):
        # combine features
        combined_features = self.combine_feat(image, text)
        output = self.combined_net(combined_features)

        return output

    #@scope_marker(marker_aux_vars)
    def combine_feat(self, image, text):
        out_image = torch.relu(self.fc_image(image))
        out_text = torch.relu(self.fc_text(text))

        # combine features
        out_image = out_image.redistribute(device_mesh1, [Replicate()]).to_local()
        out_text = out_text.redistribute(device_mesh1, [Replicate()]).to_local()
        combined_features = torch.cat((out_image, out_text), dim=1)
        return combined_features


initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
)
dp_g = mpu.get_data_parallel_group()
tp_g = mpu.get_tensor_model_parallel_group()
#print(f"dp process group: {dp_g}")
#print(f"tp process group: {tp_g}")

batch_size = 4
use_dtensor = True
image_feat_dim = 100
text_feat_dim = 56
hidden_size = 8
output_size = 12

# random inputs
if use_dtensor:
    x_image = torch.randn(batch_size, image_feat_dim, device="cuda")
    x_text = torch.randn(batch_size, text_feat_dim, device="cuda")
    x_image = distribute_tensor(x_image, device_mesh1, [Replicate()])
    x_text = distribute_tensor(x_text, device_mesh1, [Shard(1)])
else:
    x_image = torch.randn(batch_size, image_feat_dim, device="cuda")
    x_text = torch.randn(batch_size, text_feat_dim//_world_size, device="cuda")

y_target = torch.randn(batch_size, output_size, device="cuda")  # label_dim=output_size

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_model_and_optimizer(seed, learning_rate=0.01):
    set_seed(seed)
    model = ImageTextNet(input_size_image=image_feat_dim, input_size_text=text_feat_dim, hidden_size=hidden_size, output_size=output_size).to(device_id)
    model = parallelize_module(
        module=model,
        device_mesh=device_mesh1,
        parallelize_plan={
            "fc_image": ColwiseParallel(use_local_output=False),
            "fc_text": RowwiseParallel(use_local_output=False),
        },
    )
    model.combined_net = parallelize_module(
        module=model.combined_net,
        device_mesh=device_mesh2["tp"],
        parallelize_plan={
            "fc_combined": ColwiseParallel(use_local_output=False),
        },
    )
    fully_shard(model.combined_net, mesh=device_mesh2["dp"])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer

# loss
criterion = nn.MSELoss()

seed = 3

#--------------------------------------------------------
# first training session
#--------------------------------------------------------

# initialized model and optimizer
model1, optimizer1 = initialize_model_and_optimizer(seed)

model_state_dict1 = model1.state_dict()
optimizer_state_dict1 = optimizer1.state_dict()
#print(f"model state(orig):\n{model_state_dict1}")
#print(f"\noptimizer state(orig):\n{optimizer_state_dict1}")

def train_step(image, text, target):
    output = model1(image, text)
    output = output.redistribute(device_mesh=None, placements=[Replicate()]).to_local()
    loss = criterion(output, target)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

    return loss

losses1 = []
print("run origin model")
for epoch in range(10):
    loss = train_step(x_image, x_text, y_target)
    losses1.append(loss.item())

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')

# save model and optimizer stateful variables
model_state_dict1 = model1.state_dict()
optimizer_state_dict1 = optimizer1.state_dict()

#--------------------------------------------------------
# second training session
#--------------------------------------------------------

# initialized model and optimizer
model2, optimizer2 = initialize_model_and_optimizer(seed)

model_state_dict2 = model2.state_dict()
optimizer_state_dict2 = optimizer2.state_dict()
#print(f"model state(traced graph):\n{model_state_dict2}")
#print(f"\noptimizer state(traced graph):\n{optimizer_state_dict2}")

def forward_backward_pass(image, text, target):
    output = model2(image, text)
    output = output.redistribute(device_mesh=None, placements=[Replicate()]).to_local()
    loss = criterion(output, target)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    return output, loss


params = dict(model2.named_parameters())
buffers = dict(model2.named_buffers())

named_states = {}
for n, p in params.items():
    if p in optimizer2.state:
        named_states[n] = opt.state[p]

def stateless_func(params, buffers, named_states, image, text, target):
    with stateless._reparametrize_module(model2, {**params, **buffers}), _rematerialize_optimizer(
          optimizer2, named_states, params):
        ret = forward_backward_pass(image, text, target)
        if (tup := get_updated_params_states()) != (None, None):
            params, named_states = tup
        grads = {k: v.grad for k, v in params.items()}
        return params, buffers, named_states, grads, ret

traced_graph = make_fx(stateless_func)(params, buffers, named_states, x_image, x_text, y_target)

#print("\n\nbefore building sub module")
#print(traced_graph)

#traced_graph = build_scope_modules(traced_graph)
traced_graph.recompile()

#print("\n\nafter building sub module")
#print(traced_graph)

#for node in traced_graph.graph.nodes:
#    print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}, Args: {node.args}")

losses2 = []
# training loop
print("\nrun traced graph")
for epoch in range(10):
    with torch.no_grad():
        params, buffers, named_states, grads, ret = traced_graph(params, buffers, named_states, x_image, x_text, y_target)

    _, loss = ret
    losses2.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')

loss_array1 = np.array(losses1, dtype=np.float32)
loss_array2 = np.array(losses2, dtype=np.float32)
assert np.allclose(loss_array1, loss_array2, rtol=1e-01, atol=1e-01), "test failed"
