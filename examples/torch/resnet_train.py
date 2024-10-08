# Copyright (c) 2023, Alibaba Group;
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

# ENABLE_COMPILE_CACHE=1 torchrun --nproc_per_node 4 examples/torch/resnet_train.py
import os
import random

import numpy as np

import torch
import torch.distributed as dist

from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.distributed._tensor import DeviceMesh
from tqdm import tqdm

from easydist import easydist_setup
from easydist.torch.api import easydist_compile
from easydist.torch.utils import seed
from easydist.torch.device_mesh import get_device_mesh, set_device_mesh
from easydist.torch.experimental.pp.runtime import ScheduleDAPPLE
from easydist.torch.experimental.pp.compile_pipeline import annotate_split_points


criterion = torch.nn.CrossEntropyLoss()


@easydist_compile(tracing_mode="fake",
                  cuda_graph=False,
                  schedule_cls=ScheduleDAPPLE,
                  num_chunks=16)
def train_step(input, label, model, opt):
    opt.zero_grad()
    out = model(input)
    loss = criterion(out, label)
    loss.backward()
    opt.step()
    return out, loss

def validation(module, valid_dataloader):
    rank = dist.get_rank()
    module.eval()
    correct_cnt, all_cnt = 0, 0
    for x_batch, y_batch in tqdm(valid_dataloader, dynamic_ncols=True):
        x_batch = x_batch.to(f'cuda:{rank}')
        y_batch = y_batch.to(f'cuda:{rank}')
        out = module(x_batch)
        preds = out.argmax(-1)
        correct_cnt += (preds == y_batch).sum()
        all_cnt += len(y_batch)
    print(f'valid accuracy: {correct_cnt / all_cnt}')

def test_main():
    seed(42)
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    set_device_mesh(
        DeviceMesh("cuda", [[0, 2], [1, 3]],
                   mesh_dim_names=["spmd", "pp"]))
    device = torch.device('cuda')

    module = resnet18().train().to(device)
    module.fc = torch.nn.Linear(512, 10).to(device)

    annotate_split_points(module, {'layer3.0'})

    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)
    # opt = torch.optim.SGD(module.parameters(), lr=0.001, foreach=True)

    batch_size = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_data = datasets.CIFAR10('./data',
                                  train=True,
                                  download=True,
                                  transform=transform)
    valid_data = datasets.CIFAR10('./data', train=False, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data,
                                                   batch_size=batch_size)

    epochs = 5
    for _ in range(epochs):
        all_cnt, correct_cnt, loss_sum = 0, 0, 0
        for x_batch, y_batch in (tqdm(train_dataloader, dynamic_ncols=True) if rank == 0
                                    else train_dataloader):
            if x_batch.size(0) != batch_size:  # TODO need to solve this
                continue
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            out, loss = train_step(x_batch, y_batch, module, opt)
            all_cnt += len(out)
            preds = out.argmax(-1)
            correct_cnt += (preds.to(device) == y_batch.to(device)).sum()
            loss_sum += loss.mean().item()

        if rank == 0:
            print(f"loss_sum: {loss_sum} train acc: {correct_cnt / all_cnt}")

        # run validation with torch model
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(cur_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        state_dict = train_step.compiled_func.state_dict(True)

        if rank == 0:
            torch.save(state_dict, os.path.join(ckpt_dir, 'resnet.pth'))

        dist.barrier()
        if rank == world_size - 1:
            torch_model = resnet18().train().to(device)
            torch_model.fc = torch.nn.Linear(512, 10).to(device)
            torch_model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'resnet.pth')))
            validation(torch_model, valid_dataloader)
        dist.barrier()

    print(f"rank {rank} peek memory: {torch.cuda.max_memory_allocated()}")

    if rank == 0:
        os.remove(os.path.join(ckpt_dir, 'resnet.pth'))


if __name__ == '__main__':
    test_main()
