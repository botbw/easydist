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

# torchrun --nproc_per_node 4 benchmark/torch/pp/resnet101/pipeline.py
import argparse
import os
import random
import time

import numpy as np

import torch
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.distributed._tensor import DeviceMesh
from tqdm import tqdm

from benchmark.bench_case import GPTCase
from benchmark.torch.model.gpt import GPT, SequentialGPT
from easydist import easydist_setup
from easydist.torch.api import easydist_compile
from easydist.torch.device_mesh import get_pp_size, set_device_mesh
from easydist.torch.experimental.pp.PipelineStage import ScheduleDAPPLE, ScheduleGPipe
from easydist.torch.experimental.pp.compile_pipeline import (
    annotate_split_points,
    split_into_equal_size)


def seed(seed=42):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for built-in Python
    random.seed(seed)
    # Set(seed) for each of the random number generators in python:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


criterion = torch.nn.CrossEntropyLoss()

def test_main(args):
    per_chunk_sz = args.micro_batch_size
    num_chunks = args.num_chunks
    batch_size = per_chunk_sz * num_chunks
    schedule_cls = args.schedule == 'gpipe' and ScheduleGPipe or ScheduleDAPPLE

    seed(42)
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    pp_size = int(os.environ["WORLD_SIZE"])
    device = torch.device('cuda')
    torch.cuda.set_device(rank)

    case = GPTCase()
    model = SequentialGPT(depth=case.num_layers, dim=case.hidden_dim, num_heads=case.num_heads)
    data_in = torch.ones()

    annotate_split_points(module, {
            'layer1_1_residual',
            'layer2_2_relu3',
            'layer3_11_bn1'
    })
    opt = torch.optim.Adam(module.parameters(), foreach=True, capturable=True)
    # opt = torch.optim.SGD(module.parameters(), lr=0.001, foreach=True)
    @easydist_compile(parallel_mode="pp",
                    tracing_mode="fake",
                    cuda_graph=False,
                    schedule_cls=schedule_cls,
                    num_chunks=num_chunks,
                    all_gather_output=False)
    def train_step(input, label, model, opt):
        out = model(input)
        loss = criterion(out, label)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return out, loss

    dataset_size = 10000
    train_dataloader = [
        (torch.randn(case.batch_size, case.seq_size, case.hidden_dim))
    ] * (dataset_size // batch_size)

    x_batch, y_batch = next(iter(train_dataloader))
    train_step(x_batch.to(device), y_batch.to(device), module, opt) # compile
    epochs = 1
    time_sum = 0
    # with torch.autograd.profiler.profile() as prof:
    for epoch in range(epochs):
        all_cnt, correct_cnt, loss_sum = 0, 0, 0
        time_start = time.time()
        for x_batch, y_batch in tqdm(train_dataloader, dynamic_ncols=True) if rank == 0 else train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if x_batch.size(0) != batch_size:  # TODO need to solve this
                continue
            _ = train_step(x_batch, y_batch, module, opt)
        time_sum += time.time() - time_start
    with open(f'{schedule_cls.__name__}-{rank}-test.txt', 'a') as f:
        f.write(
            f'num_chunks: {num_chunks}, chunk_sz {per_chunk_sz}, time: {time_sum / epochs:2.1f}, memory: {torch.cuda.max_memory_allocated(rank) / 1024 / 1024:5.0f}MB\n'
        )
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro-batch-size', type=int, default=16)
    parser.add_argument('--num-chunks', type=int, default=2)
    parser.add_argument('--schedule', type=str, default='gpipe', choices=['gpipe', 'dapple'])
    args = parser.parse_args()
    test_main(args)