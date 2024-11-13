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

import torch
import torch.fx as fx

from typing import List, Set

from easydist.torch.experimental.pp.utils import ordered_gi_users

def get_partition(fx_module: fx.GraphModule) -> List[Set[str]]:
    partitions = []
    cur_par = set()
    for node in fx_module.graph.nodes:
        cur_par.add(node.name)
        if node.op == 'call_function' and node.target in [
            torch.ops.easydist.fw_bw_split.default,
            torch.ops.easydist.step_split.default
        ]:
            partitions.append(cur_par)
            cur_par = set() 
    partitions.append(cur_par)
    return partitions

# def fix_dangling_comm_before_step(fx_module: fx.GraphModule) -> fx.GraphModule:
#     step_node = None
#     for node in fx_module.graph.nodes:
#         if node.op == 'call_function' and node.target == torch.ops.easydist.step_split.default:
#             step_node = node
#             break

#     if step_node is None:
#         return fx_module

#     step_node_getitems = ordered_gi_users(step_node)
#     optimizer_states_nodes = [
#         node for node in fx_module.graph.nodes
#         if node.op == 'placeholder' and 'arg2' in node.name  # TODO better way of determining optimizer states
#     ]

#     for node in optimizer_states_nodes:
#         if next(iter(node.users)) != step_node:
#             idx = step_node.args.index(node)
#             gi_user = step_node_getitems[idx]
#             p_node = next(iter(p_node.users))
#             dangling = []
#             while p_node != node:
#                 dangling.append(p_node)
#                 p_node = next(iter(p_node.users))
