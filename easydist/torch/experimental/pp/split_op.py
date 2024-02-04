# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch._custom_ops
import typing
'''
The valid parameters types are: 
dict_keys([
    <class 'torch.Tensor'>, 
    typing.Optional[torch.Tensor], 
    typing.Sequence[torch.Tensor], 
    typing.Sequence[typing.Optional[torch.Tensor]],
    <class 'int'>, 
    typing.Optional[int], 
    typing.Sequence[int], 
    typing.Optional[typing.Sequence[int]], 
    <class 'float'>, 
    typing.Optional[float], 
    typing.Sequence[float], 
    typing.Optional[typing.Sequence[float]], 
    <class 'bool'>, 
    typing.Optional[bool], 
    typing.Sequence[bool], 
    typing.Optional[typing.Sequence[bool]], 
    <class 'str'>, 
    typing.Optional[str], 
    typing.Union[int, float, bool], 
    typing.Union[int, float, bool, NoneType], 
    typing.Sequence[typing.Union[int, float, bool]],
    <class 'torch.dtype'>, 
    typing.Optional[torch.dtype], 
    <class 'torch.device'>, 
    typing.Optional[torch.device]]
    )
'''
param_type = typing.Sequence[typing.Optional[torch.Tensor]]
ret_type = typing.List[torch.Tensor]

@torch._custom_ops.custom_op("easydist::fw_bw_split")
def fw_bw_split(args: param_type) -> ret_type:
    ...

@torch._custom_ops.impl_abstract("easydist::fw_bw_split")
def fw_bw_split_impl_abstract(args: param_type) -> ret_type:
    need_clone = lambda arg: isinstance(arg, torch.Tensor) and arg.requires_grad
    args = [arg.clone() if need_clone(arg) else arg for arg in args]
    return args

@torch._custom_ops.impl("easydist::fw_bw_split")
def fw_bw_split_impl(args: param_type) -> ret_type:
    need_clone = lambda arg: isinstance(arg, torch.Tensor) and arg.requires_grad
    args = [arg.clone() if need_clone(arg) else arg for arg in args]
    return args


# doc https://pytorch.org/docs/stable/notes/extending.html#how-to-use
class FWBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_list): # tensors must be passed as args
        tensor_list = list(tensor_list) # custom op requires list
        return tuple(torch.ops.easydist.fw_bw_split(tensor_list)) # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, *grads):
        return tuple(torch.ops.easydist.fw_bw_split(grads))


def fw_bw_split_func(tensor_list: typing.Tuple[torch.Tensor]) -> typing.Tuple[torch.Tensor]:
    return FWBWSplitFunc.apply(*tensor_list)


# doc https://pytorch.org/docs/stable/notes/extending.html#how-to-use
class BeforeBWSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor): # tensors must be passed as args
        return torch.ops.easydist.fw_bw_split([tensor])[0] # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, grad):
        return grad # return must be tensor of tuple of tensors

def before_bw_split_func(tensor):
    ret = BeforeBWSplitFunc.apply(tensor)
    return ret

@torch._custom_ops.custom_op("easydist::step_split")
def step_split(args: param_type) -> ret_type:
    ...

@torch._custom_ops.impl_abstract("easydist::step_split")
def step_split_impl_abstract(args: param_type) -> ret_type:
    need_clone = lambda arg: isinstance(arg, torch.Tensor) and arg.requires_grad
    args = [arg.clone() if need_clone(arg) else arg for arg in args]
    return args

@torch._custom_ops.impl("easydist::step_split")
def step_split_impl(args: param_type) -> ret_type:
    need_clone = lambda arg: isinstance(arg, torch.Tensor) and arg.requires_grad
    args = [arg.clone() if need_clone(arg) else arg for arg in args]
    return args

# doc https://pytorch.org/docs/stable/notes/extending.html#how-to-use
class StepSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *tensor_list): # tensors must be passed as args
        tensor_list = list(tensor_list)
        return tuple(torch.ops.easydist.step_split(tensor_list)) # return must be tensor of tuple of tensors

    @staticmethod
    def backward(ctx, *grad):
        return grad # return must be tensor of tuple of tensors

def step_split_func(tensor_list: typing.Tuple[torch.Tensor]) -> typing.Tuple[torch.Tensor]:
    return StepSplitFunc.apply(*tensor_list)

__params_buffers_named_states_global = None

def get_stateless_func_input_global():
    global __params_buffers_named_states_global
    return __params_buffers_named_states_global

def set_stateless_func_input_global(named_states):
    global __params_buffers_named_states_global
    __params_buffers_named_states_global = named_states


if __name__ == '__main__':
    a = torch.rand(10, 10, requires_grad=True)
    b = torch.rand(3, 10, 10, requires_grad=True)
    res = fw_bw_split_func([a, b])
    print(res[0].requires_grad, res[1].requires_grad)
    (res[0].mean() + res[1].mean()).backward()

    a = torch.rand(10, 10, requires_grad=True)
    res = before_bw_split_func(a)
    print(res.requires_grad)
    res.mean().backward()
