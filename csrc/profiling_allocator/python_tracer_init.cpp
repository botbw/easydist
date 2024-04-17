/* Copyright (c) 2024, Alibaba Group;
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
==============================================================================*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stream_tracer.h"
#include "profiling_allocator.h"

namespace py = pybind11;

PYBIND11_MODULE(profiling_allocator, m) {
    m.doc() = "memory tracer";
    m.def("_enable_stream_tracer", enableStreamTracer);
    m.def("_disable_stream_tracer", disableStreamTracer);
    m.def("_prepare_stream_tracer", prepareStreamTracer);
    m.def("_set_cur_op_name", setCurOpName);
    m.def("_activate_stream_tracer", activateStreamTracer);
    m.def("_inactivate_stream_tracer", inactivateStreamTracer);
    m.def("_enter_op_core", enterOpCore);
    m.def("_leave_op_core", leaveOpCore);
    m.def("_get_stream_trace_data", getStreamTraceData);
    
    m.def("_set_start_recording", set_start_recording);
    m.def("_set_allocator_mode", set_allocator_mode);
    m.def("_set_customized_flag", set_customized_flag);
    m.def("_set_mem_size", set_mem_size);
    m.def("_set_temp_mem_size", set_temp_mem_size);
    m.def("_set_raw_mem_allocs", set_raw_mem_allocs);
    m.def("_get_allocator_profiling_info", get_allocator_profiling_info);

    py::enum_<AllocatorMode>(m, "AllocatorMode")
        .value("PROFILE", AllocatorMode::PROFILE)
        .value("RUNTIME", AllocatorMode::RUNTIME);

    py::class_<StreamTraceData>(m, "StreamTraceData")
        .def(py::init<>())
#ifdef TRACER_VERBOSE
        .def_readwrite("op_kernels", &StreamTraceData::op_kernels_)
        .def_readwrite("op_extra_kernels", &StreamTraceData::op_extra_kernels_)
#endif
        .def_readwrite("op_streams", &StreamTraceData::op_streams_)
        .def_readwrite("op_extra_streams", &StreamTraceData::op_extra_streams_);
}


