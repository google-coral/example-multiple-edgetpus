# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cc_binary(
    name = "multiple_edgetpu_demo",
    srcs = ["multiple_edgetpu_demo.cc"],
    deps = [
        "@local_gst//:gstreamer",
        "@libedgetpu//:header",
        "@edgetpu//src/cpp/pipeline:pipelined_model_runner",
        "@edgetpu//src/cpp/pipeline:utils",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@glog//:glog",
    ],
)
