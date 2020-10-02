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

cc_library(
    name = "libedgetpu",
    deps = [
      "@libedgetpu//driver/beagle:beagle_all_driver_provider",
      "@libedgetpu//tflite/public:edgetpu",
      "@libedgetpu//tflite/public:edgetpu_c",
      "@libedgetpu//tflite:custom_op_direct",
    ],
)

cc_binary(
    name = "multiple_edgetpu_demo",
    srcs = ["multiple_edgetpu_demo.cc"],
    deps = [
        ":libedgetpu",
        "@local//:gstreamer",
        "@local//:gtk",
        "@edgetpu//src/cpp/pipeline:pipelined_model_runner",
        "@edgetpu//src/cpp/pipeline:utils",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/strings",
        "@glog//:glog",
    ],
)
