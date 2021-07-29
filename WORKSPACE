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
workspace(name = "multiple_edgetpu_demo")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

TENSORFLOW_COMMIT = "a4dfb8d1a71385bd6d122e4f27f86dcebb96712d"
# Command to calculate: curl -L <FILE-URL> | sha256sum | awk '{print $1}'
TENSORFLOW_SHA256 = "cb99f136dc5c89143669888a44bfdd134c086e1e2d9e36278c1eb0f03fe62d76"

CORAL_CROSSTOOL_COMMIT = "6bcc2261d9fc60dff386b557428d98917f0af491"
CORAL_CROSSTOOL_SHA256 = "38cb4da13009d07ebc2fed4a9d055b0f914191b344dd2d1ca5803096343958b4"

# Configure libedgetpu and downstream libraries (TF and Crosstool).
http_archive(
  name = "libedgetpu",
  sha256 = "14d5527a943a25bc648c28a9961f954f70ba4d79c0a9ca5ae226e1831d72fe80",
  strip_prefix = "libedgetpu-3164995622300286ef2bb14d7fdc2792dae045b7",
  urls = [
    "https://github.com/google-coral/libedgetpu/archive/3164995622300286ef2bb14d7fdc2792dae045b7.tar.gz"
  ],
)

load("@libedgetpu//:workspace.bzl", "libedgetpu_dependencies")
libedgetpu_dependencies(TENSORFLOW_COMMIT, TENSORFLOW_SHA256,
                        CORAL_CROSSTOOL_COMMIT,CORAL_CROSSTOOL_SHA256)

http_archive(
  name = "libcoral",
  sha256 = "e41d71a080314734f73895539c05314caf028b63dc8172931b9eb12f7d01a62c",
  strip_prefix = "libcoral-6589d0bb49c7fdbc4194ce178d06f8cdc7b5df60",
  urls = [
    "https://github.com/google-coral/libcoral/archive/6589d0bb49c7fdbc4194ce178d06f8cdc7b5df60.tar.gz"
  ],
)

new_local_repository(
    name = "local",
    path = "/usr",
    build_file = "third_party/local/BUILD",
)
# External Dependencies
http_archive(
    name = "glog",
    sha256 = "6fc352c434018b11ad312cd3b56be3597b4c6b88480f7bd4e18b3a3b2cf961aa",
    strip_prefix = "glog-3ba8976592274bc1f907c402ce22558011d6fc5e",
    urls = [
        "https://github.com/google/glog/archive/3ba8976592274bc1f907c402ce22558011d6fc5e.tar.gz",
    ],
    build_file_content = """
licenses(['notice'])
exports_files(['CMakeLists.txt'])
load(':bazel/glog.bzl', 'glog_library')
glog_library(with_gflags=0)
""",
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

load("@coral_crosstool//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool", additional_system_include_directories=["//docker/include"])
