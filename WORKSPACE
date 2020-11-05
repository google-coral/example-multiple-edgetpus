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

TENSORFLOW_COMMIT = "48c3bae94a8b324525b45f157d638dfd4e8c3be1"
# Command to calculate: curl -L <FILE-URL> | sha256sum | awk '{print $1}'
TENSORFLOW_SHA256 = "363420a67b4cfa271cd21e5c8fac0d7d91b18b02180671c3f943c887122499d8"

# These values come from the Tensorflow workspace. If the TF commit is updated,
# these should be updated to match.
IO_BAZEL_RULES_CLOSURE_COMMIT = "308b05b2419edb5c8ee0471b67a40403df940149"
IO_BAZEL_RULES_CLOSURE_SHA256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9"

CORAL_CROSSTOOL_COMMIT = "142e930ac6bf1295ff3ba7ba2b5b6324dfb42839"
CORAL_CROSSTOOL_SHA256 = "088ef98b19a45d7224be13636487e3af57b1564880b67df7be8b3b7eee4a1bfc"

# Configure libedgetpu and downstream libraries (TF and Crosstool).
http_archive(
  name = "libedgetpu",
  sha256 = "d76d18c5a96758dd620057028cdd4e129bd885480087a5c7334070bba3797e58",
  strip_prefix = "libedgetpu-14eee1a076aa1af7ec1ae3c752be79ae2604a708",
  urls = [
    "https://github.com/google-coral/libedgetpu/archive/14eee1a076aa1af7ec1ae3c752be79ae2604a708.tar.gz"
  ],
)

load("@libedgetpu//:workspace.bzl", "libedgetpu_dependencies")
libedgetpu_dependencies(TENSORFLOW_COMMIT, TENSORFLOW_SHA256,
                        IO_BAZEL_RULES_CLOSURE_COMMIT,IO_BAZEL_RULES_CLOSURE_SHA256,
                        CORAL_CROSSTOOL_COMMIT,CORAL_CROSSTOOL_SHA256)

http_archive(
  name = "libcoral",
  sha256 = "ec0e1dfd4412b3c32f7ceac7b4507d0d084173229ef17f692a4ce95342731f58",
  strip_prefix = "libcoral-982426546dfa10128376d0c24fd8a8b161daac97",
  urls = [
    "https://github.com/google-coral/libcoral/archive/982426546dfa10128376d0c24fd8a8b161daac97.tar.gz"
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

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name = "org_tensorflow")

load("@coral_crosstool//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool", additional_system_include_directories=["//docker/include"])
