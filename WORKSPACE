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

new_local_repository(
    name = "libedgetpu",
    path = "libedgetpu",
    build_file = "libedgetpu/BUILD"
)

new_local_repository(
    name = "local_gst",
    path = "/",
    build_file = "third_party/gst/BUILD",
)

http_archive(
    name = "coral_crosstool",
    sha256 = "088ef98b19a45d7224be13636487e3af57b1564880b67df7be8b3b7eee4a1bfc",
    strip_prefix = "crosstool-142e930ac6bf1295ff3ba7ba2b5b6324dfb42839",
    urls = [
        "https://github.com/google-coral/crosstool/archive/142e930ac6bf1295ff3ba7ba2b5b6324dfb42839.tar.gz",
    ],
)

load("@coral_crosstool//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool", additional_system_include_directories=["//include"])
