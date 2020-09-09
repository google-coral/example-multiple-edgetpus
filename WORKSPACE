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

TENSORFLOW_COMMIT = "d855adfc5a0195788bf5f92c3c7352e638aa1109"
TENSORFLOW_SHA256 = "b8a691dbea2bb028fa8f7ce407b70ad236dae0a8705c8010dc7bad8af7e93bac"

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz"],
) 

http_archive(
    name = "glog",
    sha256 = "835888ec47ee8065b3098f3ec4373717d641954970f009833ed6d466c397409a",
    strip_prefix = "glog-41f4bf9cbc3e8995d628b459f6a239df43c2b84a",
    urls = [
        "https://github.com/google/glog/archive/41f4bf9cbc3e8995d628b459f6a239df43c2b84a.tar.gz",
    ],
    build_file_content = """
licenses(['notice'])
exports_files(['CMakeLists.txt'])
load(':bazel/glog.bzl', 'glog_library')
glog_library(with_gflags=0)
""",
)

http_archive(
    name = "org_tensorflow",
    sha256 = TENSORFLOW_SHA256,
    strip_prefix = "tensorflow-" + TENSORFLOW_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/" + TENSORFLOW_COMMIT + ".tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name = "org_tensorflow")

http_archive(
    name = "edgetpu",
    sha256 = "3675945e7e0ce7b3f182883f321fa39f64a475a3e50a1a922fd6ef8a7563d6bc",
    strip_prefix = "edgetpu-285cacf0a879e3e343abd38c33eb622ae253529c",
    urls = [
        "https://github.com/google-coral/edgetpu/archive/285cacf0a879e3e343abd38c33eb622ae253529c.tar.gz",
    ]
)

new_local_repository(
    name = "libedgetpu",
    path = "libedgetpu",
    build_file = "libedgetpu/BUILD"
)

new_local_repository(
    name = "local",
    path = "/usr",
    build_file = "third_party/local/BUILD",
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
cc_crosstool(name = "crosstool")
