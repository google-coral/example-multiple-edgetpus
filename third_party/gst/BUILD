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

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "gstreamer",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libgstreamer-1.0.so",
    ],
    hdrs = glob(["usr/include/gstreamer-1.0/gst/*.h"]),
    includes = ["usr/include/gstreamer-1.0"],
    deps = [
        ":glib",
    ],
)

cc_library(
    name = "glib",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libgio-2.0.so",
        "usr/lib/x86_64-linux-gnu/libglib-2.0.so",
        "usr/lib/x86_64-linux-gnu/libgmodule-2.0.so",
        "usr/lib/x86_64-linux-gnu/libgobject-2.0.so",
        "usr/lib/x86_64-linux-gnu/libgthread-2.0.so",
    ],
    hdrs = glob([
        "usr/include/glib-2.0/*.h",
        "usr/include/glib-2.0/*/*.h",
        "usr/lib/x86_64-linux-gnu/glib-2.0/include/*.h",
        "usr/include/glib-2.0/glib/deprecated/*.h",
    ]),
    includes = [
        "usr/include/glib-2.0",
        "usr/include/glib-2.0/glib/deprecated",
        "usr/lib/x86_64-linux-gnu/glib-2.0/include",
    ],
)
