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
    name = "gstreamer",
    srcs = glob(
        [
            "lib/aarch64-linux-gnu/libgstbase-1.0.so",
            "lib/aarch64-linux-gnu/libgstreamer-1.0.so",
            "lib/aarch64-linux-gnu/libgstallocators-1.0.so",
            "lib/aarch64-linux-gnu/libgstapp-1.0.so",
            "lib/aarch64-linux-gnu/libgstvideo-1.0.so",
            "lib/aarch64-linux-gnu/libgstgl-1.0.so",
        ],
    ),
    hdrs = glob(
        [
            "include/gstreamer-1.0/gst/*.h",
            "include/gstreamer-1.0/gst/app/*.h",
            "include/gstreamer-1.0/gst/base/*.h",
            "include/gstreamer-1.0/gst/gl/*.h",
            "lib/aarch64-linux-gnu/gstreamer-1.0/include/gst/gl/*.h",
            "include/gstreamer-1.0/gst/video/*.h",
        ]),
    includes = ["include/gstreamer-1.0/", "lib/aarch64-linux-gnu/gstreamer-1.0/include"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
    deps = [
        ":glib",
    ],
)

cc_library(
    name = "glib",
    srcs = glob(
        [
            "lib/aarch64-linux-gnu/libglib-2.0.so",
            "lib/aarch64-linux-gnu/libgobject-2.0.so",
        ],
    ),
    hdrs = glob(
        [
            "include/glib-2.0/*.h",
            "include/glib-2.0/*/*.h",
            "include/glib-2.0/glib/deprecated/*.h",
            "include/glib-2.0/gobject/*.h",
            "lib/aarch64-linux-gnu/glib-2.0/include/*",
        ]),
    includes = ["include/glib-2.0", "lib/aarch64-linux-gnu/glib-2.0/include"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
)

# sudo apt-get install libgtk-3-dev
cc_library(
    name = "gtk",
    srcs = glob(
        [
            "lib/aarch64-linux-gnu/libgdk-3.so",
            "lib/aarch64-linux-gnu/libgtk-3.so",
        ],
    ),
    hdrs = glob(
        [
            "include/gtk-3.0/*.h",
            "include/gtk-3.0/*/*.h",
            "include/gtk-3.0/*/*/*.h",
            "include/gtk-3.0/*/deprecated/*.h",
        ]),
    includes = ["include/gtk-3.0"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
    deps = [
        ":atk",
        ":gdk-pixbuf",
        ":cairo",
        ":glib",
        ":pango",
    ],
)

cc_library(
    name = "pango",
    srcs = glob(
        [
            "lib/aarch64-linux-gnu/libpango-1.0.so",
        ],
    ),
    hdrs = glob(
        [
            "include/pango-1.0/*/*.h",
        ]),
    includes = ["include/pango-1.0"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cairo",
    srcs = glob(
        [
            "lib/aarch64-linux-gnu/libcairo.so",
        ],
    ),
    hdrs = glob(
        [
            "include/cairo/*.h",
        ]),
    includes = ["include/cairo"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gdk-pixbuf",
    srcs = glob(
        [
        ],
    ),
    hdrs = glob(
        [
            "include/gdk-pixbuf-2.0/*.h",
            "include/gdk-pixbuf-2.0/*/*.h",
        ]),
    includes = ["include/gdk-pixbuf-2.0"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "atk",
    srcs = glob(
        [
        ],
    ),
    hdrs = glob(
        [
            "include/atk-1.0/*.h",
            "include/atk-1.0/*/*.h",
        ]),
    includes = ["include/atk-1.0"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gbm",
    srcs = glob(
        [
            "lib/aarch64-linux-gnu/libgbm.so",
        ],
    ),
    hdrs = glob(
        [
            "include/gbm.h",
        ]),
    includes = ["include"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "drm",
    srcs = glob(
        [
            "lib/aarch64-linux-gnu/libdrm.so",
        ],
    ),
    hdrs = glob(
        [
            "include/xf86drm.h",
            "include/libdrm/*.h"
        ]),
    includes = ["include", "include/libdrm"],
    linkstatic = 0,
    visibility = ["//visibility:public"],
)