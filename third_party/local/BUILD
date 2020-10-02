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

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "armv7a",
    values = {"cpu": "armv7a"},
)

config_setting(
    name = "k8",
    values = {"cpu": "k8"},
)

UNSUPPORTED_CPU_ERROR = "Unsupported target CPU"

cc_library(
    name = "gstreamer",
    srcs = select(
        {
            ":aarch64": glob([
                        "lib/aarch64-linux-gnu/libgstbase-1.0.so",
                        "lib/aarch64-linux-gnu/libgstreamer-1.0.so",
                        "lib/aarch64-linux-gnu/libgstallocators-1.0.so",
                        "lib/aarch64-linux-gnu/libgstapp-1.0.so",
                        "lib/aarch64-linux-gnu/libgstvideo-1.0.so",
                        "lib/aarch64-linux-gnu/libgstgl-1.0.so",
                        ]),
            ":armv7a": glob([
                        "lib/arm-linux-gnueabihf/libgstbase-1.0.so",
                        "lib/arm-linux-gnueabihf/libgstreamer-1.0.so",
                        "lib/arm-linux-gnueabihf/libgstallocators-1.0.so",
                        "lib/arm-linux-gnueabihf/libgstapp-1.0.so",
                        "lib/arm-linux-gnueabihf/libgstvideo-1.0.so",
                        "lib/arm-linux-gnueabihf/libgstgl-1.0.so",
                        ]),
            ":k8": glob([
                        "lib/x86_64-linux-gnu/libgstbase-1.0.so",
                        "lib/x86_64-linux-gnu/libgstreamer-1.0.so",
                        "lib/x86_64-linux-gnu/libgstallocators-1.0.so",
                        "lib/x86_64-linux-gnu/libgstapp-1.0.so",
                        "lib/x86_64-linux-gnu/libgstvideo-1.0.so",
                        "lib/x86_64-linux-gnu/libgstgl-1.0.so",
                        ]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "include/gstreamer-1.0/gst/*.h",
            "include/gstreamer-1.0/gst/app/*.h",
            "include/gstreamer-1.0/gst/base/*.h",
            "include/gstreamer-1.0/gst/gl/*.h",
            "include/gstreamer-1.0/gst/video/*.h",
        ],
    ) + select(
        {
            ":aarch64": glob(["lib/aarch64-linux-gnu/gstreamer-1.0/include/gst/gl/*.h"]),
            ":armv7a": glob(["lib/arm-linux-gnueabihf/gstreamer-1.0/include/gst/gl/*.h"]),
            ":k8": glob(["lib/x86_64-linux-gnu/gstreamer-1.0/include/gst/gl/*.h"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    includes = ["include/gstreamer-1.0"] + select(
        {
            ":aarch64": ["lib/aarch64-linux-gnu/gstreamer-1.0/include"],
            ":armv7a": ["lib/arm-linux-gnueabihf/gstreamer-1.0/include"],
            ":k8": ["lib/x86_64-linux-gnu/gstreamer-1.0/include"],
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    linkstatic = 0,
    deps = [
        ":glib",
    ],
)

cc_library(
    name = "gstallocators",
    srcs = select(
        {
            ":aarch64": glob(["lib/aarch64-linux-gnu/libgstallocators-1.0.so*"]),
            ":armv7a": glob(["lib/arm-linux-gnueabihf/libgstallocators-1.0.so*"]),
            ":k8": glob(["lib/x86_64-linux-gnu/libgstallocators-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "include/gstreamer-1.0/gst/allocators/*.h",
        ],
    ),
    includes = ["include/gstreamer-1.0"],
    linkstatic = 0,
    deps = [
        ":gstreamer",
    ],
)

cc_library(
    name = "glib",
    srcs = select(
        {
            ":aarch64": glob([
                "lib/aarch64-linux-gnu/libgobject-2.0.so.0*",
                "lib/aarch64-linux-gnu/libgobject-2.0.so.0*",
                "lib/aarch64-linux-gnu/libglib-2.0.so.0*",
                "lib/aarch64-linux-gnu/libglib-2.0.so.0*",
            ]),
            ":armv7a": glob([
                "lib/arm-linux-gnueabihf/libgobject-2.0.so.0*",
                "lib/arm-linux-gnueabihf/libgobject-2.0.so.0*",
                "lib/arm-linux-gnueabihf/libglib-2.0.so.0*",
                "lib/arm-linux-gnueabihf/libglib-2.0.so.0*",
            ]),
            ":k8": glob([
                "lib/x86_64-linux-gnu/libgobject-2.0.so.0*",
                "lib/x86_64-linux-gnu/libgobject-2.0.so.0*",
                "lib/x86_64-linux-gnu/libglib-2.0.so.0*",
                "lib/x86_64-linux-gnu/libglib-2.0.so.0*",
            ]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "include/glib-2.0/*.h",
            "include/glib-2.0/*/*.h",
            "include/glib-2.0/glib/deprecated/*.h",
            "include/glib-2.0/gobject/*.h",
        ],
    ) + select(
        {
            ":aarch64": glob(["lib/aarch64-linux-gnu/glib-2.0/include/*.h"]),
            ":armv7a": glob(["lib/arm-linux-gnueabihf/glib-2.0/include/*.h"]),
            ":k8": glob(["lib/x86_64-linux-gnu/glib-2.0/include/*.h"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    includes = ["include/glib-2.0"] + select(
        {
            ":aarch64": ["lib/aarch64-linux-gnu/glib-2.0/include"],
            ":armv7a": ["lib/arm-linux-gnueabihf/glib-2.0/include"],
            ":k8": ["lib/x86_64-linux-gnu/glib-2.0/include"],
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    linkstatic = 0,
)

# sudo apt-get install libgtk-3-dev
cc_library(
    name = "gtk",
    srcs = select(
        {
            ":aarch64": glob([
                    "lib/aarch64-linux-gnu/libgdk-3.so",
                    "lib/aarch64-linux-gnu/libgtk-3.so",
            ]),
            ":armv7a": glob([
                    "lib/arm-linux-gnu/libgdk-3.so",
                    "lib/arm-linux-gnu/libgtk-3.so",
            ]),
            ":k8": glob([
                    "lib/x86_64-linux-gnu/libgdk-3.so",
                    "lib/x86_64-linux-gnu/libgtk-3.so",
            ]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
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
    srcs = select(
        {
            ":aarch64": glob(["lib/aarch64-linux-gnu/libpango-1.0.so"]),
            ":armv7a": glob(["lib/arm-linux-gnu/libpango-1.0.so"]),
            ":k8": glob(["lib/x86_64-linux-gnu/libpango-1.0.so"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
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
    srcs = select(
        {
            ":aarch64": glob(["lib/aarch64-linux-gnu/libcairo.so"]),
            ":armv7a": glob(["lib/arm-linux-gnu/libcairo.so"]),
            ":k8": glob(["lib/x86_64-linux-gnu/libcairo.so"]),
        },
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
    srcs = select(
        {
            ":aarch64": glob(["lib/aarch64-linux-gnu/libgbm.so"]),
            ":armv7a": glob(["lib/arm-linux-gnu/libgbm.so"]),
            ":k8": glob(["lib/x86_64-linux-gnu/libgbm.so"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
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
    srcs = select(
        {
            ":aarch64": glob(["lib/aarch64-linux-gnu/libdrm.so"]),
            ":armv7a": glob(["lib/arm-linux-gnu/libdrm.so"]),
            ":k8": glob(["lib/x86_64-linux-gnu/libdrm.so"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
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
