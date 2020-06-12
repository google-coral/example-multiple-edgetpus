cc_library(
    name = "gstreamer",
    srcs = [
        "usr/lib/x86_64-linux-gnu/libgstreamer-1.0.so",
    ],
    hdrs = glob(["usr/include/gstreamer-1.0/gst/*.h"]),
    includes = ["usr/include/gstreamer-1.0"],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)
