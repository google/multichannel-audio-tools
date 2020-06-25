load(":c.bzl", "c_library", "c_test")

package(default_visibility = ["//visibility:public"])

c_library(
    name = "logging",
    hdrs = ["logging.h"],
)

c_library(
    name = "math_constants",
    hdrs = ["math_constants.h"],
)

c_library(
    name = "number_util",
    srcs = ["number_util.c"],
    hdrs = ["number_util.h"],
)

c_test(
    name = "number_util_test",
    srcs = ["number_util_test.c"],
    deps = [
        ":logging",
        ":math_constants",
        ":number_util",
    ],
)
