cc_library(
    name = "benchmark",
    srcs = glob(["src/*.cc"]),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h",
    ]),
    copts = [
        "-DHAVE_POSIX_REGEX",
    ],
    includes = [
        ".",
        "include",
    ],
    linkstatic = 1,
    visibility = [
        "//visibility:public",
    ],
)
