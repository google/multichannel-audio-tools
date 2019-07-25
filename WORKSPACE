load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ===== absl =====
git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    commit = "ad5c960b2eb914881d1ceba0e996a0a8f3f6ca59",
)

# ===== fft2 =====
http_archive(
    name = "fft2d",
    build_file = "fft2d.BUILD",
    sha256 = "ada7e99087c4ed477bfdf11413f2ba8db8a840ba9bbf8ac94f4f3972e2a7cec9",
    urls = [
        "http://www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz",
    ],
)

# ===== gtest =====
git_repository(
    name = "gtest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.8.1",
)

# ===== eigen =====
http_archive(
    name = "eigen_archive",
    build_file = "eigen.BUILD",
    sha256 = "f3d69ac773ecaf3602cb940040390d4e71a501bb145ca9e01ce5464cf6d4eb68",
    strip_prefix = "eigen-eigen-049af2f56331",
    urls = [
        "http://mirror.tensorflow.org/bitbucket.org/eigen/eigen/get/049af2f56331.tar.gz",
        "https://bitbucket.org/eigen/eigen/get/049af2f56331.tar.gz",
    ],
)

# ===== benchmark =====
git_repository(
    name = "com_google_benchmark",
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.5.0",
)

# ===== gflags, required by glog =====
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "6e16c8bc91b1310a44f3965e616383dbda48f83e8c1eaa2370a215057b00cabe",
    strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
        "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
    ],
)

# ===== glog =====
http_archive(
    name = "com_github_glog_glog",
    url = "https://github.com/google/glog/archive/v0.3.5.zip",
    sha256 = "267103f8a1e9578978aa1dc256001e6529ef593e5aea38193d31c2872ee025e8",
    strip_prefix = "glog-0.3.5",
    build_file = "glog.BUILD",
)

# ===== protobuf =====
# LICENSE: The Apache Software License, Version 2.0
# proto_library rules implicitly depend on @com_google_protobuf//:protoc
#
# There seem to be some problems with the most recent bazel version and
# the protobuf libs, explained here.
# https://github.com/tensorflow/tensorflow/issues/25000
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-3.6.1.2",
    urls = ["https://github.com/google/protobuf/archive/v3.6.1.2.tar.gz"],
    sha256 = "2244b0308846bb22b4ff0bcc675e99290ff9f1115553ae9671eba1030af31bc0",
)
