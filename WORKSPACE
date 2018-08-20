# ===== absl =====
git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    commit = "ad5c960b2eb914881d1ceba0e996a0a8f3f6ca59",
)

# ===== gtest =====
new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.8.0.zip",
    sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
    build_file = "gtest.BUILD",
    strip_prefix = "googletest-release-1.8.0",
)

# ===== eigen =====

new_http_archive(
    name = "eigen",
    build_file = "eigen.BUILD",
    strip_prefix = "eigen-eigen-5a0156e40feb",
    urls = ["https://bitbucket.org/eigen/eigen/get/3.3.4.zip"],
)

# ===== benchmark =====

new_git_repository(
    name = "com_google_benchmark",
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.1.0",
    build_file = "benchmark.BUILD",
)

# ===== gflags, required by glog =====

git_repository(
    name = "com_github_gflags_gflags",
    remote = "https://github.com/gflags/gflags.git",
    tag = "v2.2.0",
)

# ===== glog =====

new_http_archive(
    name = "com_github_glog_glog",
    url = "https://github.com/google/glog/archive/v0.3.5.zip",
    strip_prefix = "glog-0.3.5",
    build_file = "glog.BUILD",
)

# ===== protobuf =====
# LICENSE: The Apache Software License, Version 2.0
# proto_library rules implicitly depend on @com_google_protobuf//:protoc
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-master",
    urls = ["https://github.com/google/protobuf/archive/master.zip"],
)
