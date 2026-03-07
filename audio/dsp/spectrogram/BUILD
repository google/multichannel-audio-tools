# Copyright 2020-2026 Google LLC
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "inverse_spectrogram",
    srcs = ["inverse_spectrogram.cc"],
    hdrs = ["inverse_spectrogram.h"],
    deps = [
        "//audio/dsp:number_util",
        "//third_party/eigen3:eigen3",
        "//third_party/fft2d",
        "@com_github_glog_glog//:glog",
    ],
)

cc_library(
    name = "spectrogram",
    srcs = ["spectrogram.cc"],
    hdrs = ["spectrogram.h"],
    deps = [
        "//audio/dsp:number_util",
        "//audio/dsp:porting",
        "//audio/dsp:window_functions",
        "//third_party/fft2d:fft2d",
        "@com_github_glog_glog//:glog",
    ],
)
