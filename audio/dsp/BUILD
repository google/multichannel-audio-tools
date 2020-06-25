# Description:
#   DSP tools for processing audio.

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "porting",
    srcs = ["porting.cc"],
    # Files get moved into audio/dsp by copy.bara.sky.
    hdrs = ["porting.h"],
)

cc_library(
    name = "attack_release_envelope",
    srcs = ["attack_release_envelope.cc"],
    hdrs = ["attack_release_envelope.h"],
    deps = [
        ":porting",
        "//audio/linear_filters:biquad_filter",
        "//audio/linear_filters:biquad_filter_coefficients",
        "//audio/linear_filters:biquad_filter_design",
        "//audio/linear_filters:discretization",
    ],
)

cc_test(
    name = "attack_release_envelope_test",
    size = "small",
    srcs = ["attack_release_envelope_test.cc"],
    deps = [
        ":attack_release_envelope",
        "//audio/linear_filters:discretization",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "bessel_functions",
    srcs = ["bessel_functions.cc"],
    hdrs = ["bessel_functions.h"],
    deps = [
        ":porting",
        ":signal_vector_util",
        "@com_github_glog_glog//:glog",
    ],
)

cc_test(
    name = "bessel_functions_test",
    size = "small",
    srcs = ["bessel_functions_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":bessel_functions",
        ":porting",
        ":signal_vector_util",
        "@com_google_absl//absl/strings",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "batch_top_n",
    hdrs = ["batch_top_n.h"],
    deps = [
        ":porting",
        "@com_github_glog_glog//:glog",
    ],
)

cc_library(
    name = "decibels",
    hdrs = ["decibels.h"],
    deps = [
        ":porting",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "decibels_test",
    size = "small",
    srcs = ["decibels_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":decibels",
        ":porting",
        ":testing_util",
        "//third_party/eigen3",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "eigen_types",
    hdrs = ["eigen_types.h"],
    deps = [
        ":porting",
        "@com_github_glog_glog//:glog",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "eigen_types_test",
    size = "small",
    srcs = ["eigen_types_test.cc"],
    deps = [
        ":porting",
        ":eigen_types",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "elliptic_functions",
    srcs = ["elliptic_functions.cc"],
    hdrs = ["elliptic_functions.h"],
    deps = [
        ":porting",
        ":signal_vector_util",
        "@com_github_glog_glog//:glog",
    ],
)

cc_test(
    name = "elliptic_functions_test",
    size = "small",
    srcs = ["elliptic_functions_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":elliptic_functions",
        ":porting",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "envelope_detector",
    srcs = ["envelope_detector.cc"],
    hdrs = ["envelope_detector.h"],
    deps = [
        ":porting",
        ":resampler_rational_factor",
        "//audio/linear_filters:biquad_filter",
        "//audio/linear_filters:biquad_filter_design",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "envelope_detector_test",
    srcs = ["envelope_detector_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":envelope_detector",
        ":porting",
        ":testing_util",
        "//audio/linear_filters:biquad_filter_design",
        "//third_party/eigen3",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "fixed_delay_line",
    hdrs = ["fixed_delay_line.h"],
    deps = [
        ":porting",
        "@com_github_glog_glog//:glog",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "fixed_delay_line_test",
    srcs = ["fixed_delay_line_test.cc"],
    deps = [
        ":fixed_delay_line",
        ":porting",
        ":testing_util",
        "@gtest//:gtest_main",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "kiss_fft",
    srcs = ["kiss_fft.cc"],
    hdrs = ["kiss_fft.h"],
    deps = [
        ":porting",
        "@com_github_glog_glog//:glog",
        "//third_party/kissfft:kiss_fft_float",
    ],
)

cc_library(
    name = "kiss_fft_util",
    srcs = ["kiss_fft_util.cc"],
    hdrs = ["kiss_fft_util.h"],
    deps = [
        ":eigen_types",
        ":kiss_fft",
        ":porting",
        ":signal_vector_util",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "nelder_mead_searcher",
    hdrs = ["nelder_mead_searcher.h"],
    deps = [
        ":porting",
        "//third_party/eigen3",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "nelder_mead_searcher_test",
    srcs = ["nelder_mead_searcher_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":nelder_mead_searcher",
        ":porting",
        ":testing_util",
        "//third_party/eigen3",
        "@com_google_benchmark//:benchmark",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "number_util",
    srcs = ["number_util.cc"],
    hdrs = ["number_util.h"],
    deps = [
        ":porting",
        "//audio/dsp/portable:number_util",
        "@com_github_glog_glog//:glog",
    ],
)

cc_test(
    name = "number_util_test",
    size = "small",
    srcs = ["number_util_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":number_util",
        ":porting",
        ":testing_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "resampler",
    hdrs = ["resampler.h"],
    deps = [
        ":porting",
    ],
)

cc_library(
    name = "resampler_rational_factor",
    srcs = ["resampler_rational_factor.cc"],
    hdrs = ["resampler_rational_factor.h"],
    deps = [
        ":bessel_functions",
        ":number_util",
        ":porting",
        ":resampler",
        ":types",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "resampler_rational_factor_test",
    size = "large",
    srcs = ["resampler_rational_factor_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":porting",
        ":resampler_rational_factor",
        ":signal_vector_util",
        ":testing_util",
        ":types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_benchmark//:benchmark",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "signal_eigen_util",
    hdrs = ["signal_eigen_util.h"],
    deps = [
        ":batch_top_n",
        ":porting",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "signal_eigen_util_test",
    size = "small",
    srcs = ["signal_eigen_util_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":porting",
        ":signal_eigen_util",
        ":testing_util",
        "//third_party/eigen3",
        "@com_google_benchmark//:benchmark",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "signal_generator",
    hdrs = ["signal_generator.h"],
    srcs = ["signal_generator.cc"],
    deps = [
        ":porting",
        "@com_github_glog_glog//:glog",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "signal_generator_test",
    size = "small",
    srcs = ["signal_generator_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":porting",
        ":signal_generator",
        ":testing_util",
        ":types",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings:str_format",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "signal_vector_util",
    hdrs = ["signal_vector_util.h"],
    deps = [
        ":porting",
        ":types",
    ],
)

cc_library(
    name = "testing_util",
    testonly = 1,
    hdrs = ["testing_util.h"],
    deps = [
        ":porting",
        "//third_party/eigen3",
        "@com_github_glog_glog//:glog",
        "@com_google_absl//absl/types:span",
        "@gtest//:gtest",
    ],
)

cc_test(
    name = "testing_util_test",
    size = "small",
    srcs = ["testing_util_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":porting",
        ":testing_util",
        "@com_github_glog_glog//:glog",
        "@com_google_absl//absl/types:span",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        ":porting",
    ],
)

cc_test(
    name = "types_test",
    size = "small",
    srcs = ["types_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        ":porting",
        ":types",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "window_functions",
    srcs = ["window_functions.cc"],
    hdrs = ["window_functions.h"],
    deps = [
        ":bessel_functions",
        ":porting",
    ],
)
