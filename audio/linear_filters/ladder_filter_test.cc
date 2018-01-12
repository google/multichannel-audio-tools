/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "audio/linear_filters/ladder_filter.h"

#include <cfloat>
#include <cmath>
#include <complex>
#include <vector>

#include "audio/dsp/testing_util.h"
#include "audio/linear_filters/biquad_filter.h"
#include "audio/linear_filters/biquad_filter_coefficients.h"
#include "audio/linear_filters/biquad_filter_design.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#include "third_party/eigen3/Eigen/Core"

#include "audio/dsp/porting.h"  // auto-added.


namespace linear_filters {
namespace {

using ::audio_dsp::EigenArrayNear;
using ::Eigen::Array;
using ::Eigen::ArrayXf;
using ::Eigen::ArrayXXf;
using ::Eigen::Dynamic;
using ::std::complex;
using ::std::vector;
using ::util::format::StringF;

// Get the name of Type as a string.
template <typename Type>
string GetTypeName() {
  return util::Demangle(typeid(Type).name());
}

template <typename ScalarType>
struct Tolerance { constexpr static float value = 0; };

template <> struct Tolerance<float> { constexpr static float value = 1e-4; };
template <> struct Tolerance<double> { constexpr static double value = 1e-8; };

template <typename T>
class LadderFilterScalarTypedTest : public ::testing::Test {};

typedef ::testing::Types<float, double> ScalarTypes;
TYPED_TEST_CASE(LadderFilterScalarTypedTest, ScalarTypes);

// Tests to make sure LadderFilter and BiquadFilter produce the same result.
TYPED_TEST(LadderFilterScalarTypedTest, MatchesBiquadFilter) {
  LadderFilter<TypeParam> ladder;
  BiquadFilter<TypeParam> biquad;

  auto coeffs = LowpassBiquadFilterCoefficients(48000, 4000, 0.707);
  ladder.InitFromTransferFunction(1, coeffs.b, coeffs.a);
  biquad.Init(1, coeffs);

  std::mt19937 rng(0 /* seed */);
  std::normal_distribution<TypeParam> dist(0, 1);
  for (int i = 0; i < 512; ++i) {
    TypeParam sample = dist(rng);
    TypeParam ladder_output;
    TypeParam biquad_output;
    ladder.ProcessSample(sample, &ladder_output);
    biquad.ProcessSample(sample, &biquad_output);
    ASSERT_NEAR(ladder_output, biquad_output, Tolerance<TypeParam>::value);
  }
}

TYPED_TEST(LadderFilterScalarTypedTest, MatchesMultipleBiquadFilters) {
  LadderFilter<TypeParam> ladder;
  BiquadFilter<TypeParam> biquad1;
  BiquadFilter<TypeParam> biquad2;

  auto coeffs1 = LowpassBiquadFilterCoefficients(48000, 4000, 0.707);
  auto coeffs2 = HighpassBiquadFilterCoefficients(48000, 200, 0.707);
  BiquadFilterCascadeCoefficients coeffs =
      BiquadFilterCascadeCoefficients({coeffs1, coeffs2});
  vector<double> k;
  vector<double> v;
  coeffs.AsLadderFilterCoefficients(&k, &v);
  ladder.InitFromLadderCoeffs(1, k, v);
  biquad1.Init(1, coeffs1);
  biquad2.Init(1, coeffs2);

  std::mt19937 rng(0 /* seed */);
  std::normal_distribution<TypeParam> dist(0, 1);
  for (int i = 0; i < 512; ++i) {
    TypeParam sample = dist(rng);
    TypeParam ladder_output;
    TypeParam biquad_output;
    ladder.ProcessSample(sample, &ladder_output);
    biquad1.ProcessSample(sample, &biquad_output);
    biquad2.ProcessSample(biquad_output, &biquad_output);
    ASSERT_NEAR(ladder_output, biquad_output, Tolerance<TypeParam>::value);
  }
}

// A sanity check to make sure that InitFromLadderCoeffs doesn't break.
TYPED_TEST(LadderFilterScalarTypedTest, LadderInit) {
  LadderFilter<TypeParam> ladder;

  ladder.InitFromLadderCoeffs(1, {0.3, 0.3, -0.2}, {0.1, 2.3, -5.2, 1.0});

  std::mt19937 rng(0 /* seed */);
  std::normal_distribution<TypeParam> dist(0, 1);
  for (int i = 0; i < 3; ++i) {
    TypeParam sample = dist(rng);
    TypeParam ladder_output;
    ladder.ProcessSample(sample, &ladder_output);
  }
}

template <typename T>
class LadderFilterMultichannelTypedTest : public ::testing::Test {};

typedef ::testing::Types<
    // With scalar SampleType.
    float,
    double,
    complex<float>,
    complex<double>,
    // With SampleType = Eigen::Array* with dynamic number of channels.
    Eigen::ArrayXf,
    Eigen::ArrayXd,
    Eigen::ArrayXcf,
    Eigen::ArrayXcd,
    // With SampleType = Eigen::Vector* with dynamic number of channels.
    Eigen::VectorXf,
    Eigen::VectorXcf,
    // With SampleType with fixed number of channels.
    Eigen::Array3f,
    Eigen::Array3cf,
    Eigen::Vector3f,
    Eigen::Vector3cf
    >
    SampleTypes;
TYPED_TEST_CASE(LadderFilterMultichannelTypedTest, SampleTypes);

// Test LadderFilter<SampleType> against BiquadFilter<SampleType> for different
// template args.
TYPED_TEST(LadderFilterMultichannelTypedTest, LadderFilter) {
  using SampleType = TypeParam;
  SCOPED_TRACE(
      testing::Message() << "SampleType: " << GetTypeName<SampleType>());

  constexpr int kNumSamples = 20;
  using BiquadFilterType = BiquadFilter<SampleType>;
  using LadderFilterType = LadderFilter<SampleType>;
  const int kNumChannelsAtCompileTime =
      LadderFilterType::kNumChannelsAtCompileTime;
  using ScalarType = typename LadderFilterType::ScalarType;
  using BlockOfSamples =
      typename Eigen::Array<ScalarType, kNumChannelsAtCompileTime, Dynamic>;
  BiquadFilterType biquad;
  LadderFilterType ladder;

  for (int num_channels : {1, 4, 7}) {
    if (kNumChannelsAtCompileTime != Dynamic &&
        kNumChannelsAtCompileTime != num_channels) {
      continue;  // Skip if SampleType is incompatible with num_channels.
    }
    SCOPED_TRACE("num_channels: " + testing::PrintToString(num_channels));

    auto coeffs = LowpassBiquadFilterCoefficients(48000, 4000, 0.707);
    ladder.InitFromTransferFunction(num_channels, coeffs.b, coeffs.a);
    biquad.Init(num_channels, coeffs);

    BlockOfSamples input = BlockOfSamples::Random(num_channels, kNumSamples);
    BlockOfSamples ladder_output;
    ladder.ProcessBlock(input, &ladder_output);
    BlockOfSamples biquad_output;
    biquad.ProcessBlock(input, &biquad_output);

    EXPECT_THAT(ladder_output, EigenArrayNear(biquad_output, 1e-5));
  }
}

template <typename T>
class LadderFilterMultichannelScalarTypedTest : public ::testing::Test {};

typedef ::testing::Types<float, double> ScalarSampleTypes;
TYPED_TEST_CASE(LadderFilterMultichannelScalarTypedTest, ScalarSampleTypes);

// We will smooth between the coefficients of two eighth order filters
// with very different looking transfer functions. This should be a pretty
// good stress test.
//
// This test should not give different results for different numbers of
// channels or complex scalar types. All the terrifying math is happening in the
// coefficient smoothing.
TYPED_TEST(LadderFilterMultichannelScalarTypedTest,
           LadderFilterCoefficientStressTest) {
  using SampleType = TypeParam;
  SCOPED_TRACE(
      testing::Message() << "SampleType: " << GetTypeName<SampleType>());

  using LadderFilterType = LadderFilter<SampleType>;
  const int kNumChannelsAtCompileTime =
      LadderFilterType::kNumChannelsAtCompileTime;
  using ScalarType = typename LadderFilterType::ScalarType;
  using BlockOfSamples =
      typename Eigen::Array<ScalarType, kNumChannelsAtCompileTime, Dynamic>;
  LadderFilterType ladder;

  vector<double> bandpass_k;
  vector<double> bandpass_v;
  ButterworthFilterDesign(4).BandpassCoefficients(48000, 3000, 4000)
      .AsLadderFilterCoefficients(&bandpass_k, &bandpass_v);
  vector<double> bandstop_k;
  vector<double> bandstop_v;
  // This filter is pretty numerically dicey. We're putting lots of poles near
  // s = 0 to achieve the very low 12Hz cutoff. If we can smooth to
  // and from this without issues, everything else should be pretty safe.
  //
  // Note that for lower cutoffs, higher order, or 32-bit precision, the
  // the coefficients can still leave the range [-1, 1]. For these reason, we
  // enforce a stability check in the smoothing code.
  ButterworthFilterDesign(4).BandstopCoefficients(48000, 12, 2000)
      .AsLadderFilterCoefficients(&bandstop_k, &bandstop_v);

  constexpr int num_channels = 1;

  if (kNumChannelsAtCompileTime != Dynamic &&
      kNumChannelsAtCompileTime != num_channels) {
    return;  // Skip if SampleType is incompatible with num_channels.
  }
  ladder.InitFromLadderCoeffs(num_channels, bandstop_k, bandstop_v);

  bool filter_switch = true;
  // These numbers are mostly random, but an attempt was made to give varied
  // block sizes to each of the two filter coefficient sets.
  int sample_count = 0;
  for (int num_samples : {10, 300, 4, 1203, 8000, 13, 13000,
                          20, 433, 1234, 10000, 100}) {
    SCOPED_TRACE(StringF("Filter has %d channels and has processed %d "
                         "samples.", num_channels, sample_count));
    BlockOfSamples input = BlockOfSamples::Random(num_channels, num_samples);
    BlockOfSamples ladder_output;

    filter_switch = !filter_switch;
    if (filter_switch) {
      ladder.ChangeLadderCoeffs(bandpass_k, bandpass_v);
    } else {
      ladder.ChangeLadderCoeffs(bandstop_k, bandstop_v);
    }
    ladder.ProcessBlock(input, &ladder_output);
    // The samples don't turn to nan at any point. This can happen if the
    // smoothing filter overshoots causing the reflection coefficients to
    // get smaller than -1, which in turn cause the scattering coefficients
    // to go nan during the sqrt computation.
    ASSERT_FALSE(isnan(std::abs(ladder_output.sum())));
    sample_count += num_samples;
  }
}

template <bool kChangeCoefficients>
void BM_LadderFilterScalarFloat(benchmark::State& state) {
  constexpr int kSamplePerBlock = 1000;
  srand(0 /* seed */);
  ArrayXf input = ArrayXf::Random(kSamplePerBlock);
  ArrayXf output(kSamplePerBlock);
  LadderFilter<float> filter;
  // Equivalent to a single biquad stage.
  vector<double> k = {-0.2, 0.9};
  vector<double> v = {0.5, -0.2, 0.1};
  filter.InitFromLadderCoeffs(1, k, v);

  while (state.KeepRunning()) {
    if (kChangeCoefficients) {
      // There is no check internally to prevent smoothing when the
      // coefficients don't *actually* change.
      filter.ChangeLadderCoeffs(k, v);
    }
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
// Old-style template benchmarking needed for open sourcing. External
// google/benchmark repo doesn't have functionality from cl/118676616 enabling
// BENCHMARK(TemplatedFunction<2>) syntax.

// No coefficient smoothing.
BENCHMARK_TEMPLATE(BM_LadderFilterScalarFloat, false);
// Also test with coefficient smoothing.
BENCHMARK_TEMPLATE(BM_LadderFilterScalarFloat, true);

template <bool kChangeCoefficients>
void BM_LadderFilterArrayXf(benchmark::State& state) {
  const int num_channels = state.range(0);
  constexpr int kSamplePerBlock = 1000;
  srand(0 /* seed */);
  ArrayXXf input = ArrayXXf::Random(num_channels, kSamplePerBlock);
  ArrayXXf output(num_channels, kSamplePerBlock);
  LadderFilter<ArrayXf> filter;
  vector<double> k = {-0.2, 0.9};
  vector<double> v = {0.5, -0.2, 0.1};
  filter.InitFromLadderCoeffs(num_channels, k, v);

  while (state.KeepRunning()) {
    if (kChangeCoefficients) {
      // There is no check internally to prevent smoothing when the
      // coefficients don't *actually* change.
      filter.ChangeLadderCoeffs(k, v);
    }
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}

// No coefficient smoothing.
BENCHMARK_TEMPLATE(BM_LadderFilterArrayXf, false)->DenseRange(1, 10);
// Also test with smoothing.
BENCHMARK_TEMPLATE(BM_LadderFilterArrayXf, true)->DenseRange(1, 10);

template <int kNumChannels, bool kChangeCoefficients>
void BM_LadderFilterArrayNf(benchmark::State& state) {
  constexpr int kSamplePerBlock = 1000;
  srand(0 /* seed */);
  using ArrayNf = Eigen::Array<float, kNumChannels, 1>;
  using ArrayNXf = Eigen::Array<float, kNumChannels, Dynamic>;
  ArrayNXf input = ArrayNXf::Random(kNumChannels, kSamplePerBlock);
  ArrayNXf output(kNumChannels, kSamplePerBlock);
  LadderFilter<ArrayNf> filter;
  vector<double> k = {-0.2, 0.9};
  vector<double> v = {0.5, -0.2, 0.1};
  filter.InitFromLadderCoeffs(kNumChannels, k, v);

  while (state.KeepRunning()) {
    if (kChangeCoefficients) {
      // There is no check internally to prevent smoothing when the
      // coefficients don't *actually* change.
      filter.ChangeLadderCoeffs(k, v);
    }
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}

// No coefficient smoothing.
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 1, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 2, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 3, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 4, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 5, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 6, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 7, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 8, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 9, false);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 10, false);
// Also test with coefficient smoothing.
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 1, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 2, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 3, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 4, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 5, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 6, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 7, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 8, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 9, true);
BENCHMARK_TEMPLATE2(BM_LadderFilterArrayNf, 10, true);
}  // namespace
}  // namespace linear_filters

/*
Run on lpac2 (32 X 2600 MHz CPUs); 2017-03-18T00:23:02.130960951-07:00
CPU: Intel Sandybridge with HyperThreading (16 cores)
Benchmark                           Time(ns)        CPU(ns)     Iterations
--------------------------------------------------------------------------
// Without Smoothing
BM_LadderFilterScalarFloat<false>       9508           9495         740083
// With Smoothing
BM_LadderFilterScalarFloat<true>       21496          21471         325679
// Without Smoothing
BM_LadderFilterArrayXf<false>/1        42159          42106         165833
BM_LadderFilterArrayXf<false>/2        57080          57013         100000
BM_LadderFilterArrayXf<false>/3        61796          61725         100000
BM_LadderFilterArrayXf<false>/4        52066          51995         100000
BM_LadderFilterArrayXf<false>/5        88576          88466          79194
BM_LadderFilterArrayXf<false>/6        90095          89985          78313
BM_LadderFilterArrayXf<false>/7       102595         102464          68312
BM_LadderFilterArrayXf<false>/8        95392          95276          73926
BM_LadderFilterArrayXf<false>/9       116094         115953          59707
BM_LadderFilterArrayXf<false>/10      118117         117957          59400
// With Smoothing
BM_LadderFilterArrayXf<true>/1         55712          55642         100000
BM_LadderFilterArrayXf<true>/2         68701          68614         100000
BM_LadderFilterArrayXf<true>/3         73837          73745          93965
BM_LadderFilterArrayXf<true>/4         64988          64899         100000
BM_LadderFilterArrayXf<true>/5        101415         101293          68447
BM_LadderFilterArrayXf<true>/6        103407         103265          67532
BM_LadderFilterArrayXf<true>/7        115700         115553          60688
BM_LadderFilterArrayXf<true>/8        107713         107568          65265
BM_LadderFilterArrayXf<true>/9        129464         129290          54210
BM_LadderFilterArrayXf<true>/10       131808         131624          53141
// Without Smoothing
BM_LadderFilterArrayNf<1, false>       10709          10696         654390
BM_LadderFilterArrayNf<2, false>       15208          15186         462941
BM_LadderFilterArrayNf<3, false>       20715          20687         340574
BM_LadderFilterArrayNf<4, false>       11568          11534         621362
BM_LadderFilterArrayNf<5, false>       31342          31220         216525
BM_LadderFilterArrayNf<6, false>       34849          34803         201868
BM_LadderFilterArrayNf<7, false>       39080          39031         179947
BM_LadderFilterArrayNf<8, false>       44369          44311         157545
BM_LadderFilterArrayNf<9, false>       72032          71941          97352
BM_LadderFilterArrayNf<10, false>      76978          76875          91614
// With Smoothing
BM_LadderFilterArrayNf<1, true>        22882          22854         307404
BM_LadderFilterArrayNf<2, true>        27208          27174         257384
BM_LadderFilterArrayNf<3, true>        32613          32575         215070
BM_LadderFilterArrayNf<4, true>        23405          23375         299919
BM_LadderFilterArrayNf<5, true>        41768          41715         167998
BM_LadderFilterArrayNf<6, true>        46440          46385         151150
BM_LadderFilterArrayNf<7, true>        51039          50973         100000
BM_LadderFilterArrayNf<8, true>        56126          56048         100000
BM_LadderFilterArrayNf<9, true>        84887          84774          82946
BM_LadderFilterArrayNf<10, true>       89664          89553          78841
*/
