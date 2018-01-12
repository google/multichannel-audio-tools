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

#include "audio/linear_filters/biquad_filter.h"

#include <complex>
#include <random>
#include <vector>

#include "audio/dsp/testing_util.h"
#include "audio/dsp/types.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#include "third_party/eigen3/Eigen/Core"

#include "audio/dsp/porting.h"  // auto-added.


namespace linear_filters {
namespace {

using ::Eigen::Array;
using ::Eigen::ArrayXf;
using ::Eigen::ArrayXXf;
using ::Eigen::Dynamic;
using ::Eigen::Map;
using ::audio_dsp::EigenArrayNear;
using ::std::complex;

// Get the name of Type as a string.
template <typename Type>
string GetTypeName() {
  return util::Demangle(typeid(Type).name());
}

template <typename Type1, typename Type2>
void TypesMatch() {
  // The preprocessor parses EXPECT_TRUE(std::is_same<Type1, Type2>::value)
  // incorrectly as a two-argument macro function because of the comma in
  // "Type1, Type2". So we write it in two lines.
  bool match = std::is_same<Type1, Type2>::value;
  EXPECT_TRUE(match) << "Types do not match "
      << GetTypeName<Type1>() << " vs. " << GetTypeName<Type2>() << ".";
}

// Simple biquad filter implementation in direct form 1 to compare against.
template <typename SampleType>
Array<SampleType, Dynamic, Dynamic> ReferenceBiquadFilter(
    const BiquadFilterCoefficients& coeffs,
    const Array<SampleType, Dynamic, Dynamic>& input) {
  CHECK_GE(input.size(), 2);

  Array<SampleType, Dynamic, Dynamic> output(input.rows(), input.cols());
  output.col(0) = coeffs.b[0] * input.col(0) / coeffs.a[0];
  output.col(1) = (coeffs.b[0] * input.col(1) + coeffs.b[1] * input.col(0) -
      coeffs.a[1] * output.col(0)) / coeffs.a[0];
  for (int n = 2; n < input.cols(); ++n) {
    output.col(n) =
        (coeffs.b[0] * input.col(n) + coeffs.b[1] * input.col(n - 1) +
         coeffs.b[2] * input.col(n - 2) -
         coeffs.a[1] * output.col(n - 1) - coeffs.a[2] * output.col(n - 2))
        .template cast<SampleType>() / coeffs.a[0];
  }
  return output;
}

// Simple test on a unit impulse.
TEST(BiquadFilterTest, IirImpulseResponse) {
  constexpr int kNumSamples = 20;
  constexpr float a = 0.95;
  constexpr float omega = (2 * M_PI) / 7;  // Period of 7 samples.
  const float cos_omega = cos(omega);
  const BiquadFilterCoefficients coeffs = {{1.0, -a * cos_omega, 0.0},
                                           {1.0, -2 * a * cos_omega, a * a}};
  // The input is a unit impulse delayed by 3 samples.
  ArrayXf input = ArrayXf::Zero(kNumSamples);
  input[3] = 1.0;

  ArrayXf output;
  BiquadFilter<float> filter;
  filter.Init(1, coeffs);
  filter.ProcessBlock(input, &output);

  // The impulse response of the filter is a^n cos(omega n).
  ArrayXf expected = ArrayXf::Zero(kNumSamples);
  for (int n = 3; n < kNumSamples; ++n) {
    expected[n] = pow(a, n - 3) * cos(omega * (n - 3));
  }
  EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));

  // Check the reference implementation too.
  ArrayXf reference_output = ReferenceBiquadFilter<float>(
      coeffs, Map<ArrayXXf>(input.data(), 1, kNumSamples)).row(0);
  EXPECT_THAT(reference_output, EigenArrayNear(expected, 1e-5));
}

TEST(BiquadFilterTest, BiquadFilterScalarInitialState) {
  constexpr int kNumSamples = 20;
  // A 100Hz lowpass filter designed for a sample rate of 48k.
  const double b0 = 0.00004244336440733159;
  const double b1 = 0.00008488672881466317;
  const double b2 = 0.00004244336440733159;
  const double a1 = -1.9814883348730705;
  const double a2 = 0.9816581083306998;

  constexpr double kDcValue = 0.4;
  constexpr double kDcGain = 3.0;
  BiquadFilterCoefficients coeffs = {{b0, b1, b2}, {1.0, a1, a2}};
  coeffs.SetGainAtFrequency(kDcGain, 0);
  srand(0 /* seed */);
  BiquadFilter<double> filter;
  filter.Init(1, coeffs);
  filter.SetSteadyStateCondition(kDcValue);

  for (int i = 0; i < kNumSamples; ++i) {
    double output;
    filter.ProcessSample(kDcValue, &output);
    EXPECT_NEAR(output, kDcValue * kDcGain, 1e-8);
  }
}

// Test basic use of BiquadFilter<float>.
TEST(BiquadFilterTest, BiquadFilterScalarFloat) {
  constexpr int kNumSamples = 20;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  BiquadFilter<float> filter;
  filter.Init(1, coeffs);

  ArrayXf input = ArrayXf::Random(kNumSamples);

  ArrayXf expected = ReferenceBiquadFilter<float>(
      coeffs, Map<ArrayXXf>(input.data(), 1, kNumSamples)).row(0);

  {  // Process all at once with ProcessBlock.
    ArrayXf output;
    filter.ProcessBlock(input, &output);
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));
  }
  {  // Process where input and output are vector<float>.
    std::vector<float> output;
    std::vector<float> input_vector(kNumSamples);
    Map<ArrayXf>(input_vector.data(), kNumSamples) = input;
    filter.Reset();
    filter.ProcessBlock(input_vector, &output);
    EXPECT_THAT(Map<ArrayXf>(output.data(), output.size()),
                EigenArrayNear(expected, 1e-5));
  }
  {  // Sample-by-sample processing with ProcessSample.
    ArrayXf output(kNumSamples);
    filter.Reset();
    for (int n = 0; n < kNumSamples; ++n) {
      filter.ProcessSample(input[n], &output[n]);
    }
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));
  }
}

// Test basic use of BiquadFilter<float> with AdjustGain
TEST(BiquadFilterTest, BiquadFilterScalarFloatAdjustGain) {
  constexpr int kNumSamples = 20;
  BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  BiquadFilter<float> filter;
  filter.Init(1, coeffs);

  ArrayXf input = ArrayXf::Random(kNumSamples);

  ArrayXf expected = ReferenceBiquadFilter<float>(
      coeffs, Map<ArrayXXf>(input.data(), 1, kNumSamples)).row(0);

  // Only these three lines modify BiquadFilterScalarFloat.
  double gain_factor = 1.9;
  coeffs.AdjustGain(gain_factor);  // Increase the gain using a double.
  filter.Init(1, coeffs);
  expected *= gain_factor;  // And adjust the expected result.

  {  // Process all at once with ProcessBlock.
    ArrayXf output;
    filter.ProcessBlock(input, &output);
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));
  }
  {  // Process where input and output are vector<float>.
    std::vector<float> output;
    std::vector<float> input_vector(kNumSamples);
    Map<ArrayXf>(input_vector.data(), kNumSamples) = input;
    filter.Reset();
    filter.ProcessBlock(input_vector, &output);
    EXPECT_THAT(Map<ArrayXf>(output.data(), output.size()),
                EigenArrayNear(expected, 1e-5));
  }
  {  // Sample-by-sample processing with ProcessSample.
    ArrayXf output(kNumSamples);
    filter.Reset();
    for (int n = 0; n < kNumSamples; ++n) {
      filter.ProcessSample(input[n], &output[n]);
    }
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));
  }
}

// Test basic use of BiquadFilter<ArrayXf>.
TEST(BiquadFilterTest, BiquadFilterMultichannelFloat) {
  constexpr int kNumSamples = 20;
  constexpr int kNumChannels = 3;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  BiquadFilter<ArrayXf> filter;
  filter.Init(kNumChannels, coeffs);

  ArrayXXf input = ArrayXXf::Random(kNumChannels, kNumSamples);
  ArrayXXf expected = ReferenceBiquadFilter<float>(coeffs, input);

  {  // Process all at once with ProcessBlock.
    ArrayXXf output;
    filter.ProcessBlock(input, &output);
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));
  }
  {  // Sample-by-sample processing with ProcessSample.
    ArrayXXf output(kNumChannels, kNumSamples);
    filter.Reset();
    for (int n = 0; n < kNumSamples; ++n) {
      ArrayXf output_sample;
      filter.ProcessSample(input.col(n), &output_sample);
      output.col(n) = output_sample;
    }
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));
  }
}

TEST(BiquadFilterTest, BiquadFilterAligned) {
  const int num_channels = 3;
  constexpr int kGeneratedBlockSize = 14;  // Different sizes to avoid going
  constexpr int kSampledBlockSize = 12;    // off edges of arrays.
  constexpr int kVerifiedBlockSize = 10;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  ArrayXXf input = ArrayXXf::Random(num_channels, kGeneratedBlockSize + 4);
  // First sample shouldn't impact results.
  input.col(0).setZero();

  ArrayXXf output_aligned(num_channels, kSampledBlockSize);
  ArrayXXf output_unaligned(num_channels, kSampledBlockSize);

  BiquadFilter<ArrayXf> filter;
  filter.Init(num_channels, coeffs);

  Map<ArrayXXf> data_ptr_aligned(input.data(), num_channels,
                                 kSampledBlockSize);
  filter.ProcessBlock(data_ptr_aligned, &output_aligned);
  filter.Reset();
  Map<ArrayXXf> data_ptr_unaligned(input.data() + 1, num_channels,
                                   kSampledBlockSize);
  filter.ProcessBlock(data_ptr_unaligned, &output_unaligned);
  // The channel order switches because the block is unaligned.
  // output_aligned:   output_unaligned:
  // 0AAAAAAAAAAAAAA   0BBBBBBBBBBBBBB
  // 0BBBBBBBBBBBBBB   0CCCCCCCCCCCCCC
  // 0CCCCCCCCCCCCCC   AAAAAAAAAAAAAAA

  // Compare the BC block in the above illustration.
  EXPECT_THAT(output_unaligned.block(0, 1, 2, kVerifiedBlockSize),
              EigenArrayNear(
      output_aligned.block(1, 1, 2, kVerifiedBlockSize), 1e-5));
  // Compare the A block.
  EXPECT_THAT(output_unaligned.block(2, 0, 1, kVerifiedBlockSize),
              EigenArrayNear(
      output_aligned.block(0, 1, 1, kVerifiedBlockSize), 1e-5));
}

// Test basic use of BiquadFilterCascade<float>.
TEST(BiquadFilterCascadeTest, BiquadFilterCascadeScalarFloat) {
  constexpr int kNumSamples = 20;
  constexpr int kNumStages = 2;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);

  std::vector<BiquadFilterCoefficients> all_coeffs(kNumStages, coeffs);
  const BiquadFilterCascadeCoefficients cascade_coeffs(all_coeffs);

  // A single channel, higher order filter.
  BiquadFilterCascade<float> filter;
  filter.Init(1, cascade_coeffs);

  ArrayXf input = ArrayXf::Random(kNumSamples);

  ArrayXf expected = input;
  for (int i = 0; i < kNumStages; ++i) {
    expected = ReferenceBiquadFilter<float>(
        coeffs, Map<ArrayXXf>(expected.data(), 1, kNumSamples)).row(0);
  }

  {  // Process all at once with ProcessBlock.
    ArrayXf output;
    filter.ProcessBlock(input, &output);
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));
  }
  {  // Process where input and output are vector<float>.
    std::vector<float> output;
    std::vector<float> input_vector(kNumSamples);
    Map<ArrayXf>(input_vector.data(), kNumSamples) = input;
    filter.Reset();
    filter.ProcessBlock(input_vector, &output);
    EXPECT_THAT(Map<ArrayXf>(output.data(), output.size()),
                EigenArrayNear(expected, 1e-5));
  }
  {  // Sample-by-sample processing with ProcessSample.
    ArrayXf output(kNumSamples);
    filter.Reset();
    for (int n = 0; n < kNumSamples; ++n) {
      filter.ProcessSample(input[n], &output[n]);
    }
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-5));
  }
}

// Test basic use of BiquadFilterCascade<float> with AdjustGain.
TEST(BiquadFilterCascadeTest, BiquadFilterCascadeScalarFloatAdjustGain) {
  constexpr int kNumSamples = 20;
  constexpr int kNumStages = 2;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);

  std::vector<BiquadFilterCoefficients> all_coeffs(kNumStages, coeffs);
  // This cascade_coeffs is not const so we can modify it:
  BiquadFilterCascadeCoefficients cascade_coeffs(all_coeffs);

  // A single channel, higher order filter.
  BiquadFilterCascade<float> filter;
  filter.Init(1, cascade_coeffs);

  ArrayXf input = ArrayXf::Random(kNumSamples);

  ArrayXf expected = input;
  for (int i = 0; i < kNumStages; ++i) {
    expected = ReferenceBiquadFilter<float>(
        coeffs, Map<ArrayXXf>(expected.data(), 1, kNumSamples)).row(0);
  }

  // Only these three lines modify BiquadFilterCascadeScalarFloat.
  double gain_factor = 1.7;
  cascade_coeffs.AdjustGain(gain_factor);  // Increase the gain using a double.
  filter.Init(1, cascade_coeffs);
  expected *= gain_factor;  // And adjust the expected result.

  {  // Process all at once with ProcessBlock.
    ArrayXf output;
    filter.ProcessBlock(input, &output);
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-4));
  }
  {  // Process where input and output are vector<float>.
    std::vector<float> output;
    std::vector<float> input_vector(kNumSamples);
    Map<ArrayXf>(input_vector.data(), kNumSamples) = input;
    filter.Reset();
    filter.ProcessBlock(input_vector, &output);
    EXPECT_THAT(Map<ArrayXf>(output.data(), output.size()),
                EigenArrayNear(expected, 1e-4));
  }
  {  // Sample-by-sample processing with ProcessSample.
    ArrayXf output(kNumSamples);
    filter.Reset();
    for (int n = 0; n < kNumSamples; ++n) {
      filter.ProcessSample(input[n], &output[n]);
    }
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-4));
  }
}

// Test BiquadFilterCascadeCoefficients::FindPeakFrequencyRadiansPerSample.
TEST(BiquadFilterCascadeTest, BiquadFilterCascadePeakFrequency) {
  constexpr int kNumStages = 2;
  // Make filter with poles of various Q, no zeros.
  for (double Q : {0.5, 1.0 / M_SQRT2, 1.0, 10.0, 100.0}) {
    for (double frequency = 0.8 * M_PI; frequency > 1e-4; frequency *= 0.8) {
      complex<double> pole_s(-frequency / (2 * Q),
                             frequency * sqrt(1.0 - 1.0 / (4 * Q * Q)));
      complex<double> pole_z = exp(pole_s);  // Map to z plane.
      double pole_x = pole_z.real();
      double pole_radius = std::abs(pole_z);
      std::vector<double> poles({1.0, -2 * pole_x, pole_radius * pole_radius});
      const BiquadFilterCoefficients coeffs{{1.0, 0.0, 0.0}, poles};
      std::vector<BiquadFilterCoefficients> all_coeffs(kNumStages, coeffs);
      BiquadFilterCascadeCoefficients cascade_coeffs(all_coeffs);

      // The thing to test:
      double peak_frequency =
          cascade_coeffs.FindPeakFrequencyRadiansPerSample().first;

      // This solves for the minimum absolute value of the poles (denominator)
      // as a function of frequency.
      double b = poles[1];
      double c = poles[2];
      double ratio = -(b + b * c) / (4 * c);
      double expected_frequency;
      if (std::abs(ratio) <= 1.0) {
        expected_frequency = std::acos(ratio);
      } else if (ratio > 1.0) {
        expected_frequency = 0.0;
      } else {
        expected_frequency = M_PI;
      }

      if (Q == 1.0 / M_SQRT2) {
        // Allow extra tolerance for the 'maximally flat' poles situation,
        // where numerical inaccuracy in the analysis is itself an issue
        // at low frequency where the response is flat around 0.0.
        EXPECT_NEAR(peak_frequency, expected_frequency,
                    4e-6 + 1e-5 * expected_frequency);
      } else {
        // In the non-flat case it can be very much more accurate.
        EXPECT_NEAR(peak_frequency, expected_frequency,
                    4e-8 + 1e-5 * expected_frequency);
      }
    }
  }
}

// Test basic use of BiquadFilterCascade<float> with SetPeakGain.
TEST(BiquadFilterCascadeTest, BiquadFilterCascadeScalarFloatSetPeakGain) {
  constexpr int kNumSamples = 20;
  constexpr int kNumStages = 2;
  // Make filter with peak at pi/2, using zeros at z = 1 and z = -1,
  // and poles at radius^2 = 0.25 and real_part = 0:
  const BiquadFilterCoefficients coeffs = {{1.0, 0.0, -1.0}, {1.0, 0.0, 0.25}};
  srand(0 /* seed */);

  std::vector<BiquadFilterCoefficients> all_coeffs(kNumStages, coeffs);
  // This cascade_coeffs is not const so we can modify it:
  BiquadFilterCascadeCoefficients cascade_coeffs(all_coeffs);

  // A single channel, higher order filter.
  BiquadFilterCascade<float> filter;
  filter.Init(1, cascade_coeffs);

  ArrayXf input = ArrayXf::Random(kNumSamples);

  ArrayXf expected = input;
  for (int i = 0; i < kNumStages; ++i) {
    expected = ReferenceBiquadFilter<float>(
        coeffs, Map<ArrayXXf>(expected.data(), 1, kNumSamples)).row(0);
  }

  // Coefficients and these lines modify BiquadFilterCascadeScalarFloat.
  double peak_frequency =
      cascade_coeffs.FindPeakFrequencyRadiansPerSample().first;
  std::complex<float> z_peak(cos(peak_frequency), sin(peak_frequency));
  double old_peak_gain = abs(cascade_coeffs.EvalTransferFunction(z_peak));
  // Product of distances from zeros (2.0) over distances from poles, squared,
  // which is expected_peak_gain = (8/3)^2 = 64/9 = 7.111111:
  double expected_peak_gain = pow(2.0 / (0.5 * 1.5), kNumStages);
  EXPECT_NEAR(7.111111, expected_peak_gain, 1e-5);
  EXPECT_NEAR(old_peak_gain, expected_peak_gain, 1e-5);
  double new_peak_gain = 0.1;
  cascade_coeffs.SetPeakGain(new_peak_gain);
  filter.Init(1, cascade_coeffs);

  expected *= new_peak_gain / old_peak_gain;
  EXPECT_NEAR(new_peak_gain, abs(cascade_coeffs.EvalTransferFunction(z_peak)),
              1e-5);

  EXPECT_NEAR(M_PI / 2, peak_frequency, 1e-5);

  {  // Process all at once with ProcessBlock.
    ArrayXf output;
    filter.ProcessBlock(input, &output);
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-4));
  }
  {  // Process where input and output are vector<float>.
    std::vector<float> output;
    std::vector<float> input_vector(kNumSamples);
    Map<ArrayXf>(input_vector.data(), kNumSamples) = input;
    filter.Reset();
    filter.ProcessBlock(input_vector, &output);
    EXPECT_THAT(Map<ArrayXf>(output.data(), output.size()),
                EigenArrayNear(expected, 1e-4));
  }
  {  // Sample-by-sample processing with ProcessSample.
    ArrayXf output(kNumSamples);
    filter.Reset();
    for (int n = 0; n < kNumSamples; ++n) {
      filter.ProcessSample(input[n], &output[n]);
    }
    EXPECT_THAT(output, EigenArrayNear(expected, 1e-4));
  }
}

template <typename TypeParam>
class BiquadFilterTypedTest : public ::testing::Test {};

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
    > TestTypes;
TYPED_TEST_CASE(BiquadFilterTypedTest, TestTypes);

// Test BiquadFilter<SampleType> for different template args.
TYPED_TEST(BiquadFilterTypedTest, BiquadFilter) {
  using SampleType = TypeParam;
  SCOPED_TRACE(util::format::StringF("SampleType: %s",
                                     GetTypeName<SampleType>().c_str()));

  constexpr int kNumSamples = 20;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  const complex<double> z = {0.3, -0.5};
  const complex<double> z_inv = 1.0 / z;
  EXPECT_LE(std::abs(
      coeffs.EvalTransferFunction(z) -
      (coeffs.b[0] + z_inv * (coeffs.b[1] + z_inv * coeffs.b[2])) /
      (coeffs.a[0] + z_inv * (coeffs.a[1] + z_inv * coeffs.a[2]))), 1e-4);
  srand(0 /* seed */);

  using FilterType = BiquadFilter<SampleType>;
  const int kNumChannelsAtCompileTime = FilterType::kNumChannelsAtCompileTime;
  using ScalarType = typename FilterType::ScalarType;
  using BlockOfSamples = typename Eigen::Array<
      ScalarType, kNumChannelsAtCompileTime, Dynamic>;
  FilterType filter;
  TypesMatch<SampleType, typename FilterType::SampleType>();

  for (int num_channels : {1, 2, 3}) {
    if (kNumChannelsAtCompileTime != Dynamic &&
        kNumChannelsAtCompileTime != num_channels) {
      continue;  // Skip if SampleType is incompatible with num_channels.
    }
    SCOPED_TRACE("num_channels: " + testing::PrintToString(num_channels));
    filter.Init(num_channels, coeffs);
    EXPECT_EQ(filter.num_channels(), num_channels);

    BlockOfSamples input = BlockOfSamples::Random(num_channels, kNumSamples);
    BlockOfSamples output;
    filter.ProcessBlock(input, &output);

    EXPECT_THAT(output, EigenArrayNear(
        ReferenceBiquadFilter<ScalarType>(coeffs, input), 1e-5));
  }
}

// Test BiquadFilterCascade<SampleType> for different template
// args.
TYPED_TEST(BiquadFilterTypedTest, BiquadFilterCascade) {
  using SampleType = TypeParam;
  SCOPED_TRACE(util::format::StringF("SampleType: %s",
                                     GetTypeName<SampleType>().c_str()));

  constexpr int kNumSamples = 20;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);

  using FilterType = BiquadFilterCascade<SampleType>;
  const int kNumChannelsAtCompileTime = FilterType::kNumChannelsAtCompileTime;
  using ScalarType = typename FilterType::ScalarType;
  using BlockOfSamples = typename Eigen::Array<
      ScalarType, kNumChannelsAtCompileTime, Dynamic>;
  FilterType filter;
  TypesMatch<SampleType, typename FilterType::SampleType>();

  for (int num_channels : {1, 2}) {
    for (int num_stages : {1, 2, 3}) {
      if (kNumChannelsAtCompileTime != Dynamic &&
          kNumChannelsAtCompileTime != num_channels) {
        continue;  // Skip if SampleType is incompatible with num_channels.
      }
      SCOPED_TRACE(util::format::StringF("num_channels: %d  num_stages: %d",
                                         num_channels, num_stages));

      std::vector<BiquadFilterCoefficients> all_coeffs(num_stages, coeffs);
      filter.Init(num_channels, BiquadFilterCascadeCoefficients(all_coeffs));

      EXPECT_EQ(filter.num_channels(), num_channels);

      BlockOfSamples input = BlockOfSamples::Random(num_channels, kNumSamples);
      BlockOfSamples output;
      filter.ProcessBlock(input, &output);

      BlockOfSamples reference_output = input;
      for (int i = 0; i < num_stages; ++i) {
        reference_output = ReferenceBiquadFilter<ScalarType>(coeffs,
                                                             reference_output);
      }
      EXPECT_THAT(output, EigenArrayNear(reference_output, 5e-5));
    }
  }
}

void BM_BiquadFilterScalarFloat(benchmark::State& state) {
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  ArrayXf input = ArrayXf::Random(kSamplePerBlock);
  ArrayXf output(kSamplePerBlock);
  BiquadFilter<float> filter;
  filter.Init(1, coeffs);

  while (state.KeepRunning()) {
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
BENCHMARK(BM_BiquadFilterScalarFloat);

void BM_BiquadFilterArrayXf(benchmark::State& state) {
  const int num_channels = state.range(0);
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  ArrayXXf input = ArrayXXf::Random(num_channels, kSamplePerBlock);
  ArrayXXf output(num_channels, kSamplePerBlock);
  BiquadFilter<ArrayXf> filter;
  filter.Init(num_channels, coeffs);

  while (state.KeepRunning()) {
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
BENCHMARK(BM_BiquadFilterArrayXf)
    ->DenseRange(1, 10);

template <bool kIsAligned>
void BM_BiquadFilterMapImpl(benchmark::State& state) {
  const int num_channels = 4;
  constexpr int kSamplePerBlock = 1001;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  ArrayXXf input = ArrayXXf::Random(num_channels, kSamplePerBlock);
  ArrayXXf output(num_channels, kSamplePerBlock);
  BiquadFilter<ArrayXf> filter;
  filter.Init(num_channels, coeffs);

  float* data_ptr = kIsAligned ? input.data() : input.data() + 1;
  Map<ArrayXXf> input_mapped(data_ptr, num_channels, kSamplePerBlock - 20);
  while (state.KeepRunning()) {
    filter.ProcessBlock(input_mapped, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}

void BM_BiquadFilterMapAligned(benchmark::State& state) {
  BM_BiquadFilterMapImpl<true>(state);
}
BENCHMARK(BM_BiquadFilterMapAligned);

void BM_BiquadFilterMapUnaligned(benchmark::State& state) {
  BM_BiquadFilterMapImpl<false>(state);
}
BENCHMARK(BM_BiquadFilterMapUnaligned);

template <int kNumChannels>
void BM_BiquadFilterArrayNf(benchmark::State& state) {
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  using ArrayNf = Eigen::Array<float, kNumChannels, 1>;
  using ArrayNXf = Eigen::Array<float, kNumChannels, Dynamic>;
  ArrayNXf input = ArrayNXf::Random(kNumChannels, kSamplePerBlock);
  ArrayNXf output(kNumChannels, kSamplePerBlock);
  BiquadFilter<ArrayNf> filter;
  filter.Init(kNumChannels, coeffs);

  while (state.KeepRunning()) {
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
// Old-style template benchmarking needed for open sourcing. External
// google/benchmark repo doesn't have functionality from cl/118676616 enabling
// BENCHMARK(TemplatedFunction<2>) syntax.
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 1);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 2);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 3);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 4);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 5);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 6);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 7);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 8);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 9);
BENCHMARK_TEMPLATE(BM_BiquadFilterArrayNf, 10);

void BM_BiquadFilterCascadeScalarFloatSample(benchmark::State& state) {
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  ArrayXf input = ArrayXf::Random(kSamplePerBlock);
  ArrayXf output(kSamplePerBlock);
  BiquadFilterCascade<float> filter;
  std::vector<BiquadFilterCoefficients> all_coeffs(4, coeffs);
  filter.Init(1, BiquadFilterCascadeCoefficients(all_coeffs));

  while (state.KeepRunning()) {
    for (int i = 0; i < kSamplePerBlock; ++i) {
      filter.ProcessSample(input[i], &output[i]);
    }
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
BENCHMARK(BM_BiquadFilterCascadeScalarFloatSample);

void BM_BiquadFilterCascadeScalarFloatBlock(benchmark::State& state) {
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  ArrayXf input = ArrayXf::Random(kSamplePerBlock);
  ArrayXf output(kSamplePerBlock);
  BiquadFilterCascade<float> filter;
  std::vector<BiquadFilterCoefficients> all_coeffs(4, coeffs);
  filter.Init(1, BiquadFilterCascadeCoefficients(all_coeffs));

  while (state.KeepRunning()) {
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
BENCHMARK(BM_BiquadFilterCascadeScalarFloatBlock);

void BM_BiquadFilterCascadeArrayXfSample(benchmark::State& state) {
  const int num_channels = state.range(0);
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  ArrayXXf input = ArrayXXf::Random(num_channels, kSamplePerBlock);
  ArrayXXf output(num_channels, kSamplePerBlock);
  BiquadFilterCascade<ArrayXf> filter;
  std::vector<BiquadFilterCoefficients> all_coeffs(4, coeffs);
  filter.Init(num_channels, BiquadFilterCascadeCoefficients(all_coeffs));

  while (state.KeepRunning()) {
    for (int i = 0; i < kSamplePerBlock; ++i) {
      Map<ArrayXf> output_map(output.col(i).data(), output.rows());
      filter.ProcessSample(input.col(i), &output_map);
    }
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
BENCHMARK(BM_BiquadFilterCascadeArrayXfSample)
    ->DenseRange(1, 10);

void BM_BiquadFilterCascadeArrayXfBlock(benchmark::State& state) {
  const int num_channels = state.range(0);
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  ArrayXXf input = ArrayXXf::Random(num_channels, kSamplePerBlock);
  ArrayXXf output(num_channels, kSamplePerBlock);
  BiquadFilterCascade<ArrayXf> filter;
  std::vector<BiquadFilterCoefficients> all_coeffs(4, coeffs);
  filter.Init(num_channels, BiquadFilterCascadeCoefficients(all_coeffs));

  while (state.KeepRunning()) {
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
BENCHMARK(BM_BiquadFilterCascadeArrayXfBlock)
    ->DenseRange(1, 10);

template <int kNumChannels>
void BM_BiquadFilterCascadeArrayNfSample(benchmark::State& state) {
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  using ArrayNf = Eigen::Array<float, kNumChannels, 1>;
  using ArrayNXf = Eigen::Array<float, kNumChannels, Dynamic>;
  ArrayNXf input = ArrayNXf::Random(kNumChannels, kSamplePerBlock);
  ArrayNXf output(kNumChannels, kSamplePerBlock);
  BiquadFilterCascade<ArrayNf> filter;
  std::vector<BiquadFilterCoefficients> all_coeffs(4, coeffs);
  filter.Init(kNumChannels, BiquadFilterCascadeCoefficients(all_coeffs));

  while (state.KeepRunning()) {
    for (int i = 0; i < kSamplePerBlock; ++i) {
      Map<ArrayXf> output_map(output.col(i).data(), output.rows());
      filter.ProcessSample(input.col(i), &output_map);
    }
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 1);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 2);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 3);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 4);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 5);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 6);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 7);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 8);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 9);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfSample, 10);

template <int kNumChannels>
void BM_BiquadFilterCascadeArrayNfBlock(benchmark::State& state) {
  constexpr int kSamplePerBlock = 1000;
  const BiquadFilterCoefficients coeffs = {{-0.2, 1.9, 0.4}, {0.5, -0.2, 0.1}};
  srand(0 /* seed */);
  using ArrayNf = Eigen::Array<float, kNumChannels, 1>;
  using ArrayNXf = Eigen::Array<float, kNumChannels, Dynamic>;
  ArrayNXf input = ArrayNXf::Random(kNumChannels, kSamplePerBlock);
  ArrayNXf output(kNumChannels, kSamplePerBlock);
  BiquadFilterCascade<ArrayNf> filter;
  std::vector<BiquadFilterCoefficients> all_coeffs(4, coeffs);
  filter.Init(kNumChannels, BiquadFilterCascadeCoefficients(all_coeffs));

  while (state.KeepRunning()) {
    filter.ProcessBlock(input, &output);
    benchmark::DoNotOptimize(output);
  }
  state.SetItemsProcessed(kSamplePerBlock * state.iterations());
}
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 1);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 2);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 3);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 4);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 5);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 6);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 7);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 8);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 9);
BENCHMARK_TEMPLATE(BM_BiquadFilterCascadeArrayNfBlock, 10);

}  // namespace
}  // namespace linear_filters

/*
Benchmark results:
Run on lpac20 (32 X 2600 MHz CPUs); 2017-05-10T17:42:33.360930491-07:00
CPU: Intel Sandybridge with HyperThreading (16 cores)
Benchmark                                 Time(ns)        CPU(ns)     Iterations
--------------------------------------------------------------------------------
BM_BiquadFilterScalarFloat                    3398           3393        2063324
BM_BiquadFilterArrayXf/1                      4863           4856        1438957
BM_BiquadFilterArrayXf/2                      5976           5966        1000000
BM_BiquadFilterArrayXf/3                      8183           8170         854966
BM_BiquadFilterArrayXf/4                      6869           6858        1000000
BM_BiquadFilterArrayXf/5                     15886          15861         441756
BM_BiquadFilterArrayXf/6                     18693          18663         373732
BM_BiquadFilterArrayXf/7                     21309          21277         329144
BM_BiquadFilterArrayXf/8                     24244          24204         288171
BM_BiquadFilterArrayXf/9                     26989          26946         259185
BM_BiquadFilterArrayXf/10                    29791          29745         235030
BM_BiquadFilterMap<true>                      6604           6595        1000000
BM_BiquadFilterMap<false>                     6734           6723        1000000
BM_BiquadFilterArrayNf<1>                     4769           4762        1470101
BM_BiquadFilterArrayNf<2>                     5556           5547        1000000
BM_BiquadFilterArrayNf<3>                     7507           7496         934751
BM_BiquadFilterArrayNf<4>                     6156           6146        1000000
BM_BiquadFilterArrayNf<5>                    12276          12257         574693
BM_BiquadFilterArrayNf<6>                    15666          15641         448280
BM_BiquadFilterArrayNf<7>                    18080          18051         387383
BM_BiquadFilterArrayNf<8>                     9556           9542         734240
BM_BiquadFilterArrayNf<9>                    23605          23570         297713
BM_BiquadFilterArrayNf<10>                   27302          27261         255606
BM_BiquadFilterCascadeScalarFloatSample      11569          11550         604878
BM_BiquadFilterCascadeScalarFloatBlock       13735          13713         515864
BM_BiquadFilterCascadeArrayXfSample/1        18025          17998         386689
BM_BiquadFilterCascadeArrayXfSample/2        28424          28386         243501
BM_BiquadFilterCascadeArrayXfSample/3        39358          39305         178192
BM_BiquadFilterCascadeArrayXfSample/4        50622          50544         100000
BM_BiquadFilterCascadeArrayXfSample/5        61354          61261         100000
BM_BiquadFilterCascadeArrayXfSample/6        73064          72959          96419
BM_BiquadFilterCascadeArrayXfSample/7        85828          85702          81138
BM_BiquadFilterCascadeArrayXfSample/8        96669          96529          72792
BM_BiquadFilterCascadeArrayXfSample/9       107177         107014          65340
BM_BiquadFilterCascadeArrayXfSample/10      118531         118361          59018
BM_BiquadFilterCascadeArrayXfBlock/1         20770          20740         339993
BM_BiquadFilterCascadeArrayXfBlock/2         25357          25318         275245
BM_BiquadFilterCascadeArrayXfBlock/3         34476          34421         204151
BM_BiquadFilterCascadeArrayXfBlock/4         27732          27690         252813
BM_BiquadFilterCascadeArrayXfBlock/5         62601          62504         100000
BM_BiquadFilterCascadeArrayXfBlock/6         74159          74051          94613
BM_BiquadFilterCascadeArrayXfBlock/7         85524          85398          82287
BM_BiquadFilterCascadeArrayXfBlock/8         97076          96931          72126
BM_BiquadFilterCascadeArrayXfBlock/9        108911         108740          63970
BM_BiquadFilterCascadeArrayXfBlock/10       120293         120103          57146
BM_BiquadFilterCascadeArrayNfSample<1>       11564          11546         605937
BM_BiquadFilterCascadeArrayNfSample<2>       27433          27388         256140
BM_BiquadFilterCascadeArrayNfSample<3>       38077          38020         183713
BM_BiquadFilterCascadeArrayNfSample<4>       25708          25669         272567
BM_BiquadFilterCascadeArrayNfSample<5>       58665          58572         100000
BM_BiquadFilterCascadeArrayNfSample<6>       71413          71303          97771
BM_BiquadFilterCascadeArrayNfSample<7>       76718          76603          91528
BM_BiquadFilterCascadeArrayNfSample<8>       45814          45748         153068
BM_BiquadFilterCascadeArrayNfSample<9>       90321          90190          77043
BM_BiquadFilterCascadeArrayNfSample<10>     108846         108679          64552
BM_BiquadFilterCascadeArrayNfBlock<1>        19126          19096         366067
BM_BiquadFilterCascadeArrayNfBlock<2>        23789          23755         313954
BM_BiquadFilterCascadeArrayNfBlock<3>        28988          28941         225793
BM_BiquadFilterCascadeArrayNfBlock<4>        28245          28201         248161
BM_BiquadFilterCascadeArrayNfBlock<5>        54014          53928         100000
BM_BiquadFilterCascadeArrayNfBlock<6>        62439          62347         100000
BM_BiquadFilterCascadeArrayNfBlock<7>        72409          72298          97229
BM_BiquadFilterCascadeArrayNfBlock<8>        38167          38106         183813
BM_BiquadFilterCascadeArrayNfBlock<9>        94294          94145          74420
BM_BiquadFilterCascadeArrayNfBlock<10>      108906         108731          64491
*/
