/*
 * Copyright 2019 Google LLC
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

#include "audio/dsp/mfcc/mel_filterbank.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "audio/dsp/porting.h"  // auto-added.


namespace audio_dsp {

using ::testing::Each;
using ::testing::Gt;

TEST(MelFilterbankTest, AgreesWithPythonGoldenValues) {
  // This test verifies the Mel filterbank against "golden values".
  // Golden values are from an independent Python Mel implementation.
  MelFilterbank filterbank;

  std::vector<double> input;
  const int kSampleCount = 513;
  for (int i = 0; i < kSampleCount; ++i) {
    input.push_back(i + 1);
  }
  const int kChannelCount = 20;
  filterbank.Initialize(input.size(),
                        22050 /* sample rate */,
                        kChannelCount /* channels */,
                        20.0 /*  lower frequency limit */,
                        4000.0 /* upper frequency limit */);

  std::vector<double> output;
  filterbank.Compute(input, &output);

  std::vector<double> expected = {
      7.38894574,   10.30330648, 13.72703292,  17.24158686,  21.35253118,
      25.77781089,  31.30624108, 37.05877236,  43.9436536,   51.80306637,
      60.79867148,  71.14363376, 82.90910141,  96.50069158,  112.08428368,
      129.96721968, 150.4277597, 173.74997634, 200.86037462, 231.59802942};

  ASSERT_EQ(output.size(), kChannelCount);

  for (int i = 0; i < kChannelCount; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-04);
  }
}

TEST(MelFilterbankTest, IgnoresExistingContentOfOutputVector) {
  // Test for bug where the output vector was not cleared before
  // accumulating next frame's weighted spectral values.
  MelFilterbank filterbank;

  const int kSampleCount = 513;
  std::vector<double> input;
  std::vector<double> output;

  filterbank.Initialize(kSampleCount,
                        22050 /* sample rate */,
                        20 /* channels */,
                        20.0 /*  lower frequency limit */,
                        4000.0 /* upper frequency limit */);


  // First call with nonzero input value, and an empty output vector,
  // will resize the output and fill it with the correct, nonzero outputs.
  input.assign(kSampleCount, 1.0);
  filterbank.Compute(input, &output);
  EXPECT_THAT(output, Each(Gt(0.0)));

  // Second call with zero input should also generate zero output.  However,
  // the output vector now is already the correct size, but full of nonzero
  // values.  Make sure these don't affect the output.
  input.assign(kSampleCount, 0.0);
  filterbank.Compute(input, &output);
  EXPECT_THAT(output, Each(0.0));
}

TEST(MelFilterbankTest, LowerEdgeAtZeroIsOk) {
  // Original code objected to lower_frequency_edge == 0, but it's OK really.
  MelFilterbank filterbank;

  std::vector<double> input;
  const int kSampleCount = 513;
  for (int i = 0; i < kSampleCount; ++i) {
    input.push_back(i + 1);
  }
  const int kChannelCount = 20;
  filterbank.Initialize(input.size(),
                        22050 /* sample rate */,
                        kChannelCount /* channels */,
                        0.0 /*  lower frequency limit */,
                        4000.0 /* upper frequency limit */);

  std::vector<double> output;
  filterbank.Compute(input, &output);

  ASSERT_EQ(output.size(), kChannelCount);

  for (int i = 0; i < kChannelCount; ++i) {
    float t = output[i];
    EXPECT_FALSE(std::isnan(t) || std::isinf(t));
  }

  // Golden values for min_frequency=0.0 from mfcc_mel.py via
  // http://colab/v2/notebook#fileId=0B4TJPzYpfSM2RDY3MWk0bEFSdFE .
  std::vector<double> expected = {
    6.55410799, 9.56411605, 13.01286477, 16.57608704, 20.58962488,
    25.13380881, 30.52101218, 36.27805982, 43.40116347, 51.30065013,
    60.04552778, 70.85208474, 82.60955902, 96.41872603, 112.26929653,
    130.46661761, 151.28700221, 175.39139009, 202.84483315, 234.63080493};
  for (int i = 0; i < kChannelCount; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-4);
  }
}

}  // namespace audio_dsp
