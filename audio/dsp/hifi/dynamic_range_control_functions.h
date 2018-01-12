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

// Input-output level characteristics for dynamic gain control modules.
// They are templated for use with Eigen ArrayXf and Map<ArrayXf> types.

#ifndef AUDIO_DSP_HIFI_DYNAMIC_RANGE_CONTROL_FUNCTIONS_H_
#define AUDIO_DSP_HIFI_DYNAMIC_RANGE_CONTROL_FUNCTIONS_H_

#include <type_traits>

#include "audio/dsp/porting.h"  // auto-added.


namespace audio_dsp {

namespace internal {

// Generate a function that is zero below the knee and has a slope of -1 above
// transition_center. There is a smooth transition of length transition_width
// centered around transition_center. This function is actually the negative of
// what you usually expect a ReLU to be.
template <typename InputEigenType, typename OutputEigenType>
void SmoothReLU(const InputEigenType& input,
                float transition_center,
                float transition_width,
                OutputEigenType* output) {
  static_assert(std::is_same<typename InputEigenType::Scalar, float>::value,
                "Scalar type must be float.");
  if (transition_width > 0) {
    const float start_knee = transition_center - transition_width / 2;
    auto temp = input - start_knee;
    const float knee_inv = 1 / transition_width;
    *output =
        -0.5f * (temp.cwiseMax(0).cwiseMin(transition_width) * temp * knee_inv +
        (temp - transition_width).cwiseMax(0));
  } else {
    *output = (transition_center - input).cwiseMin(0);
  }
}

}  // namespace internal

// The input-output characteristic for a compressor/limiter is such that below
// the threshold, input level = output level, and above the threshold additional
// increase in gain is reduced by a factor equal to the ratio.
//
// When the knee width is nonzero, a quadratic polynomial is used to interpolate
// between the two behaviors. The transition goes from
// threshold_db - knee_width_db / 2 to threshold_db + knee_width_db / 2.
//
// The computation for the knee can be found here:
// http://c4dm.eecs.qmul.ac.uk/audioengineering/compressors/documents/Reiss-Tutorialondynamicrangecompression.pdf
//
// Equivalent code:
//   const float half_knee = knee_width_db / 2;
//   if (input_level_db - threshold_db <= -half_knee) {
//     return input_level_db;
//   } else if (input_level_db - threshold_db >= half_knee) {
//     return threshold_db + (input_level_db - threshold_db) / ratio;
//   } else {
//     const float knee_end = input_level_db - threshold_db + half_knee;
//     return input_level_db +
//         ((1 / ratio) - 1) * knee_end * knee_end / (knee_width_db * 2);
//   }
template <typename InputEigenType, typename OutputEigenType>
void OutputLevelCompressor(const InputEigenType& input_level_db,
                           float threshold_db,
                           float ratio,
                           float knee_width_db,
                           OutputEigenType* output_level_db) {
  internal::SmoothReLU(input_level_db, threshold_db, knee_width_db,
                       output_level_db);
  const float slope = 1 - (1 / ratio);
  *output_level_db = input_level_db + *output_level_db * slope;
}

template <typename InputEigenType, typename OutputEigenType>
void OutputLevelLimiter(const InputEigenType& input_level_db,
                        float threshold_db,
                        float knee_width_db,
                        OutputEigenType* output_level_db) {
  internal::SmoothReLU(input_level_db, threshold_db, knee_width_db,
                       output_level_db);
  *output_level_db += input_level_db;
}

// The input-output characteristic for an expander/noise gate is such that above
// the threshold, input level = output level, and below the threshold additional
// increase in gain is reduced by a factor equal to the ratio.
//
// When the knee width is nonzero, a quadratic polynomial is used to interpolate
// between the two behaviors. The transition goes from
// threshold_db - knee_width_db / 2 to threshold_db + knee_width_db / 2.
//
// Equivalent code:
//   const float half_knee = knee_width_db / 2;
//   if (input_level_db - threshold_db >= half_knee) {
//     return input_level_db;
//   } else if (input_level_db - threshold_db <= -half_knee) {
//     return threshold_db + (input_level_db - threshold_db) * ratio;
//   } else {
//     const float knee_end = input_level_db - threshold_db - half_knee;
//     return input_level_db +
//         (1 - ratio) * knee_end * knee_end / (knee_width_db * 2);
//   }
template <typename InputEigenType, typename OutputEigenType>
void OutputLevelExpander(const InputEigenType& input_level_db,
                          float threshold_db,
                          float ratio,
                          float knee_width_db,
                          OutputEigenType* output_level_db) {
  internal::SmoothReLU(input_level_db, threshold_db, knee_width_db,
                       output_level_db);
  const float slope = ratio - 1;
  *output_level_db = input_level_db +
      (input_level_db - threshold_db + *output_level_db) * slope;
}

}  // namespace audio_dsp

#endif  // AUDIO_DSP_HIFI_DYNAMIC_RANGE_CONTROL_FUNCTIONS_H_
