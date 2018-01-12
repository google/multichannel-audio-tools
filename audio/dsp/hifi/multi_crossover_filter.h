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

// A filterbank whose bands have the property that their magnitude responses
// sum to unity.

#ifndef AUDIO_DSP_HIFI_MULTI_CROSSOVER_FILTER_H_
#define AUDIO_DSP_HIFI_MULTI_CROSSOVER_FILTER_H_

#include <vector>

#include "audio/linear_filters/crossover.h"
#include "audio/linear_filters/ladder_filter.h"
#include "glog/logging.h"
#include "third_party/eigen3/Eigen/Core"

#include "audio/dsp/porting.h"  // auto-added.


namespace audio_dsp {

// The sum of the magnitude responses of each band will equal unity as long as
// the CrossoverType = kLinkwitzRiley.
// The crossover frequencies can be changed without generating audio artifacts.
class MultiCrossoverFilter {
 public:
  MultiCrossoverFilter(int num_bands, int order,
                       linear_filters::CrossoverType type =
                           linear_filters::kLinkwitzRiley)
      : type_(type),
        order_(order),
        num_bands_(num_bands),
        num_channels_(0 /* uninitialized */),
        sample_rate_hz_(0 /* uninitialized */),
        highpass_filters_(num_bands - 1),
        lowpass_filters_(num_bands - 1),
        filtered_output_(num_bands) {
    CHECK_GT(num_bands, 1);
  }

  // crossover_frequencies_hz.size() must equal num_bands - 1 and have
  // monotonically increasing elements.
  void Init(int num_channels, float sample_rate_hz,
            const std::vector<float>& crossover_frequencies_hz);

  void Reset() {
    for (auto& f : highpass_filters_) { f.Reset(); }
    for (auto& f : lowpass_filters_) { f.Reset(); }
    for (auto& output : filtered_output_) { output.setZero(); }
  }

  // crossover_frequencies_hz.size() must equal num_bands - 1 and have
  // monotonically increasing elements.
  // Interpolation is done in the filters so that audio artifacts are not
  // caused. During interpolation, magnitude responses are not guaranteed to sum
  // to unity.
  void SetCrossoverFrequencies(
      const std::vector<float>& crossover_frequencies_hz);

  // Process a block of samples. input is a 2D Eigen array with contiguous
  // column-major data, where the number of rows equals GetNumChannels().
  void ProcessBlock(const Eigen::ArrayXXf& input);

  int num_bands() const {
    return num_bands_;
  }

  // Filtered output from the filter_stage-th of the filterbank. Channels are
  // ordered by increasing passband frequency.
  const Eigen::ArrayXXf& FilteredOutput(int band_number) const {
    DCHECK_LT(band_number, num_bands_);
    return filtered_output_[band_number];
  }

 private:
  void SetCrossoverFrequenciesInternal(
      const std::vector<float>& crossover_frequencies_hz, bool initial);

  const linear_filters::CrossoverType type_;
  const int order_;
  const int num_bands_;
  int num_channels_;
  float sample_rate_hz_;

  // The successive crossover stages, processed in order, starting with the
  // first.
  std::vector<linear_filters::LadderFilter<Eigen::ArrayXf>> highpass_filters_;
  std::vector<linear_filters::LadderFilter<Eigen::ArrayXf>> lowpass_filters_;

  // filtered_output_[i] is the filtered output for the ith stage of the
  // cascade.
  std::vector<Eigen::ArrayXXf> filtered_output_;
};

}  // namespace audio_dsp

#endif  // AUDIO_DSP_HIFI_MULTI_CROSSOVER_FILTER_H_
