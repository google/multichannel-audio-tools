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

#include "audio/dsp/hifi/dynamic_range_control.h"

#include "audio/dsp/decibels.h"
#include "audio/dsp/hifi/dynamic_range_control_functions.h"

namespace audio_dsp {

using ::Eigen::ArrayXf;

DynamicRangeControl::DynamicRangeControl(
    const DynamicRangeControlParams& params)
      : params_(params) {
  CHECK_GE(params.knee_width_db, 0);
  CHECK_GT(params.ratio, 0);
  CHECK_GT(params.attack_s, 0);
  CHECK_GT(params.release_s, 0);
}

void DynamicRangeControl::Init(int num_channels, int max_block_size_samples,
                               float sample_rate_hz) {
  CHECK_GT(num_channels, 0);
  CHECK_GT(max_block_size_samples, 0);
  CHECK_GT(sample_rate_hz, 0);
  num_channels_ = num_channels;
  sample_rate_hz_ = sample_rate_hz;
  max_block_size_samples_ = max_block_size_samples;
  workspace_ = ArrayXf::Zero(max_block_size_samples_);
  workspace_drc_output_ = ArrayXf::Zero(max_block_size_samples_);
  envelope_.reset(new AttackReleaseEnvelope(params_.attack_s,
                                            params_.release_s,
                                            sample_rate_hz_));
}

void DynamicRangeControl::Reset() {
  envelope_->Reset();
}

void DynamicRangeControl::ComputeGain(VectorType* data_ptr) {
  VectorType& data = *data_ptr;
  for (int i = 0; i < data.rows(); ++i) {
    data[i] = envelope_->Output(data[i]);  // Calls abs().
  }
  // Convert to decibels.
  // TODO: Consider downsampling the envelope by an integer factor
  // to 8k or lower.
  if (params_.envelope_type == kRms) {
    PowerRatioToDecibels(data + 1e-12f, &data);
  } else {
    AmplitudeRatioToDecibels(data + 1e-12f, &data);
  }
  // Store the gain computation in the workspace.
  VectorType workspace_map = workspace_drc_output_.head(data.size());
  ApplyGainControl(data, &workspace_map);
  workspace_map += params_.output_gain_db;

  // Convert back to linear.
  DecibelsToAmplitudeRatio(workspace_map, &data);
  DCHECK(data.allFinite());
}

void DynamicRangeControl::ApplyGainControl(
    const VectorType& input_level, VectorType* output_gain) {
  switch (params_.dynamics_type) {
    case kCompressor:
      OutputLevelCompressor(
          input_level + params_.input_gain_db, params_.threshold_db,
          params_.ratio, params_.knee_width_db, output_gain);

      break;
    case kLimiter:
      OutputLevelLimiter(
          input_level + params_.input_gain_db,
          params_.threshold_db, params_.knee_width_db, output_gain);

      break;
    case kExpander:
      OutputLevelExpander(
          input_level + params_.input_gain_db, params_.threshold_db,
          params_.ratio, params_.knee_width_db, output_gain);
      break;
    case kNoiseGate:
      // Avoid actually using infinity, which could cause numerical problems.
      // 1000dB of suppression is way more than enough for any use case.
      constexpr float kInfiniteRatio = 1000;
      OutputLevelExpander(
          input_level + params_.input_gain_db, kInfiniteRatio,
          params_.threshold_db, params_.knee_width_db, output_gain);
      break;
  }
  // Compute the gain to apply rather than the output level.
  *output_gain -= input_level;
}

}  // namespace audio_dsp
