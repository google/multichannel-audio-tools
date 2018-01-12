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

// Apply dynamic range control to an audio signal. This class is vectorized
// using Eigen for fast computation and supports processing multichannel
// inputs.
//
// Common other names for this (to help with code searching):
//  - automatic gain control (AGC)
//  - dynamic range compression (DRC)
//  - limiting, limiter
//  - noise gate
//  - expander
//
// All measures with units in decibels are with reference to unity. This
// is commonly called dB full-scale (dBFS).
#ifndef AUDIO_DSP_HIFI_DYNAMIC_RANGE_CONTROL_H_
#define AUDIO_DSP_HIFI_DYNAMIC_RANGE_CONTROL_H_

#include <memory>

#include "audio/dsp/attack_release_envelope.h"
#include "glog/logging.h"
#include "third_party/eigen3/Eigen/Core"

#include "audio/dsp/porting.h"  // auto-added.


namespace audio_dsp {

// Chooses whether to use the peak or the RMS value of the signal when
// computing the envelope.
enum EnvelopeType {
  kPeak,
  kRms,
};

// Chooses the type of gain control that will be applied.
// See http://www.rane.com/note155.html for some explanation of each.
enum DynamicsControlType {
  // Compression reduces dynamic range above a certain level.
  kCompressor,
  // Limiters are a means of gain control to prevent the level from exceeding
  // the threshold (this is a much softer nonlinearity than a hard clipper).
  // For typical signals, a limiter will prevent the level from exceeding
  // the threshold very well, but is easy to come up with a highly impulsive
  // signal causes the output to exceed the threshold.
  kLimiter,
  // The expander increases the dynamic range when the signal is below the
  // threshold.
  kExpander,
  // Noise gates leave signals above the threshold alone and silence sounds that
  // are less than the threshold.
  kNoiseGate,
};

// NOTE: There is no one-size-fits-all solution for dynamic range control.
// The type of signal and the input level/desired output level will have a lot
// of impact over how the threshold, ratio, and gains should be set.
struct DynamicRangeControlParams {
  DynamicRangeControlParams()  // Mostly for putting params in a defined state.
      : envelope_type(kRms),
        dynamics_type(kCompressor),
        input_gain_db(0),
        output_gain_db(0),
        threshold_db(0),
        ratio(1),
        knee_width_db(0),
        attack_s(0.001f),
        release_s(0.05f) {}

  // See NOTE above.
  static DynamicRangeControlParams ReasonableCompressorParams() {
    DynamicRangeControlParams params;
    params.envelope_type = kRms;
    params.dynamics_type = kCompressor;
    params.input_gain_db = 0.0f;
    params.output_gain_db = 30.0f;
    params.threshold_db = -37.0f;
    params.ratio = 4.6f;
    params.knee_width_db = 4.0f;
    params.attack_s = 0.001f;
    params.release_s = 0.08f;
    return params;
  }

  // See NOTE above.
  static DynamicRangeControlParams ReasonableLimiterParams() {
    DynamicRangeControlParams params;
    params.envelope_type = kPeak;
    params.dynamics_type = kLimiter;
    params.input_gain_db = 0.0f;
    params.output_gain_db = 0.0f;
    params.threshold_db = -3.0f;
    params.knee_width_db = 1.0f;
    params.attack_s = 0.0004f;
    params.release_s = 0.004f;
    return params;
  }

  // Choose whether to use the peak value of the signal or the RMS when
  // computing the gain.
  EnvelopeType envelope_type;

  // Describes the type of gain control.
  DynamicsControlType dynamics_type;

  // Applied before the nonlinearity.
  float input_gain_db;

  // Applied at the output to make up for lost signal energy.
  float output_gain_db;

  // The amplitude in decibels (dBFS) at which the range control kicks in.
  // For compression and limiting, a signal below this threshold is not
  // scaled (ignoring the knee).
  // For a noise gate or an expander, a signal above the threshold is not
  // scaled (ignoring the knee).
  float threshold_db;

  // Except in a transitional knee around threshold_db, the input/output
  // relationship of a compressor is
  //   output_db = input_db, for input_db < threshold_db
  //   output_db = threshold_db + (input_db - threshold_db) / ratio,
  //     for input_db > threshold_db.
  // Likewise, for an expander
  //   output_db = input_db, for input_db > threshold_db
  //   output_db = threshold_db + (input_db - threshold_db) * ratio,
  //     for input_db < threshold_db.
  float ratio;  // Ignored for limiter and noise gate.

  // A gentle transition between into range control around the threshold.
  float knee_width_db;  // Two-sided (diameter).

  // Exponential time constants associated with the gain estimator. Depending
  // on the application, these may range from 1ms to over 1s.
  float attack_s;
  float release_s;
};

// Multichannel, feed-forward dynamic range control. Note that the gain
// adjustment is the same on each channel.
class DynamicRangeControl {
 public:
  explicit DynamicRangeControl(const DynamicRangeControlParams& params);

  // sample_rate_hz is the audio sample rate.
  void Init(int num_channels, int max_block_size_samples, float sample_rate_hz);

  void Reset();

  // InputType and OutputType are Eigen 2D blocks (ArrayXXf or MatrixXf) or
  // a similar mapped type containing audio samples in column-major format.
  // The number of rows must be equal to num_channels, and the number of cols
  // must be less than or equal to max_block_size_samples.
  //
  // Templating is necessary for supporting Eigen::Map output types.
  template <typename InputType, typename OutputType>
  void ProcessBlock(const InputType& input, OutputType* output) {
    static_assert(std::is_same<typename InputType::Scalar, float>::value,
                  "Scalar type must be float.");
    static_assert(std::is_same<typename OutputType::Scalar, float>::value,
                  "Scalar type must be float.");

    DCHECK_LE(input.cols(), max_block_size_samples_);
    DCHECK_EQ(input.rows(), num_channels_);
    DCHECK_EQ(input.cols(), output->cols());
    DCHECK_EQ(input.rows(), output->rows());

    // Map the needed amount of space for computations.
    VectorType workmap = workspace_.head(input.cols());

    // Compute the average power/amplitude across channels. The signal envelope
    // is monaural.
    if (params_.envelope_type == kRms) {
      workmap = input.square().colwise().mean();
    } else {
      workmap = input.abs().colwise().mean();
    }
    ComputeGain(&workmap);
    // Scale the input sample-wise by the gains.
    *output = input.rowwise() * workmap.transpose();
  }

 private:
  using VectorType = Eigen::VectorBlock<Eigen::ArrayXf, Eigen::Dynamic>;

  // Compute the linear gain to apply to the input.
  void ComputeGain(VectorType* data_ptr);
  // Defer gain computation to specific types of range control.
  void ApplyGainControl(const VectorType& input_level,
                        VectorType* output_gain);

  int num_channels_;
  float sample_rate_hz_;
  int max_block_size_samples_;

  Eigen::ArrayXf workspace_;
  Eigen::ArrayXf workspace_drc_output_;
  DynamicRangeControlParams params_;
  std::unique_ptr<AttackReleaseEnvelope> envelope_;
};

}  // namespace audio_dsp

#endif  // AUDIO_DSP_HIFI_DYNAMIC_RANGE_CONTROL_H_
