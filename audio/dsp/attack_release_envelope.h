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
#ifndef AUDIO_DSP_ATTACK_RELEASE_ENVELOPE_H_
#define AUDIO_DSP_ATTACK_RELEASE_ENVELOPE_H_

#include "audio/dsp/porting.h"  // auto-added.


namespace audio_dsp {

// Computes an approximate envelope of a rectified signal with an asymmetrical
// time constant.
// TODO: Add multichannel support.
// TODO: Add an Init function and make this class a little less
// bare-bones.
class AttackReleaseEnvelope {
 public:
  // attack_s and release_s are time constants for the filter in seconds. When
  // input > output, the attack coefficient is used. When input < output, the
  // release coefficient is used.
  AttackReleaseEnvelope(float attack_s, float release_s, float sample_rate_hz);

  void Reset() {
    envelope_ = 0;
  }
  // Process a single sample.
  float Output(float input);

 private:
  // State variable.
  float envelope_;

  float attack_;
  float release_;
};

}  // namespace audio_dsp

#endif  // AUDIO_DSP_ATTACK_RELEASE_ENVELOPE_H_
