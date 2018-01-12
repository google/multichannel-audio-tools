#include "audio/dsp/attack_release_envelope.h"

#include <cmath>

#include "audio/linear_filters/discretization.h"

namespace audio_dsp {

using ::linear_filters::FirstOrderCoefficientFromTimeConstant;

AttackReleaseEnvelope::AttackReleaseEnvelope(float attack_s,
                                             float release_s,
                                             float sample_rate_hz)
    :  envelope_(0.0),
       attack_(FirstOrderCoefficientFromTimeConstant(attack_s,
                                                     sample_rate_hz)),
       release_(FirstOrderCoefficientFromTimeConstant(release_s,
                                                      sample_rate_hz)) {}

float AttackReleaseEnvelope::Output(float input) {
  float rectified = std::abs(input);
  envelope_ +=
      (rectified > envelope_ ? attack_ : release_) * (rectified - envelope_);
  return envelope_;
}

}  // namespace audio_dsp
