#include "audio/dsp/attack_release_envelope.h"

#include "gtest/gtest.h"

#include "audio/linear_filters/discretization.h"

#include "audio/dsp/porting.h"  // auto-added.


namespace audio_dsp {
namespace {

using ::linear_filters::FirstOrderCoefficientFromTimeConstant;

constexpr float kAttackSeconds = 5.0f;
constexpr float kReleaseSeconds = 0.2f;
constexpr float kSampleRateHz = 1000.0f;

TEST(AttackReleaseEnvelopeTest, ExponentialDecayTest) {
  AttackReleaseEnvelope envelope(
      kAttackSeconds, kReleaseSeconds, kSampleRateHz);

  int tau_samples = std::round(kSampleRateHz * kReleaseSeconds);
  const float alpha =
      FirstOrderCoefficientFromTimeConstant(kReleaseSeconds, kSampleRateHz);

  float first_sample = envelope.Output(1);  // An impulse.
  // Check for exponential decay.
  for (int i = 1; i < tau_samples; ++i){
    ASSERT_NEAR(envelope.Output(0),
                first_sample * std::pow(1 - alpha, i), 1e-4);
  }
  // The time constant is correct.
  EXPECT_NEAR(envelope.Output(0), first_sample / std::exp(1), 1e-4);
}

TEST(AttackReleaseEnvelopeTest, StepFunctionTest) {
  AttackReleaseEnvelope envelope(
      kAttackSeconds, kReleaseSeconds, kSampleRateHz);

  int tau_samples = std::round(kSampleRateHz * kAttackSeconds);
  const float alpha =
      FirstOrderCoefficientFromTimeConstant(kAttackSeconds, kSampleRateHz);

  // Check for output exponentially approaching input.
  for (int i = 0; i < tau_samples; ++i){
    // Use a step function.
    ASSERT_NEAR(1 - envelope.Output(1), std::pow(1 - alpha, i + 1), 1e-4);
  }
  // The time constant is correct.
  EXPECT_NEAR(1 - envelope.Output(1), 1.0 / std::exp(1), 1e-4);
}

}  // namespace
}  // namespace audio_dsp
