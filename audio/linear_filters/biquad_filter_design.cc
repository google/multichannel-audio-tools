#include "audio/linear_filters/biquad_filter_design.h"

#include <cmath>

#include "audio/linear_filters/discretization.h"

namespace linear_filters {
namespace {

void CheckArguments(double sample_rate_hz, double corner_frequency_hz,
                    double quality_factor) {
  CHECK_LT(corner_frequency_hz, sample_rate_hz / 2);
  CHECK_GT(corner_frequency_hz, 0.0);
  CHECK_GT(quality_factor, 0.0);
}

void CheckArgumentsBandEdges(double sample_rate_hz, double lower_band_edge_hz,
                             double upper_band_edge_hz) {
  CHECK_GT(sample_rate_hz / 2, upper_band_edge_hz);
  CHECK_GT(upper_band_edge_hz, lower_band_edge_hz);
  CHECK_GT(lower_band_edge_hz, 0.0);
}
}  // namespace

BiquadFilterCoefficients LowpassBiquadFilterCoefficients(
    double sample_rate_hz, double corner_frequency_hz, double quality_factor) {
  CheckArguments(sample_rate_hz, corner_frequency_hz, quality_factor);
  // New alternative design approach based on explicit bilinear transform of
  // s-domain numerator and denominator:
  const double omega_n = 2.0 * M_PI * corner_frequency_hz;
  return BilinearTransform(
      {0.0, 0.0, omega_n * omega_n},
      {1.0, omega_n / quality_factor, omega_n * omega_n},
      sample_rate_hz, corner_frequency_hz);
}

BiquadFilterCoefficients HighpassBiquadFilterCoefficients(
    double sample_rate_hz, double corner_frequency_hz, double quality_factor) {
  CheckArguments(sample_rate_hz, corner_frequency_hz, quality_factor);
  // New alternative design approach based on explicit bilinear transform of
  // s-domain numerator and denominator:
  const double omega_n = 2.0 * M_PI * corner_frequency_hz;
  return BilinearTransform(
      {1.0, 0.0, 0.0},
      {1.0, omega_n / quality_factor, omega_n * omega_n},
      sample_rate_hz, corner_frequency_hz);
}

BiquadFilterCoefficients BandpassBiquadFilterCoefficients(
    double sample_rate_hz, double center_frequency_hz, double quality_factor) {
  CheckArguments(sample_rate_hz, center_frequency_hz, quality_factor);
  const double omega_n = 2.0 * M_PI * center_frequency_hz;
  return BilinearTransform(
      {0.0, omega_n / quality_factor, 0.0},
      {1.0, omega_n / quality_factor, omega_n * omega_n},
      sample_rate_hz, center_frequency_hz);
}

BiquadFilterCoefficients BandstopBiquadFilterCoefficients(
    double sample_rate_hz, double center_frequency_hz, double quality_factor) {
  CheckArguments(sample_rate_hz, center_frequency_hz, quality_factor);
  const double omega_n = 2.0 * M_PI * center_frequency_hz;
  return BilinearTransform(
      {1.0, 0.0, omega_n * omega_n},
      {1.0, omega_n / quality_factor, omega_n * omega_n},
      sample_rate_hz, center_frequency_hz);
}

BiquadFilterCoefficients RangedBandpassBiquadFilterCoefficients(
    double sample_rate_hz, double lower_passband_edge_hz,
    double upper_passband_edge_hz) {
  CheckArgumentsBandEdges(sample_rate_hz, lower_passband_edge_hz,
                          upper_passband_edge_hz);
  // Prewarp the band edges rather than warping the transform to match at
  // one frequeny.
  const double omega_1 = BilinearPrewarp(lower_passband_edge_hz,
                                         sample_rate_hz);
  const double omega_2 = BilinearPrewarp(upper_passband_edge_hz,
                                         sample_rate_hz);
  return BilinearTransform(
      {0.0, omega_2 - omega_1, 0.0},
      {1.0, omega_2 - omega_1, omega_2 * omega_1},
      sample_rate_hz, 0.0);
}

BiquadFilterCoefficients RangedBandstopBiquadFilterCoefficients(
    double sample_rate_hz, double lower_stopband_edge_hz,
    double upper_stopband_edge_hz) {
  CheckArgumentsBandEdges(sample_rate_hz, lower_stopband_edge_hz,
                          upper_stopband_edge_hz);
  // Prewarp the band edges rather than warping the transform to match at
  // one frequeny.
  const double omega_1 = BilinearPrewarp(lower_stopband_edge_hz,
                                         sample_rate_hz);
  const double omega_2 = BilinearPrewarp(upper_stopband_edge_hz,
                                         sample_rate_hz);
  return BilinearTransform(
      {1.0, 0.0, omega_2 * omega_1},
      {1.0, omega_2 - omega_1, omega_2 * omega_1},
      sample_rate_hz, 0.0);
}

BiquadFilterCoefficients LowShelfBiquadFilterCoefficients(
    float sample_rate_hz,
    float corner_frequency_hz,
    float Q,
    float gain) {
  CheckArguments(sample_rate_hz, corner_frequency_hz, Q);
  CHECK_GT(gain, 0);
  const double sqrtk = std::sqrt(gain);
  const double omega = 2 * M_PI * corner_frequency_hz / sample_rate_hz;
  const double beta = std::sin(omega) * std::sqrt (sqrtk) / Q;

  const double sqrtk_minus_one_cos_omega = (sqrtk - 1) * std::cos(omega);
  const double sqrtk_plus_one_cos_omega = (sqrtk + 1) * std::cos(omega);

  return {{sqrtk * ((sqrtk + 1) - sqrtk_minus_one_cos_omega + beta),
           sqrtk * 2.0 * ((sqrtk - 1) - sqrtk_plus_one_cos_omega),
           sqrtk * ((sqrtk + 1) - sqrtk_minus_one_cos_omega - beta)},
          {(sqrtk + 1) + sqrtk_minus_one_cos_omega + beta,
           -2.0 * ((sqrtk - 1) + sqrtk_plus_one_cos_omega),
           (sqrtk + 1) + sqrtk_minus_one_cos_omega - beta}};
}

BiquadFilterCoefficients HighShelfBiquadFilterCoefficients(
    float sample_rate_hz,
    float corner_frequency_hz,
    float Q,
    float gain) {
  CheckArguments(sample_rate_hz, corner_frequency_hz, Q);
  CHECK_GT(gain, 0);
  const double sqrtk = std::sqrt(gain);
  const double omega = 2 * M_PI * corner_frequency_hz / sample_rate_hz;
  const double beta = std::sin (omega) * std::sqrt(sqrtk) / Q;

  const double sqrtk_minus_one_cos_omega = (sqrtk - 1) * std::cos(omega);
  const double sqrtk_plus_one_cos_omega = (sqrtk + 1) * std::cos(omega);

  return {{sqrtk * ((sqrtk + 1) + sqrtk_minus_one_cos_omega + beta),
           sqrtk * -2.0 * ((sqrtk - 1) + sqrtk_plus_one_cos_omega),
           sqrtk * ((sqrtk + 1) + sqrtk_minus_one_cos_omega - beta)},
          {(sqrtk + 1) - sqrtk_minus_one_cos_omega + beta,
           2.0 * ((sqrtk - 1) - sqrtk_plus_one_cos_omega),
           (sqrtk + 1) - sqrtk_minus_one_cos_omega - beta}};
}

BiquadFilterCoefficients ParametricPeakBiquadFilterCoefficients(
    float sample_rate_hz,
    float center_frequency_hz,
    float Q,
    float gain) {
  CheckArguments(sample_rate_hz, center_frequency_hz, Q);
  CHECK_GE(gain, 0);
  const double omega = 2 * M_PI * center_frequency_hz / sample_rate_hz;
  const double alpha = std::sin(omega) / (2 * Q);

  const double a2 = (1.0 - alpha) / (1.0 + alpha);
  const double b1 = -(1 + a2) * std::cos(omega);
  return {{0.5 * ((1.0 + a2) + (1.0 - a2) * gain),
           -(1 + a2) * std::cos(omega),
           0.5 * ((1.0 + a2) - (1.0 - a2) * gain)},
          {1.0,  b1, (1.0 - alpha) / (1.0 + alpha)}};
}

// Uses the notation from:
// http://faculty.tru.ca/rtaylor/publications/allpass2_align.pdf
BiquadFilterCoefficients AllpassBiquadFilterCoefficients(
    float sample_rate_hz,
    float corner_frequency_hz,
    float quality_factor) {
  CHECK_LT(corner_frequency_hz, sample_rate_hz / 2);
  CHECK_GT(corner_frequency_hz, 0.0);
  CHECK_GT(quality_factor, 0);
  const double p = M_PI * corner_frequency_hz / sample_rate_hz;
  const double q_p_sq_plus_one = quality_factor * (p * p + 1);
  const double b0 = (q_p_sq_plus_one - p) / (q_p_sq_plus_one + p);
  const double b1 = 2 * (quality_factor * (p * p - 1)) / (q_p_sq_plus_one + p);
  // Numerator and denominator are complex conjugates of each other.
  return {{b0, b1, 1.0}, {1.0, b1, b0}};
}
}  // namespace linear_filters
