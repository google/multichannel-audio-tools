#include "audio/linear_filters/biquad_filter_design.h"

#include <complex>
#include <vector>

#include "audio/dsp/testing_util.h"
#include "audio/linear_filters/biquad_filter_test_tools.h"
#include "gtest/gtest.h"


#include "audio/dsp/porting.h"  // auto-added.


namespace linear_filters {

using ::std::complex;
using ::std::vector;
using ::testing::DoubleNear;
using ::testing::Le;
using ::util::format::StringF;

namespace {

constexpr float kSampleRateHz = 48000.0f;
constexpr float kNyquistHz = kSampleRateHz / 2;
constexpr int kNumPoints = 40;  // Points to check for monotonicity.

// This test verifies that the DC gain of the filter is unity, the
// high frequency gain is zero, and the gain at the corner frequency is equal
// to the quality factor. The response monotonically decreases above the cutoff.
TEST(BiquadFilterDesignTest, LowpassCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float corner_frequency_hz : {10.0f, 100.0f, 1000.0f, 10000.0f}) {
    for (float quality_factor : {0.707f, 3.0f, 10.0f}) {
      BiquadFilterCoefficients coeffs =
          LowpassBiquadFilterCoefficients(kSampleRateHz,
                                          corner_frequency_hz,
                                          quality_factor);
      SCOPED_TRACE(StringF("Lowpass (Q = %f) with corner = %f.",
                           quality_factor, corner_frequency_hz));
      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                              0.0f, kSampleRateHz));
      ASSERT_THAT(coeffs,
                  MagnitudeResponseIs(DoubleNear(quality_factor, kTolerance),
                                      corner_frequency_hz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(0.0, kTolerance),
                                              kNyquistHz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
          corner_frequency_hz, kNyquistHz, kSampleRateHz, kNumPoints));
    }
  }
}

// This test verifies that the high frequency gain of the filter is unity, the
// DC is zero, and the gain at the corner frequency is equal to the quality
// factor. The response monotonically increases below the cutoff.
TEST(BiquadFilterDesignTest, HighpassCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float corner_frequency_hz : {100.0f, 1000.0f, 10000.0f}) {
    for (float quality_factor : {0.707f, 3.0f, 10.0f}) {
      BiquadFilterCoefficients coeffs =
           HighpassBiquadFilterCoefficients(kSampleRateHz,
                                            corner_frequency_hz,
                                            quality_factor);
      SCOPED_TRACE(StringF("Highpass (Q = %f) with corner = %f.",
                           quality_factor, corner_frequency_hz));
      ASSERT_THAT(coeffs,
                  MagnitudeResponseIs(DoubleNear(0.0, kTolerance),
                                      0.0f, kSampleRateHz));
      ASSERT_THAT(coeffs,
                  MagnitudeResponseIs(DoubleNear(quality_factor, kTolerance),
                                      corner_frequency_hz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                              kNyquistHz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
          20.0f, corner_frequency_hz, kSampleRateHz, kNumPoints));
    }
  }
}

// Verifies that the shape of the bandpass filter is such that around the
// center frequency, frequencies are passed with unity gain and away from that
// frequency, we see monotonically decreasing response.
TEST(BiquadFilterDesignTest, BandpassCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float center_frequency_hz : {100.0f, 1000.0f, 10000.0f}) {
    for (float quality_factor : {0.707f, 3.0f, 10.0f}) {
      BiquadFilterCoefficients coeffs =
           BandpassBiquadFilterCoefficients(kSampleRateHz,
                                            center_frequency_hz,
                                            quality_factor);
      SCOPED_TRACE(
          StringF("Bandpass (Q = %f) with center frequency = %f.",
                  quality_factor,
                  center_frequency_hz));
      ASSERT_THAT(coeffs,
                  MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                      center_frequency_hz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
          20.0f, center_frequency_hz, kSampleRateHz, kNumPoints));
      ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
          center_frequency_hz, kNyquistHz, kSampleRateHz, kNumPoints));
    }
  }
}

// Verifies that the shape of the bandstop filter is such that around the
// center frequency, frequencies are blocked, and moving away from that
// frequency we see monotonically increasing response.
TEST(BiquadFilterDesignTest, BandstopCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float center_frequency_hz : {100.0f, 1000.0f, 10000.0f}) {
    for (float quality_factor : {0.707f, 3.0f, 10.0f}) {
      BiquadFilterCoefficients coeffs =
          BandstopBiquadFilterCoefficients(kSampleRateHz,
                                           center_frequency_hz,
                                           quality_factor);
      SCOPED_TRACE(
          StringF("Bandstop (Q = %f) with center frequency = %f.",
                  quality_factor,
                  center_frequency_hz));
      ASSERT_THAT(coeffs,
                  MagnitudeResponseIs(DoubleNear(0.0, kTolerance),
                                      center_frequency_hz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
          20.0f, center_frequency_hz, kSampleRateHz, kNumPoints));
      ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
          center_frequency_hz, kNyquistHz, kSampleRateHz, kNumPoints));
    }
  }
}

// Verifies that the shape of the bandpass filter is such that around some
// center frequency, frequencies are passed with unity gain and moving away from
// that frequency, we see monotonically decreasing response. These filters are
// specified by their band edges rather than a center frequency and a quality
// factor.
TEST(BiquadFilterDesignTest, RangedBandpassCoefficientsTest) {
  // These loops are a construct to give us reasonable upper and lower cutoffs.
  // The band edges used in this test are:
  //   approximate_center_hz +/- approximate_half bandwidth.
  for (float approximate_center_hz : {100.0f, 1000.0f, 10000.0f}) {
    for (float approximate_half_bandwidth : {1.0f, 15.0f, 50.0f, 200.0f}) {
      if (approximate_half_bandwidth > approximate_center_hz / 2) { continue; }

      const float lower_band_edge_hz =
          approximate_center_hz - approximate_half_bandwidth;
      const float upper_band_edge_hz =
          approximate_center_hz + approximate_half_bandwidth;
      BiquadFilterCoefficients coeffs =
          RangedBandpassBiquadFilterCoefficients(kSampleRateHz,
                                                 lower_band_edge_hz,
                                                 upper_band_edge_hz);
      SCOPED_TRACE(
          StringF("Ranged bandpass with approximate center = %f and "
                  "half bandwidth %f",
                  approximate_center_hz,
                  approximate_half_bandwidth));

      // The actual center of the filter is located near the geometric mean of
      // the cutoff specifications.
      const float better_approximation_center =
          sqrt(lower_band_edge_hz * upper_band_edge_hz);

      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(0.95, 0.051),
                                              better_approximation_center,
                                              kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
          lower_band_edge_hz, better_approximation_center,
          kSampleRateHz, kNumPoints));
      ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
          better_approximation_center, upper_band_edge_hz,
          kSampleRateHz, kNumPoints));
    }
  }
}

// Verifies that the shape of the bandstop filter is such that around some
// center frequency, frequencies are blocked (near zero gain) and away from
// that frequency, we see monotonically increasing response. These filters are
// specified by their band edges rather than a center frequency and a quality
// factor.
TEST(BiquadFilterDesignTest, RangedBandstopCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  std::vector<double> coeffs_b, coeffs_a;
  // These loops are a construct to give us reasonable upper and lower cutoffs.
  // The band edges used in this test are:
  //   approximate_center_hz +/- approximate_half bandwidth.
  for (float approximate_center_hz : {100.0f, 1000.0f, 10000.0f}) {
    for (float approximate_half_bandwidth : {1.0f, 15.0f, 50.0f, 200.0f}) {
      if (approximate_half_bandwidth > approximate_center_hz / 2) { continue; }

      const float lower_band_edge_hz =
          approximate_center_hz - approximate_half_bandwidth;
      const float upper_band_edge_hz =
          approximate_center_hz + approximate_half_bandwidth;
      BiquadFilterCoefficients coeffs =
          RangedBandstopBiquadFilterCoefficients(kSampleRateHz,
                                                 lower_band_edge_hz,
                                                 upper_band_edge_hz);
      SCOPED_TRACE(
          StringF("Ranged bandstop with approximate center = %f and "
                  "half bandwidth %f",
                  approximate_center_hz,
                  approximate_half_bandwidth));

      // The actual center of the filter is located near the geometric mean of
      // the cutoff specifications.
      const float better_approximation_center =
          sqrt(lower_band_edge_hz * upper_band_edge_hz);

      ASSERT_THAT(coeffs, MagnitudeResponseIs(Le(0.01),
                                              better_approximation_center,
                                              kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                              0.0f, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                              kNyquistHz, kSampleRateHz));

      ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
          lower_band_edge_hz, better_approximation_center,
          kSampleRateHz, kNumPoints));
      ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
          better_approximation_center, upper_band_edge_hz,
          kSampleRateHz, kNumPoints));
    }
  }
}

// Verifies that the shape of the low shelf filter is such that below the
// corner frequency, the gain is set to a specified constant, and above that
// frequency, the gain is unity (with some transition region.
TEST(BiquadFilterDesignTest, LowShelfCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float corner_frequency_hz : {500.0f, 2000.0f, 10000.0f}) {
    for (float quality_factor : {0.707f, 2.0f}) {
      for (float gain : {0.5, 2.0}) {
        BiquadFilterCoefficients coeffs =
            LowShelfBiquadFilterCoefficients(kSampleRateHz,
                                             corner_frequency_hz,
                                             quality_factor,
                                             gain);
        SCOPED_TRACE(
            StringF("LowShelf (Q = %f) with center frequency = %f and gain %f.",
                    quality_factor, corner_frequency_hz, gain));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                        kSampleRateHz / 2, kSampleRateHz));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(std::sqrt(gain), kTolerance),
                                        corner_frequency_hz, kSampleRateHz));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(gain, kTolerance),
                                        0, kSampleRateHz));
        if (quality_factor < 1 / M_SQRT2) {
          if (gain < 1.0) {
            ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
                20.0f, kSampleRateHz / 2, kSampleRateHz, kNumPoints));
          } else {
            ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
                20.0f, kSampleRateHz / 2, kSampleRateHz, kNumPoints));
          }
        }
      }
    }
  }
}

// Verifies that the shape of the high shelf filter is such that above the
// center frequency, the gain is set to a specified constant, and below that
// frequency, the gain is unity (with some transition region.
TEST(BiquadFilterDesignTest, HighShelfCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float center_frequency_hz : {500.0f, 2000.0f, 10000.0f}) {
    for (float quality_factor : {0.707f, 3.0f}) {
      for (float gain : {0.5, 2.0}) {
        BiquadFilterCoefficients coeffs =
            HighShelfBiquadFilterCoefficients(kSampleRateHz,
                                              center_frequency_hz,
                                              quality_factor,
                                              gain);
        SCOPED_TRACE(
            StringF("LowShelf (Q = %f) with center frequency = %f and gain %f.",
                    quality_factor, center_frequency_hz, gain));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                        0, kSampleRateHz));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(std::sqrt(gain), kTolerance),
                                        center_frequency_hz, kSampleRateHz));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(gain, kTolerance),
                                        kSampleRateHz / 2, kSampleRateHz));
        if (quality_factor < 1/ M_SQRT2) {
          if (gain < 1.0) {
            ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
                20.0f, kSampleRateHz / 2, kSampleRateHz, kNumPoints));
          } else {
            ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
                20.0f, kSampleRateHz / 2, kSampleRateHz, kNumPoints));
          }
        }
      }
    }
  }
}

// Verifies that the shape of the peak filter is such that at the
// center frequency, the gain is set to a specified constant, and away from that
// frequency, the gain is unity (with some transition region.
TEST(BiquadFilterDesignTest, ParametricPeakCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float center_frequency_hz : {500.0f, 2000.0f, 10000.0f}) {
    for (float quality_factor : {0.707f, 3.0f}) {
      for (float gain : {0.5, 2.0}) {
        BiquadFilterCoefficients coeffs =
            ParametricPeakBiquadFilterCoefficients(kSampleRateHz,
                                                   center_frequency_hz,
                                                   quality_factor,
                                                   gain);
        SCOPED_TRACE(
            StringF("LowShelf (Q = %f) with center frequency = %f and gain %f.",
                    quality_factor, center_frequency_hz, gain));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                        0, kSampleRateHz));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(gain, kTolerance),
                                        center_frequency_hz, kSampleRateHz));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                        kSampleRateHz / 2, kSampleRateHz));
        if (gain < 1.0) {
          ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
              20.0f, center_frequency_hz, kSampleRateHz, kNumPoints));
          ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
              center_frequency_hz, kSampleRateHz / 2, kSampleRateHz,
              kNumPoints));
        } else {
          ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
              20.0f, center_frequency_hz, kSampleRateHz, kNumPoints));
          ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
              center_frequency_hz, kSampleRateHz / 2, kSampleRateHz,
              kNumPoints));
        }
      }
    }
  }
}

// Verifies that the allpass filter is flat across the entire bandwidth.
TEST(BiquadFilterDesignTest, AllpassCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (double corner_frequency_hz : {100.0f, 1000.0f, 10000.0f}) {
    for (double quality_factor : {0.2f, 0.9f, 2.0f}) {
      BiquadFilterCoefficients coeffs =
          AllpassBiquadFilterCoefficients(kSampleRateHz,
                                          corner_frequency_hz,
                                          quality_factor);
      SCOPED_TRACE(
          StringF("Allpass (quality_factor = %f) with pole frequency = %f.",
                  quality_factor, corner_frequency_hz));
      // There is a discontinuity at corner_frequency_hz where the phase
      // switches from -pi to pi radians.
      ASSERT_THAT(coeffs, PhaseResponseDecreases(
                  20.0f, 0.99 * corner_frequency_hz, kSampleRateHz,
                  kNumPoints));
      ASSERT_THAT(coeffs, PhaseResponseDecreases(
                  1.01 * corner_frequency_hz, kSampleRateHz / 2, kSampleRateHz,
                  kNumPoints));

      for (int i = 0; i < kNumPoints; ++i) {
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                        i * kSampleRateHz / (2.0 * kNumPoints),
                                        kSampleRateHz));
      }
    }
  }
  BiquadFilterCoefficients spot_check =
      AllpassBiquadFilterCoefficients(kSampleRateHz, 100, 0.2);
  ASSERT_THAT(spot_check, PhaseResponseIs(DoubleNear(-0.93536, 1e-3),
                                          10, kSampleRateHz));
  ASSERT_THAT(spot_check, PhaseResponseIs(DoubleNear(0.0852, 1e-3),
                                          10000, kSampleRateHz));
}

string ToString(const complex<double>& value) {
  return util::format::StringF("%g+%gj", value.real(), value.imag());
}

// TODO: Replace this with a matcher once there is a tolerance that
// works well for all of the tests.
void AssertRootNear(std::complex<double> actual,
                    std::complex<double> expected,
                    double tol = 1e-6) {
  SCOPED_TRACE("Found root " + ToString(actual) + " but was expecting " +
               ToString(expected) + ".");
  ASSERT_NEAR(actual.real(), expected.real(), tol);
  ASSERT_NEAR(actual.imag(), expected.imag(), tol);
}

TEST(PoleZeroFilterDesignTest, ButterworthAnalogPrototype) {
  // Compare with scipy.signal.cheby1(n, 0.25, 1, analog=True, output="zpk").
  // First order.
  FilterPolesAndZeros zpk1 = ButterworthFilterDesign(1).GetAnalogPrototype();
  EXPECT_EQ(zpk1.GetPolesDegree(), 1);
  EXPECT_EQ(zpk1.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk1.GetGain(), 1, 1e-9);
  EXPECT_EQ(zpk1.GetRealPoles()[0], -1.0);
  // Second order.
  FilterPolesAndZeros zpk2 = ButterworthFilterDesign(2).GetAnalogPrototype();
  EXPECT_EQ(zpk2.GetPolesDegree(), 2);
  EXPECT_EQ(zpk2.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk2.GetGain(), 1, 1e-9);
  AssertRootNear(zpk2.GetConjugatedPoles()[0],
                 {-0.707106781187, 0.707106781187});
  // Third order.
  FilterPolesAndZeros zpk3 = ButterworthFilterDesign(3).GetAnalogPrototype();
  EXPECT_EQ(zpk3.GetPolesDegree(), 3);
  EXPECT_EQ(zpk3.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk3.GetGain(), 1, 1e-9);
  EXPECT_EQ(zpk3.GetRealPoles()[0], -1.0);
  AssertRootNear(zpk3.GetConjugatedPoles()[0], {-0.5, 0.866025403784});
  // Seventh order.
  FilterPolesAndZeros zpk7 = ButterworthFilterDesign(7).GetAnalogPrototype();
  EXPECT_EQ(zpk7.GetPolesDegree(), 7);
  EXPECT_EQ(zpk7.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk7.GetGain(), 1.0, 1e-9);
  EXPECT_EQ(zpk7.GetRealPoles()[0], -1.0);
  AssertRootNear(zpk7.GetConjugatedPoles()[0],
                 {-0.900968867902, 0.433883739118});
  AssertRootNear(zpk7.GetConjugatedPoles()[1],
                 {-0.623489801859, 0.781831482468});
  AssertRootNear(zpk7.GetConjugatedPoles()[2],
                 {-0.222520933956, 0.974927912182});
}

TEST(PoleZeroFilterDesignTest, Chebyshev1AnalogPrototype) {
  // Compare with scipy.signal.cheby1(n, 0.25, 1, analog=True, output="zpk").
  // First order.
  FilterPolesAndZeros zpk1 =
      ChebyshevType1FilterDesign(1, 0.25).GetAnalogPrototype();
  EXPECT_EQ(zpk1.GetPolesDegree(), 1);
  EXPECT_EQ(zpk1.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk1.GetGain(), 4.1081110, 1e-6);
  EXPECT_NEAR(zpk1.GetRealPoles()[0], -4.10811100915, 1e-6);
  // Second order.
  FilterPolesAndZeros zpk2 =
      ChebyshevType1FilterDesign(2, 0.25).GetAnalogPrototype();
  EXPECT_EQ(zpk2.GetPolesDegree(), 2);
  EXPECT_EQ(zpk2.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk2.GetGain(), 2.054055504, 1e-6);
  AssertRootNear(zpk2.GetConjugatedPoles()[0],
                 {-0.898341529763, 1.14324866241});
  // Third order.
  FilterPolesAndZeros zpk3 =
      ChebyshevType1FilterDesign(3, 0.25).GetAnalogPrototype();
  EXPECT_EQ(zpk3.GetPolesDegree(), 3);
  EXPECT_EQ(zpk3.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk3.GetRealPoles()[0], -0.767222665927, 1e-6);
  EXPECT_NEAR(zpk3.GetGain(), 1.02702775228, 1e-6);
  AssertRootNear(zpk3.GetConjugatedPoles()[0],
                 {-0.383611332964, 1.09154613477});
  // Seventh order.
  FilterPolesAndZeros zpk7 =
      ChebyshevType1FilterDesign(7, 0.25).GetAnalogPrototype();
  EXPECT_EQ(zpk7.GetPolesDegree(), 7);
  EXPECT_EQ(zpk7.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk7.GetRealPoles()[0], -0.307598675776, 1e-6);
  EXPECT_NEAR(zpk7.GetGain(), 0.0641892345179, 1e-6);
  AssertRootNear(zpk7.GetConjugatedPoles()[0],
                 {-0.0684471446174, 1.02000802334});
  AssertRootNear(zpk7.GetConjugatedPoles()[1],
                 {-0.191784637412, 0.817982924742});
  AssertRootNear(zpk7.GetConjugatedPoles()[2],
                 {-0.277136830682, 0.453946275994});
}

TEST(PoleZeroFilterDesignTest, Chebyshev2AnalogPrototype) {
  // Compare with scipy.signal.cheby2(n, 0.25, 1, analog=True, output="zpk").
  // First order.
  FilterPolesAndZeros zpk1 =
      ChebyshevType2FilterDesign(1, 0.25).GetAnalogPrototype();
  EXPECT_EQ(zpk1.GetPolesDegree(), 1);
  EXPECT_EQ(zpk1.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk1.GetGain(), 4.10811100915, 1e-6);
  EXPECT_NEAR(zpk1.GetRealPoles()[0], -4.10811100915, 1e-6);
  // Second order.
  FilterPolesAndZeros zpk2 =
      ChebyshevType2FilterDesign(2, 0.25).GetAnalogPrototype();
  EXPECT_EQ(zpk2.GetPolesDegree(), 2);
  EXPECT_EQ(zpk2.GetZerosDegree(), 2);
  EXPECT_NEAR(zpk2.GetGain(), 0.971627951577, 1e-6);
  AssertRootNear(zpk2.GetConjugatedZeros()[0], {0, 1.41421356});
  AssertRootNear(zpk2.GetConjugatedPoles()[0],
                 {-0.16603335596, 1.38408411156});
  // Third order.
  FilterPolesAndZeros zpk3 =
      ChebyshevType2FilterDesign(3, 0.25).GetAnalogPrototype();
  EXPECT_EQ(zpk3.GetPolesDegree(), 3);
  EXPECT_NEAR(zpk3.GetRealPoles()[0], -12.4306769322, 1e-6);
  EXPECT_NEAR(zpk3.GetGain(), 12.3243330274, 1e-6);
  AssertRootNear(zpk3.GetConjugatedZeros()[0], {0, 1.15470054});
  AssertRootNear(zpk3.GetConjugatedPoles()[0],
                 {-0.0531719523966, 1.14852055605});
  // Seventh order.
  FilterPolesAndZeros zpk7 =
      ChebyshevType2FilterDesign(7, 0.25).GetAnalogPrototype();
  EXPECT_EQ(zpk7.GetPolesDegree(), 7);
  EXPECT_NEAR(zpk7.GetRealPoles()[0], -29.0304010997, 1e-6);
  EXPECT_NEAR(zpk7.GetGain(), 28.756777064, 1e-6);
  AssertRootNear(zpk7.GetConjugatedZeros()[0], {-0, 1.02571686});
  AssertRootNear(zpk7.GetConjugatedZeros()[1], {-0, 1.27904801});
  AssertRootNear(zpk7.GetConjugatedZeros()[2], {-0, 2.30476487});
  AssertRootNear(zpk7.GetConjugatedPoles()[0],
                 {-0.00805435931033, 1.02504557345});
  AssertRootNear(zpk7.GetConjugatedPoles()[1],
                 {-0.0350677400886, 1.27732709156});
  AssertRootNear(zpk7.GetConjugatedPoles()[2],
                 {-0.163825398625, 2.29168734868});
}

// TODO: A lot of these tolerances had to be loosened to pass.
// It's not clear why.
TEST(PoleZeroFilterDesignTest, EllipticAnalogPrototype) {
  // Compare with scipy.signal.cheby2(n, 0.25, 0.35, 1,
  //                                  analog=True, output="zpk").
  // First order.
  FilterPolesAndZeros zpk1 =
      EllipticFilterDesign(1, 0.25, 0.35).GetAnalogPrototype();
  EXPECT_EQ(zpk1.GetPolesDegree(), 1);
  EXPECT_EQ(zpk1.GetZerosDegree(), 0);
  EXPECT_NEAR(zpk1.GetGain(), 4.10811101, 1e-6);
  EXPECT_NEAR(zpk1.GetRealPoles()[0], -4.10811100914, 1e-6);
  // Second order.
  FilterPolesAndZeros zpk2 =
      EllipticFilterDesign(2, 0.25, 0.35).GetAnalogPrototype();
  EXPECT_EQ(zpk2.GetPolesDegree(), 2);
  EXPECT_EQ(zpk2.GetZerosDegree(), 2);
  EXPECT_NEAR(zpk2.GetGain(), 0.960513960388, 1e-5);
  AssertRootNear(zpk2.GetConjugatedZeros()[0], {0, 1.04644836}, 1e-5);
  AssertRootNear(zpk2.GetConjugatedPoles()[0],
                 {-0.02246578, 1.04020366}, 1e-5);
  // Third order.
  FilterPolesAndZeros zpk3 =
      EllipticFilterDesign(3, 0.25, 0.35).GetAnalogPrototype();
  EXPECT_EQ(zpk3.GetPolesDegree(), 3);
  EXPECT_NEAR(zpk3.GetRealPoles()[0], -3.75707890, 3e-3);
  EXPECT_NEAR(zpk3.GetGain(), 3.7561382314, 3e-3);
  AssertRootNear(zpk3.GetConjugatedZeros()[0], {0, 1.00098779});
  AssertRootNear(zpk3.GetConjugatedPoles()[0],
                 {-4.70332379e-04, 1.00086237});
  // Seventh order.
  FilterPolesAndZeros zpk7 =
      EllipticFilterDesign(7, 0.25, 0.35).GetAnalogPrototype();
  EXPECT_EQ(zpk7.GetPolesDegree(), 7);
  EXPECT_NEAR(zpk7.GetRealPoles()[0], -3.75516861, 5e-3);
  EXPECT_NEAR(zpk7.GetGain(), 3.754227606, 5e-3);
  AssertRootNear(zpk7.GetConjugatedZeros()[0], {-0, 1.00098779}, 1e-3);
  AssertRootNear(zpk7.GetConjugatedZeros()[1], {-0, 1.00000045}, 1e-3);
  AssertRootNear(zpk7.GetConjugatedZeros()[2], {-0, 1.0}, 1e-3);
  AssertRootNear(zpk7.GetConjugatedPoles()[0],
                 {-1.05115740e-10, 1});
  AssertRootNear(zpk7.GetConjugatedPoles()[1],
                 {-2.22403043e-07, 1.00000039});
  AssertRootNear(zpk7.GetConjugatedPoles()[2],
                 {-4.70721817e-04, 1.0008207});
}

// These tests are more thorough than the typed tests below.
TEST(PoleZeroFilterDesignTest, ButterworthLowpassTest) {
  constexpr double kTolerance = 1e-4;
  for (float corner_frequency_hz : {10.0f, 100.0f, 1000.0f, 10000.0f}) {
    for (int order : {2, 5, 8}) {
      BiquadFilterCascadeCoefficients coeffs =
          ButterworthFilterDesign(order).LowpassCoefficients(
              kSampleRateHz, corner_frequency_hz);
      SCOPED_TRACE(StringF("Butterworth lowpass with corner = %f and order %d.",
                           corner_frequency_hz, order));
      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                              0.0f, kSampleRateHz));
      ASSERT_THAT(coeffs,
                  MagnitudeResponseIs(DoubleNear(1 / M_SQRT2, kTolerance),
                                      corner_frequency_hz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(0.0, kTolerance),
                                              kNyquistHz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
          corner_frequency_hz, kNyquistHz, kSampleRateHz, kNumPoints));
    }
  }
}

TEST(PoleZeroFilterDesignTest, ButterworthHighpassCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float corner_frequency_hz : {100.0f, 1000.0f, 10000.0f}) {
    for (int order : {2, 5, 8}) {
      BiquadFilterCascadeCoefficients coeffs =
          ButterworthFilterDesign(order).HighpassCoefficients(
              kSampleRateHz, corner_frequency_hz);
      SCOPED_TRACE(
          StringF("Butterworth highpass with corner = %f and order %d.",
                  corner_frequency_hz, order));
      ASSERT_THAT(coeffs,
                  MagnitudeResponseIs(DoubleNear(0.0, kTolerance),
                                      0.0f, kSampleRateHz));
      ASSERT_THAT(coeffs,
                  MagnitudeResponseIs(DoubleNear(1 / M_SQRT2, kTolerance),
                                      corner_frequency_hz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                              kNyquistHz, kSampleRateHz));
      ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
          20.0f, corner_frequency_hz, kSampleRateHz, kNumPoints));
    }
  }
}

TEST(PoleZeroFilterDesignTest, ButterworthBandpassCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float low_frequency_hz : {1000.0f, 10000.0f}) {
    for (int order : {2, 3, 4}) {
      for (float bandwidth_hz : {200.0f, 400.0f}) {
        const float high_frequency_hz = low_frequency_hz + bandwidth_hz;
        const float center_frequency_hz =
            std::sqrt(low_frequency_hz * high_frequency_hz);
        BiquadFilterCascadeCoefficients coeffs =
            ButterworthFilterDesign(order).BandpassCoefficients(
                kSampleRateHz, low_frequency_hz, high_frequency_hz);
        SCOPED_TRACE(
            StringF("Butterworth bandpass (order %d) with range = [%f, %f].",
                    order, low_frequency_hz, high_frequency_hz));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(1.0, kTolerance),
                                        center_frequency_hz, kSampleRateHz));
        ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
            20.0f, center_frequency_hz, kSampleRateHz, kNumPoints));
        ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
            center_frequency_hz, kNyquistHz, kSampleRateHz, kNumPoints));
      }
    }
  }
}

TEST(PoleZeroFilterDesignTest, ButterworthBandstopCoefficientsTest) {
  constexpr double kTolerance = 1e-4;
  for (float low_frequency_hz : {1000.0f, 10000.0f}) {
    for (int order : {2, 3, 4}) {
      for (float bandwidth_hz : {200.0f, 400.0f}) {
        const float high_frequency_hz = low_frequency_hz + bandwidth_hz;
        const float center_frequency_hz =
            std::sqrt(low_frequency_hz * high_frequency_hz);
        BiquadFilterCascadeCoefficients coeffs =
            ButterworthFilterDesign(order).BandstopCoefficients(
                kSampleRateHz, low_frequency_hz, high_frequency_hz);
        SCOPED_TRACE(
            StringF("Butterworth bandstop (order %d) with range = [%f, %f].",
                    order, low_frequency_hz, high_frequency_hz));
        ASSERT_THAT(coeffs,
                    MagnitudeResponseIs(DoubleNear(0.0, kTolerance),
                                        center_frequency_hz, kSampleRateHz));
        ASSERT_THAT(coeffs, MagnitudeResponseDecreases(
            20.0f, center_frequency_hz, kSampleRateHz, kNumPoints));
        ASSERT_THAT(coeffs, MagnitudeResponseIncreases(
            center_frequency_hz, kNyquistHz, kSampleRateHz, kNumPoints));
      }
    }
  }
}

static constexpr float kRippleDb = 0.25;
static constexpr float kRippleToleranceLinear = 0.35;
static constexpr int kOrder = 4;

class ButterworthDefinedFilterDesign : public ButterworthFilterDesign  {
 public:
  ButterworthDefinedFilterDesign()
      : ButterworthFilterDesign(kOrder) {}
};

class ChebyshevType1DefinedFilterDesign : public ChebyshevType1FilterDesign  {
 public:
  ChebyshevType1DefinedFilterDesign()
      : ChebyshevType1FilterDesign(kOrder, kRippleDb) {}
};

class ChebyshevType2DefinedFilterDesign : public ChebyshevType2FilterDesign  {
 public:
  ChebyshevType2DefinedFilterDesign()
      : ChebyshevType2FilterDesign(kOrder, kRippleDb) {}
};

class EllipticDefinedFilterDesign : public EllipticFilterDesign  {
 public:
  EllipticDefinedFilterDesign()
      : EllipticFilterDesign(kOrder, kRippleDb, kRippleDb) {}
};

template <typename TypeParam>
class PoleZeroFilterDesignTypedTest : public ::testing::Test {};

typedef ::testing::Types<
    ButterworthDefinedFilterDesign
// TODO: Enable these once we've looked into the numerical problems.
//     ChebyshevType1DefinedFilterDesign,
//     ChebyshevType2DefinedFilterDesign,
//     EllipticDefinedFilterDesign
    > TestTypes;
TYPED_TEST_CASE(PoleZeroFilterDesignTypedTest, TestTypes);

constexpr float kTransitionFactor = 3;

TYPED_TEST(PoleZeroFilterDesignTypedTest, LowpassBasicTest) {
  constexpr double kTolerance = 1e-4;
  constexpr float kCornerFrequency = 1000.0f;
  BiquadFilterCascadeCoefficients coeffs =
      TypeParam().LowpassCoefficients(kSampleRateHz, kCornerFrequency);
  ASSERT_THAT(coeffs, MagnitudeResponseIs(
      DoubleNear(1.0, kRippleToleranceLinear), 0.0f, kSampleRateHz));
  ASSERT_THAT(coeffs, MagnitudeResponseIs(
      DoubleNear(1 / M_SQRT2, kTolerance), kCornerFrequency, kSampleRateHz));
  ASSERT_THAT(coeffs, MagnitudeResponseIs(
      DoubleNear(0.0, kRippleToleranceLinear), kNyquistHz, kSampleRateHz));
  // Test that response is less than kRippleToleranceLinear above the
  // transition and within kRippleToleranceLinear of 1 above.
  for (int f = 20; f < kCornerFrequency / kTransitionFactor; f *= 1.1) {
    ASSERT_THAT(coeffs, MagnitudeResponseIs(
        DoubleNear(1.0, kRippleToleranceLinear), f, kSampleRateHz));
  }
  for (int f = kCornerFrequency * kTransitionFactor; f < kNyquistHz; f *= 1.1) {
    ASSERT_THAT(coeffs, MagnitudeResponseIs(
        DoubleNear(0.0, kRippleToleranceLinear), f, kSampleRateHz));
  }
}

TYPED_TEST(PoleZeroFilterDesignTypedTest, HighpassBasicTest) {
  constexpr double kTolerance = 1e-4;
  constexpr float kCornerFrequency = 1000.0f;
  BiquadFilterCascadeCoefficients coeffs =
      TypeParam().HighpassCoefficients(kSampleRateHz, kCornerFrequency);
  ASSERT_THAT(coeffs, MagnitudeResponseIs(
      DoubleNear(0.0, kRippleToleranceLinear), 0.0f, kSampleRateHz));
  ASSERT_THAT(coeffs, MagnitudeResponseIs(
      DoubleNear(1 / M_SQRT2, kTolerance), kCornerFrequency, kSampleRateHz));
  ASSERT_THAT(coeffs, MagnitudeResponseIs(
      DoubleNear(1.0, kRippleToleranceLinear), kNyquistHz, kSampleRateHz));
  // Test that response is less than kRippleToleranceLinear below the
  // transition and within kRippleToleranceLinear of 1 above.
  for (int f = 20; f < kCornerFrequency / kTransitionFactor; f *= 1.1) {
    ASSERT_THAT(coeffs, MagnitudeResponseIs(
        DoubleNear(0.0, kRippleToleranceLinear), f, kSampleRateHz));
  }
  for (int f = kCornerFrequency * kTransitionFactor; f < kNyquistHz; f *= 1.1) {
    ASSERT_THAT(coeffs, MagnitudeResponseIs(
        DoubleNear(1.0, kRippleToleranceLinear), f, kSampleRateHz));
  }
}

TYPED_TEST(PoleZeroFilterDesignTypedTest, BandpassBasicTest) {
  constexpr float kLowFrequency = 1000.0f;
  constexpr float kHighFrequency = 3000.0f;

  const float center_frequency_hz =
      std::sqrt(kLowFrequency * kHighFrequency);
  BiquadFilterCascadeCoefficients coeffs = TypeParam().BandpassCoefficients(
      kSampleRateHz, kLowFrequency, kHighFrequency);
  ASSERT_THAT(coeffs,
              MagnitudeResponseIs(DoubleNear(1.0, kRippleToleranceLinear),
                                  center_frequency_hz, kSampleRateHz));
  for (int f = 20; f < kLowFrequency / kTransitionFactor; f *= 1.1) {
    ASSERT_THAT(coeffs, MagnitudeResponseIs(
        DoubleNear(0.0, kRippleToleranceLinear), f, kSampleRateHz));
  }
  for (int f = kHighFrequency * kTransitionFactor; f < kNyquistHz; f *= 1.1) {
    ASSERT_THAT(coeffs, MagnitudeResponseIs(
        DoubleNear(0.0, kRippleToleranceLinear), f, kSampleRateHz));
  }
}

TYPED_TEST(PoleZeroFilterDesignTypedTest, BandstopBasicTest) {
  constexpr float kLowFrequency = 1000.0f;
  constexpr float kHighFrequency = 3000.0f;
  const float center_frequency_hz =
      std::sqrt(kLowFrequency * kHighFrequency);
  BiquadFilterCascadeCoefficients coeffs = TypeParam().BandstopCoefficients(
      kSampleRateHz, kLowFrequency, kHighFrequency);
  ASSERT_THAT(coeffs,
              MagnitudeResponseIs(DoubleNear(0.0, kRippleToleranceLinear),
                                  center_frequency_hz, kSampleRateHz));

  for (int f = 20; f < kLowFrequency / kTransitionFactor; f *= 1.1) {
    ASSERT_THAT(coeffs, MagnitudeResponseIs(
        DoubleNear(1.0, kRippleToleranceLinear), f, kSampleRateHz));
  }
  for (int f = kHighFrequency * kTransitionFactor; f < kNyquistHz; f *= 1.1) {
    ASSERT_THAT(coeffs, MagnitudeResponseIs(
        DoubleNear(1.0, kRippleToleranceLinear), f, kSampleRateHz));
  }
}

}  // namespace
}  // namespace linear_filters
