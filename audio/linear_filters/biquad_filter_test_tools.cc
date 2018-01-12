#include "audio/linear_filters/biquad_filter_test_tools.h"

#include <cmath>

#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"

namespace linear_filters {

using ::std::complex;
using ::util::format::StringF;

namespace internal {

namespace {
double GetMagnitudeResponseSingleStage(
    const BiquadFilterCoefficients& coeffs,
    float frequency_hz, float sample_rate_hz) {
  complex<double> z = std::polar(1.0, 2 * M_PI * frequency_hz / sample_rate_hz);
  return std::abs(coeffs.EvalTransferFunction(z));
}
}  // namespace

// Pretty print a set of coefficients.
string AsString(const BiquadFilterCoefficients& coeffs) {
  return absl::StrCat(
      "B = [", absl::StrJoin(coeffs.b, ", "),
      "] and ", "A = [",
      absl::StrJoin(coeffs.a, ", "), "]");
}

// Pretty print the coefficients for a multi-stage filter.
string AsString(const BiquadFilterCascadeCoefficients& coeffs) {
  string output;
  for (int i = 0; i < coeffs.size(); ++i) {
    absl::SubstituteAndAppend(&output, "Stage $0: $1 ", i, AsString(coeffs[i]));
  }
  return output;
}

// static
bool BiquadFilterTestTools::IsMonotonicOnFrequencyRange(
    const std::function<double(double, double)>& response,
    double low_frequency_hz, double high_frequency_hz, double sample_rate_hz,
    int num_points, bool expect_increasing) {
  CHECK_GT(low_frequency_hz, 0);
  CHECK_GT(num_points, 0);
  const double step_factor = pow(high_frequency_hz / low_frequency_hz,
                                1.0 / num_points);
  double current_freq = low_frequency_hz;
  double current_response = response(current_freq, sample_rate_hz);
  current_freq *= step_factor;
  while (current_freq < high_frequency_hz) {
    double next_response = response(current_freq, sample_rate_hz);
    if (!std::isfinite(next_response)) {
      LOG(INFO) << "Nonfinite response at " << current_freq;
      return false;
    }
    // We use a very small tolerance when checking for monotonicity so that
    // functions that are roughly constant over part of the frequency range
    // do not fail due to numerical imprecision.
    if (expect_increasing) {
      if (next_response + 1e-5 < current_response) {
        LOG(INFO) <<
            StringF("Failed check for monotonic increase at frequency %f. "
                    "(%f < %f)", current_freq, next_response, current_response);
        return false;
      }
    } else {
      if (next_response - 1e-5 > current_response) {
        LOG(INFO) <<
            StringF("Failed check for monotonic decrease at frequency %f.  "
                    "(%f > %f)", current_freq, next_response, current_response);
       return false;
      }
    }
    current_response = next_response;
    current_freq *= step_factor;
  }
  return true;
}

}  // namespace internal
}  // namespace linear_filters
