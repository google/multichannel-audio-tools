#include "audio/dsp/porting.h"

#include <cfloat>
#include <cstdarg>

// static
MathUtil::QuadraticRootType MathUtil::RealRootsForQuadratic(
    long double a,
    long double b,
    long double c,
    long double *r1,
    long double *r2) {
  // Deal with degenerate cases where leading coefficients vanish.
  if (a == 0.0) {
    if (b == 0.0) {
      if (c == 0.0) {
        *r1 = *r2 = 0.0;
        return kAmbiguous;
      }
      return kNoRealRoots;
    }
    // The linear equation has a single root at x = -c / b, not a double
    // one.  Respond as if a==epsilon: The other root is at "infinity",
    // which we signal with HUGE_VAL so that the behavior stays consistent
    // as a->0.
    *r1 = -c / b;
    *r2 = HUGE_VAL;
    return kTwoRealRoots;
  }

  // General case: the quadratic formula, rearranged for greater numerical
  // stability.

  // If the discriminant is zero to numerical precision, regardless of
  // sign, treat it as zero and return kAmbiguous.
  const long double discriminant =  b * b - 4 * a * c;
  if (std::abs(discriminant) <=
      DBL_EPSILON * std::max(2 * b * b, fabsl(4 * a * c))) {
    *r2 = *r1 = -b / 2 / a;  // The quadratic is (2*a*x + b)^2 = 0.
    return kAmbiguous;
  }

  if (discriminant < 0) {
    // The discriminant is definitely negative so there are no real roots.
    return kNoRealRoots;
  }

  long double const q = -0.5 *
      (b + ((b >= 0) ? std::sqrt(discriminant) : -std::sqrt(discriminant)));
  *r1 = q / a;  // If a is very small this produces +/- HUGE_VAL.
  *r2 = c / q;  // q cannot be too small.
  return kTwoRealRoots;
}

namespace util {
bool Demangle(const char *mangled, char *out, int out_size) {
  // We're not even going to attempt to demangle. This is just to get the
  // open source code to build.
  *out = *mangled;
  return true;
}

string Demangle(const char *mangled) {
  return "";
}

namespace format {
string StringF(const char *fmt, ...) {
  string s;
  va_list argptr;
  va_start(argptr, fmt);
  constexpr int kMaxStringLength = 512;
  char temp[kMaxStringLength];
  vsnprintf(temp, kMaxStringLength, fmt, argptr);
  va_end(argptr);
  s = temp;
  return s;
}
}  // namespace format
}  // namespace util

