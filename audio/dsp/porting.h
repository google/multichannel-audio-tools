#ifndef AUDIO_DSP_OPEN_SOURCE_PORTING_H_
#define AUDIO_DSP_OPEN_SOURCE_PORTING_H_

#include <iostream>
#include <cmath>
#include <string>

using std::string;

typedef uint8_t uint8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  TypeName& operator=(const TypeName&) = delete

// Replacing util/math/mathutil.h.
template<typename T> struct MathLimits {
  // Not a number, i.e. result of 0/0.
  // Present only if !kIsInteger.
  static const T kNaN;
  // Positive infinity, i.e. result of 1/0.
  // Present only if !kIsInteger.
  static const T kPosInf;
  // Negative infinity, i.e. result of -1/0.
  // Present only if !kIsInteger.
  static const T kNegInf;
};


template <> struct MathLimits<float> {
  static constexpr float kNaN = HUGE_VALF - HUGE_VALF;
  static constexpr float kPosInf = HUGE_VALF;
  static constexpr float kNegInf = -HUGE_VALF;
};

class MathUtil {
 public:
  enum QuadraticRootType {kNoRealRoots = 0, kAmbiguous = 1, kTwoRealRoots = 2};

  static QuadraticRootType RealRootsForQuadratic(long double a,
                                                 long double b,
                                                 long double c,
                                                 long double *r1,
                                                 long double *r2);
};

// Replacing util/symbolize/demangle.h.
// TODO: glog actually includes a demangler function, but it isn't obvious how
//       to access it with our BUILD setup.
namespace util {
// Note that these don't do any demangling in the open source code.
bool Demangle(const char *mangled, char *out, int out_size);
string Demangle(const char* mangled);

namespace format {
string StringF(const char *fmt, ...);
}  // namespace format

}  // namespace util

#endif  // AUDIO_DSP_OPEN_SOURCE_PORTING_H_
