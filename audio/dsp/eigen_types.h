/*
 * Copyright 2020 Google LLC
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

// Utilities for working generically with Eigen types.

#ifndef AUDIO_DSP_EIGEN_TYPES_H_
#define AUDIO_DSP_EIGEN_TYPES_H_

#include <type_traits>

#include "third_party/eigen3/Eigen/Core"

#include "audio/dsp/porting.h"  // auto-added.


namespace audio_dsp {

// Traits class to test at compile time whether a type is a contiguous 1D Eigen
// type, such that the ith element is accessible through .data()[i].
//
// Examples:
//   IsContiguous1DEigenType<ArrayXf>::Value  // = true
//   IsContiguous1DEigenType<Map<VectorXf>>::Value  // = true
//   IsContiguous1DEigenType<std::vector<float>>::Value  // = false
template <typename Type, typename = void>
struct IsContiguous1DEigenType { enum { Value = false }; };

namespace internal {
// For use with SFINAE to enable only when T is an Eigen::DenseBase type.
template <typename T> void WellFormedIfDenseEigenType(Eigen::DenseBase<T>&&);
}  // namespace internal

template <typename EigenType>
struct IsContiguous1DEigenType<EigenType,
    decltype(internal::WellFormedIfDenseEigenType(std::declval<EigenType>()))> {
  enum {
    Value = EigenType::IsVectorAtCompileTime &&
        EigenType::InnerStrideAtCompileTime == 1
  };
};

}  // namespace audio_dsp

#endif  // AUDIO_DSP_EIGEN_TYPES_H_
