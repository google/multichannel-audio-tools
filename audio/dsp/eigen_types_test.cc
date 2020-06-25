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

#include "audio/dsp/eigen_types.h"

#include <vector>

#include "gtest/gtest.h"

#include "audio/dsp/porting.h"  // auto-added.


namespace audio_dsp {
namespace {

using ::Eigen::ArrayXf;
using ::Eigen::ArrayXXf;
using ::Eigen::Dynamic;
using ::Eigen::InnerStride;
using ::Eigen::MatrixXf;
using ::Eigen::Map;
using ::Eigen::RowMajor;
using ::Eigen::VectorXf;

using RowMajorMatrixXf = Eigen::Matrix<float, Dynamic, Dynamic, RowMajor>;

TEST(EigenTypesTest, IsContiguous1DEigenType) {
  // Return true on Array, Vector, and Mapped Array and Vector types.
  EXPECT_TRUE(IsContiguous1DEigenType<ArrayXf>::Value);
  EXPECT_TRUE(IsContiguous1DEigenType<VectorXf>::Value);
  EXPECT_TRUE(IsContiguous1DEigenType<Map<ArrayXf>>::Value);
  EXPECT_TRUE(IsContiguous1DEigenType<Map<const VectorXf>>::Value);
  // Return true on segments of 1D contiguous arrays.
  EXPECT_TRUE(IsContiguous1DEigenType<
              decltype(ArrayXf().segment(0, 9))>::Value);
  // Return true on 1D matrix slices along the major direction.
  EXPECT_TRUE(IsContiguous1DEigenType<
              decltype(MatrixXf().col(0))>::Value);
  EXPECT_TRUE(IsContiguous1DEigenType<
              decltype(RowMajorMatrixXf().row(0))>::Value);

  // Return false on non-Eigen types.
  EXPECT_FALSE(IsContiguous1DEigenType<std::vector<float>>::Value);
  // Return false on 2D Eigen types.
  EXPECT_FALSE(IsContiguous1DEigenType<ArrayXXf>::Value);
  EXPECT_FALSE(IsContiguous1DEigenType<MatrixXf>::Value);
  // Return false on expressions whose elements aren't stored.
  EXPECT_FALSE(IsContiguous1DEigenType<
               decltype(ArrayXf::Random(9))>::Value);
  // Return false if inner stride isn't 1.
  using MappedWithStride2 = Map<VectorXf, 0, InnerStride<2>>;
  EXPECT_FALSE(IsContiguous1DEigenType<MappedWithStride2>::Value);
  EXPECT_FALSE(IsContiguous1DEigenType<
               decltype(ArrayXf().reverse())>::Value);
  EXPECT_FALSE(IsContiguous1DEigenType<
               decltype(MatrixXf().row(0))>::Value);
  EXPECT_FALSE(IsContiguous1DEigenType<
               decltype(RowMajorMatrixXf().col(0))>::Value);
}

}  // namespace
}  // namespace audio_dsp
