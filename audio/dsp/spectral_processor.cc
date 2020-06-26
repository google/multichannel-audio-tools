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

#include "audio/dsp/spectral_processor.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "audio/dsp/number_util.h"
#include "absl/types/span.h"

namespace audio_dsp {

using ::absl::Span;
using ::Eigen::ArrayXf;
using ::Eigen::ArrayXXcf;
using ::Eigen::ArrayXXf;
using ::Eigen::Map;
using ::std::vector;

SpectralProcessor::SpectralProcessor(int num_in_channels, int num_out_channels,
                                     int chunk_length, Span<float> window,
                                     int block_length, int hop_size,
                                     Callback* block_processor)
    : num_in_channels_(num_in_channels),
      num_out_channels_(num_out_channels),
      hop_size_(hop_size),
      block_length_(block_length),
      chunk_length_(chunk_length),
      block_processor_(block_processor),
      position_in_output_(0),
      initial_delay_(block_length_ -
                     GreatestCommonDivisor(chunk_length, hop_size)),
      out_overflow_(num_out_channels_, (chunk_length_ + block_length_)),
      window_(block_length_),
      transformer_(block_length_, true),
      time_workspace_(ArrayXXf::Zero(
          block_length_, std::max(num_in_channels_, num_out_channels))),
      unprocessed_block_workspace_(
          ArrayXXf::Zero(num_in_channels, block_length_)),
      processed_block_workspace_(
          ArrayXXf::Zero(num_out_channels_, block_length_)),
      in_fft_workspace_(RowMajorArrayXXcf(num_in_channels_,
                                          transformer_.GetTransformedSize())),
      out_fft_workspace_(RowMajorArrayXXcf(num_out_channels_,
                                           transformer_.GetTransformedSize())) {
  CHECK_GT(num_in_channels_, 0);
  CHECK_GT(block_length_, 0);
  CHECK_GT(chunk_length_, 0);
  CHECK_EQ(window.size(), block_length_);
  CHECK(block_processor_);
  window_ = Map<const ArrayXf>(window.data(), block_length_);
  Reset();
}

void SpectralProcessor::Reset() {
  out_overflow_.setZero();
  in_circular_buffer_.Init(num_in_channels_ * (chunk_length_ + block_length_));

  position_in_output_ = 0;

  int initial_delay_samples = num_in_channels_ * initial_delay_;
  vector<float> delay(initial_delay_samples);
  in_circular_buffer_.Write(Span<float>(delay.data(), delay.size()));
}

void SpectralProcessor::ProcessChunk(
    const Map<const ArrayXXf>& input, Map<ArrayXXf> output) {
  DCHECK_EQ(input.rows(), num_in_channels());
  DCHECK_EQ(input.cols(), chunk_length());
  DCHECK_EQ(output.rows(), num_out_channels());
  DCHECK_EQ(output.cols(), chunk_length());
  output.setZero();
  MoveOverflowIntoOutput(output);

  in_circular_buffer_.Write(
      Span<const float>(input.data(), num_in_channels_ * chunk_length_));

  while (in_circular_buffer_.NumReadableEntries() >=
         num_in_channels_ * block_length_) {
    // Read and deinterleave (transpose) a block.
    in_circular_buffer_.Peek(Span<float>(unprocessed_block_workspace_.data(),
                                         unprocessed_block_workspace_.size()));
    in_circular_buffer_.Advance(num_in_channels_ * hop_size_);

    const Map<const ArrayXXf> processed_block =
        WindowAndProcessInFrequencyDomain(unprocessed_block_workspace_);

    MoveProcessedIntoOutputAndOverflow(processed_block, position_in_output_,
                                       output);
    position_in_output_ += hop_size_;
    // We have processed enough input to be able to generate some output.
    if (position_in_output_ >= chunk_length_) {
      break;
    }
  }
  // Adjust our counter to account for the samples produced.
  position_in_output_ -= chunk_length_;
}

const Map<const ArrayXXf>
SpectralProcessor::WindowAndProcessInFrequencyDomain(
    const ArrayXXf& input) {
  CHECK_EQ(input.rows(), num_in_channels());
  CHECK_EQ(input.cols(), block_length_);

  // Window and transform the input.
  Map<ArrayXXf> input_block(time_workspace_.data(), block_length_,
                            num_in_channels());
  input_block = input.transpose();
  input_block.colwise() *= window_;
  for (int i = 0; i < num_in_channels_; ++i) {
    transformer_.ForwardTransform(input_block.col(i).data(),
                                  in_fft_workspace_.row(i).data());
  }
  block_processor_->ProcessSTFTBlock(in_fft_workspace_,
                                     transformer_.GetTransformedSize(),
                                     &out_fft_workspace_);

  // Transform and window the output.
  Map<ArrayXXf> output_block(time_workspace_.data(), block_length_,
                             num_out_channels());
  for (int i = 0; i < num_out_channels_; ++i) {
    transformer_.InverseTransform(out_fft_workspace_.row(i).data(),
                                  output_block.col(i).data());
  }
  output_block.colwise() *= window_;
  // Interleave (transpose) and return the audio to the buffer.
  processed_block_workspace_ = output_block.transpose();
  return Map<const ArrayXXf>(processed_block_workspace_.data(),
                             num_out_channels(), block_length_);
}

void SpectralProcessor::MoveOverflowIntoOutput(Map<ArrayXXf> out_chunk) {
  // Copy overlap into output buffer and clear the rest.
  const int frames_to_copy = std::min<int>(out_overflow_.cols(), chunk_length_);
  out_chunk.leftCols(frames_to_copy) = out_overflow_.leftCols(frames_to_copy);
  // Shift remaining overlap left.
  const int shift = frames_to_copy;
  for (int i = 0; i < out_overflow_.cols() - shift; ++i) {
    out_overflow_.col(i) = out_overflow_.col(i + shift);
    out_overflow_.col(i + shift).setZero();
  }
}

void SpectralProcessor::MoveProcessedIntoOutputAndOverflow(
    Map<const ArrayXXf> processed, int start_frame, Map<ArrayXXf> out_chunk) {
  // Copy the processed samples into the output.
  int frames_to_copy = std::max<int>(
      0, std::min<int>(processed.cols(), chunk_length_ - start_frame));
  out_chunk.middleCols(start_frame, frames_to_copy) +=
      processed.leftCols(frames_to_copy);

  // When we run out of room, put samples into the overflow.
  int overflow_frames = processed.cols() - frames_to_copy;
  out_overflow_.leftCols(overflow_frames) +=
      processed.middleCols(frames_to_copy, overflow_frames);
  // Clear remaining overflow.
  out_overflow_.rightCols(out_overflow_.cols() - overflow_frames).setZero();
}

}  // namespace audio_dsp
