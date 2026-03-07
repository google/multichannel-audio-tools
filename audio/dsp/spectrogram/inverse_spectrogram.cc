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

#include "audio/dsp/spectrogram/inverse_spectrogram.h"

#include <cmath>
#include <complex>
#include <vector>

#include "audio/dsp/number_util.h"
#include "glog/logging.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/fft2d/fft.h"

namespace audio_dsp {

std::vector<double> InverseSpectrogram::GenerateSynthesisWindow(
    const std::vector<double>& analysis_window, int step_length) {
  const int frame_length = analysis_window.size();
  // Perfect reconstruction is not possible if the stft skips samples.
  DCHECK_LE(step_length, frame_length);

  std::vector<double> result(analysis_window);
  // The result is just the original window, with a normalization so the sum of
  // each element with its overlapped elements would be one.
  for (int i = 0; i < step_length; ++i) {
    // Calculate the normalization factor:
    double denom = 0.0;
    for (int offset = i; offset < analysis_window.size();
         offset += step_length) {
      double window_element = analysis_window[offset];
      denom += window_element * window_element;
    }

    // Apply the normalization:
    DCHECK_NE(denom, 0.0);  // Can't reconstruct if the weight is zero.
    for (int offset = i; offset < analysis_window.size();
         offset += step_length) {
      result[offset] /= denom;
    }
  }
  return result;
}

bool InverseSpectrogram::InternalInitialize() {
  fft_input_output_.resize(fft_length_);
  int half_fft_length = fft_length_ / 2;
  fft_double_working_area_.assign(half_fft_length, 0.0);
  fft_integer_working_area_.assign(2 + static_cast<int>(sqrt(half_fft_length)),
                                   0);
  // Set flag element to ensure that the working areas are initialized
  // on the first call to cdft (redundant given the assign above, but leaving
  // it here as a reminder.
  fft_integer_working_area_[0] = 0;

  overlap_ = frame_size_ - step_length_;
  if (overlap_ < 0) {
    overlap_ = 0;
  }
  working_output_.assign(overlap_, 0.0);
  working_output_.reserve(frame_size_);

  at_least_one_frame_processed_ = false;
  initialized_ = true;

  return true;
}

bool InverseSpectrogram::Initialize(int fft_length, int step_length) {
  initialized_ = false;
  if (fft_length < 2) {
    LOG(ERROR) << "FFT length too short.";
    return false;
  }
  if (!IsPowerOfTwoOrZero(fft_length)) {
    LOG(ERROR) << "FFT length not a power of 2.";
    return false;
  }
  fft_length_ = fft_length;
  frame_size_ = fft_length;

  if (step_length < 1) {
    LOG(ERROR) << "Step length must be positive.";
    return false;
  }
  step_length_ = step_length;

  with_window_function_ = false;

  return InternalInitialize();
}

bool InverseSpectrogram::Initialize(const std::vector<double>& window,
                                    int step_length) {
  initialized_ = false;
  if (window.size() < 2) {
    LOG(ERROR) << "Window length too short.";
    return false;
  }
  frame_size_ = window.size();
  with_window_function_ = true;
  fft_length_ = NextPowerOfTwo(window.size());
  const int half_fft_length = fft_length_ / 2;
  // Copy window.
  window_function_ =
      Eigen::Map<const Eigen::ArrayXd>(window.data(), window.size()) /
      half_fft_length;

  if (step_length < 1) {
    LOG(ERROR) << "Step length must be positive.";
    return false;
  }
  step_length_ = step_length;

  return InternalInitialize();
}

template <typename Container>
struct ContainerTraits {};

// Traits for std::vector<vector<complex<RealScalar>>>.
template <typename RealScalar_>
struct ContainerTraits<std::vector<std::vector<std::complex<RealScalar_>>>> {
  using RealScalar = RealScalar_;

  static const auto& AsIterable(
      const std::vector<std::vector<std::complex<RealScalar>>>& input) {
    return input;
  }
};

// Traits for Eigen::MatrixXcf and Eigen::MatrixXcd.
template <typename RealScalar_>
struct ContainerTraits<
    Eigen::Matrix<std::complex<RealScalar_>, Eigen::Dynamic, Eigen::Dynamic>> {
  using RealScalar = RealScalar_;

  static auto AsIterable(
      const Eigen::Matrix<std::complex<RealScalar_>, Eigen::Dynamic,
                          Eigen::Dynamic>& input) -> decltype(input.colwise()) {
    return input.colwise();
  }
};

template <class InputContainer, class OutputSample>
bool InverseSpectrogram::Process(const InputContainer& input,
                                 std::vector<OutputSample>* output) {
  using InputSample = typename ContainerTraits<InputContainer>::RealScalar;

  if (!initialized_) {
    LOG(ERROR) << "Process() called before successful call to Initialize().";
    return false;
  }
  CHECK(output);
  output->clear();
  // Note that input_iterable.end() - input_iterable.begin() != input.size()
  // in general, e.g. for Eigen::MatrixXcf. Also, const auto& is safe here even
  // for Eigen::MatrixXcf because the lifetime of Eigen::MatrixXcf::colwise() is
  // extended by C++ lifetime extension.
  const auto& input_iterable =
      ContainerTraits<InputContainer>::AsIterable(input);
  const int output_size = (input_iterable.end() - input_iterable.begin()) *
                          (frame_size_ - overlap_);
  output->reserve(output_size);

  for (const auto& slice : input_iterable) {
    // Check that slice is the right size for the complex-to-real inverse FFT.
    DCHECK_EQ(slice.size(), fft_length_ / 2 + 1);

    fft_input_output_[0] = std::real(slice[0]);
    // For rdft, put Nyquist-frequency real value into zero-frequency imag slot.
    fft_input_output_[1] = std::real(slice[fft_length_ / 2]);
    // Copy the rest of the slice.
    fft_input_output_.tail(fft_length_ - 2) =
        Eigen::Map<const Eigen::Array<InputSample, Eigen::Dynamic, 1>>(
            reinterpret_cast<const InputSample*>(slice.data() + 1),
            fft_length_ - 2)
            // fft2d's rdft() expects doubles, so cast to double if needed.
            .template cast<double>();
    ProcessCoreFFT();
    int num_output_samples = working_output_.size() - overlap_;
    // Copy the done samples, converting to float if necessary.
    output->insert(output->end(), working_output_.begin(),
                   working_output_.begin() + num_output_samples);
    // And remove them from working_output_.
    working_output_.erase(working_output_.begin(),
                          working_output_.begin() + num_output_samples);
    at_least_one_frame_processed_ = true;
  }
  return true;
}
// Instantiate it five ways:
template bool InverseSpectrogram::Process(
    const std::vector<std::vector<std::complex<float>>>&, std::vector<float>*);
template bool InverseSpectrogram::Process(
    const std::vector<std::vector<std::complex<double>>>&, std::vector<float>*);
template bool InverseSpectrogram::Process(
    const std::vector<std::vector<std::complex<float>>>&, std::vector<double>*);
template bool InverseSpectrogram::Process(
    const std::vector<std::vector<std::complex<double>>>&,
    std::vector<double>*);

template bool InverseSpectrogram::Process(const Eigen::MatrixXcf&,
                                          std::vector<float>*);

void InverseSpectrogram::ProcessCoreFFT() {
  constexpr int kReverseFFT = -1;  // 1 means forward; -1 reverse.
  rdft(fft_length_, kReverseFFT, fft_input_output_.data(),
       fft_integer_working_area_.data(), fft_double_working_area_.data());

  if (with_window_function_) {
    // Apply the window, which includes the FFT normalization factor.
    fft_input_output_.head(frame_size_) *= window_function_;
  } else {
    // Otherwise just apply the FFT normalization.
    const int half_fft_length = fft_length_ / 2;
    fft_input_output_.head(frame_size_) /= half_fft_length;
  }

  // Add the overlapping portion.
  Eigen::Map<Eigen::ArrayXd>(working_output_.data(), working_output_.size()) +=
      fft_input_output_.head(overlap_);
  // Append the new portion.
  auto new_portion =
      fft_input_output_.segment(overlap_, frame_size_ - overlap_);
  working_output_.insert(working_output_.end(), new_portion.begin(),
                         new_portion.end());
}

template <class OutputSample>
bool InverseSpectrogram::Flush(std::vector<OutputSample>* output) {
  if (output) {
    if (at_least_one_frame_processed_) {
      output->assign(working_output_.begin(), working_output_.end());
    } else {
      output->clear();
    }
  }
  if (!initialized_) {
    LOG(ERROR) << "Flush() called before successful call to Initialize().";
    return false;
  }
  // Reset to original state so that Process() may be called again.
  Initialize(fft_length_, step_length_);
  return true;
}
template bool InverseSpectrogram::Flush(std::vector<double>*);
template bool InverseSpectrogram::Flush(std::vector<float>*);

}  // namespace audio_dsp

