# Multichannel Audio Tools

(This is not an official Google product!)

Multichannel Audio Tools contains common signal processing building blocks,
vectorized for multichannel processing using [Eigen](eigen.tuxfamily.org/).

A non-exhaustive list of libraries in this repo:
- biquad filters
- ladder filters (with time-varying coefficients and enforced stability)
- filter design libraries
  - lowpass, highpass, etc.
  - 2 way crossover, N-way crossover
  - auditory cascade filterbank
  - parametric equalizer
  - perceptual loudness filters for implementing ITU standards
- a fast rational factor resampler (single channel only)
- dynamic range control
  - compression
  - limiter
  - noise gate
  - expanders
  - multiband dynamic range control
- envelope detectors
- gmock matchers for vector/Eigen types

Contact multichannel-audio-tools-maintainers@google with questions/issues.
