/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::config {

/// The result of synthesizing one `u3`.
struct U3SynthesisResult {
  /// Clifford+T word in chronological order (the first gate is applied first).
  /// Gate names are `h`, `s`, `sdg`, `t`, `tdg`, `x`, `y` and `z`.
  std::vector<std::string> gates;
  /// Set when synthesis failed.
  std::optional<std::string> error;
};

/// A user-provided synthesizer for `u3(theta, phi, lambda)`, up to global
/// phase, within operator-norm error `epsilon`.
struct U3Synthesizer {
  using Fn = std::function<U3SynthesisResult(double theta, double phi,
                                             double lambda, double epsilon)>;
  std::shared_ptr<const Fn> fn;

  bool operator==(const U3Synthesizer &other) const { return fn == other.fn; }
};

} // namespace cudaq::config
