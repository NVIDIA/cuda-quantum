/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Resources.h"
#include "common/cudaq_json.h"
#include <memory>
#include <string>
#include <vector>

namespace cudaq {
namespace detail {
struct EstimateResultImpl;
} // namespace detail

/// @brief The `estimate_result` encapsulates all data returned from a kernel
/// invocation with the `estimate_policy`.
class estimate_result {
  // Keep data on the heap to keep the return type small.
  std::unique_ptr<detail::EstimateResultImpl> impl;

public:
  /// @brief A group of identical instructions to report in `Resources`.
  struct InstructionCount {
    std::string name;
    std::size_t nControls = 0;
    std::size_t count = 1;
  };

  estimate_result(Resources counts, cudaq_json annotations = {});

  /// @brief Build the `Resources` from a list of instruction counts. Each
  /// entry is applied on wires `0..nControls`, so all entries share wire 0:
  /// the reported depth (and T depth) is the serial sum over the entries. This
  /// is an upper bound for producers that have no per-qubit dependency
  /// information, such as logical estimates.
  estimate_result(const std::vector<InstructionCount> &instructions,
                  std::size_t numQubits, cudaq_json annotations = {});

  estimate_result();
  ~estimate_result();
  estimate_result(estimate_result &&) noexcept;
  estimate_result &operator=(estimate_result &&);
  estimate_result(const estimate_result &);
  estimate_result &operator=(const estimate_result &);

  // TODO: update methods and fields according to
  // https://github.com/NVIDIA/cuda-quantum/issues/5050
  const Resources &get_resources() const;

  /// Arbitrary metadata attached by the producer of this result.
  const cudaq_json &get_annotations() const;
  cudaq_json &get_annotations();
};

} // namespace cudaq
