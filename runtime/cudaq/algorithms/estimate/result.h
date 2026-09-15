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
  estimate_result(Resources counts, cudaq_json annotations = {});

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
