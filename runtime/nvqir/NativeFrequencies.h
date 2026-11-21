/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include <vector>

namespace nvqir {
/// Optional prototype extension. Only the qpp state-vector backend opts in;
/// the common CircuitSimulator interface and its existing ABI are unchanged.
class __attribute__((visibility("default"))) NativeFrequencies {
public:
  virtual ~NativeFrequencies() = default;
  virtual bool supportsNativeFrequencies() const = 0;
  virtual cudaq::ExecutionResult
  frequencies(const std::vector<std::size_t> &qubits, int shots) = 0;
};
} // namespace nvqir
