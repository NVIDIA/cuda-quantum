/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform/qpu.h"

namespace cudaq {
namespace details {

/// \cond
// The DefaultQPU models a simulated QPU by specifically
// targeting the QIS ExecutionManager.
class DefaultQPU : public cudaq::QPU {
public:
  DefaultQPU() = default;
  virtual ~DefaultQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override;
  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t argsSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override;

  void
  configureExecutionContext(cudaq::ExecutionContext &context) const override;
  void beginExecution() override;

  void endExecution() override;

  void
  finalizeExecutionContext(cudaq::ExecutionContext &context) const override;
};
/// \endcond
} // namespace details
} // namespace cudaq
