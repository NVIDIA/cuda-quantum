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

/// @brief This QPU implementation enqueues kernel execution tasks and sets the
/// CUDA GPU device that it represents. There is a GPUEmulatedQPU per available
/// GPU.
class GPUEmulatedQPU : public QPU {
public:
  GPUEmulatedQPU();
  GPUEmulatedQPU(std::size_t id);

  void enqueue(QuantumTask &task) override;

  KernelThunkResultType launchKernel(const cudaq::SourceModule &src,
                                     cudaq::KernelArgs args) override;

  void configureExecutionContext(ExecutionContext &context) const override;
  void beginExecution() override;
  void endExecution() override;
  void finalizeExecutionContext(ExecutionContext &context) const override;
};

} // namespace cudaq
