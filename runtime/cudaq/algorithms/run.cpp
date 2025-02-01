/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/run.h"
#include "common/ExecutionContext.h"
#include "cudaq/simulators.h"
#include "nvqir/CircuitSimulator.h"

cudaq::details::RunResultSpan cudaq::details::runTheKernel(
    std::function<void()> &&kernel, quantum_platform &platform,
    const std::string &kernel_name, std::size_t shots) {
  // 1. Clear the outputLog.
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  auto *currentContext = circuitSimulator->getExecutionContext();
  currentContext->outputLog.clear();

  // 2. Launch the kernel on the QPU.
  // 3. Pass the outputLog to the decoder (target-specific?)
  // 4. Get the buffer and length of buffer (in bytes) from the decoder.
  // 5. Clear the outputLog (?)
  // 6. Pass the span back as a RunResultSpan.
  return {nullptr, 0};
}
