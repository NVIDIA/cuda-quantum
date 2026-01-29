/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PTSBEInterface.h"
#include "common/ExecutionContext.h"
#include "cudaq/platform.h"
#include <stdexcept>

namespace cudaq::ptsbe {

inline bool hasMidCircuitMeasurements(const ExecutionContext &ctx) {
  return !ctx.registerNames.empty();
}

inline void throwIfMidCircuitMeasurements(const ExecutionContext &ctx) {
  if (hasMidCircuitMeasurements(ctx)) {
    throw std::runtime_error(
        "PTSBE does not support mid-circuit measurements. "
        "Circuits with conditional logic based on measurement outcomes "
        "cannot be pre-trajectory sampled.");
  }
}

inline std::vector<std::size_t> extractMeasureQubits(const Trace &trace) {
  std::vector<std::size_t> qubits;
  auto numQudits = trace.getNumQudits();
  qubits.reserve(numQudits);
  for (std::size_t i = 0; i < numQudits; ++i) {
    qubits.push_back(i);
  }
  return qubits;
}

inline sample_result dispatchPTSBE(const PTSBatch &batch) {
  throw std::runtime_error(
      "dispatchPTSBE: Not implemented. "
      "Full implementation requires simulator access and trajectory generation.");
}

template <typename QuantumKernel, typename... Args>
sample_result sampleWithPTSBE(QuantumKernel &&kernel, std::size_t shots,
                               Args &&...args) {
  ExecutionContext trace_ctx("tracer");
  auto &platform = get_platform();
  platform.set_exec_ctx(&trace_ctx);
  kernel(std::forward<Args>(args)...);
  platform.reset_exec_ctx();

  throwIfMidCircuitMeasurements(trace_ctx);

  PTSBatch batch;
  batch.kernel_trace = std::move(trace_ctx.kernelTrace);
  batch.measure_qubits = extractMeasureQubits(batch.kernel_trace);

  return dispatchPTSBE(batch);
}

template <typename QuantumKernel, typename... Args>
PTSBatch capturePTSBatch(QuantumKernel &&kernel, Args &&...args) {
  ExecutionContext trace_ctx("tracer");
  auto &platform = get_platform();
  platform.set_exec_ctx(&trace_ctx);
  kernel(std::forward<Args>(args)...);
  platform.reset_exec_ctx();

  throwIfMidCircuitMeasurements(trace_ctx);

  PTSBatch batch;
  batch.kernel_trace = std::move(trace_ctx.kernelTrace);
  batch.measure_qubits = extractMeasureQubits(batch.kernel_trace);
  return batch;
}

} // namespace cudaq::ptsbe
