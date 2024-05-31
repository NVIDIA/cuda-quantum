/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_manager.h"
#include "cudaq/platform.h"
#include <iostream>

bool cudaq::__nvqpp__MeasureResultBoolConversion(int result) {
  auto &platform = get_platform();
  auto *ctx = platform.get_exec_ctx();
  if (ctx && ctx->name == "tracer")
    ctx->registerNames.push_back("");
  return result == 1;
}

namespace cudaq {
__attribute__((weak)) ExecutionManager *getExecutionManager() {
  std::cerr << "Error: must link with an execution manager implementation.\n";
  return nullptr;
}
} // namespace cudaq

extern "C" {
/// C interface to the (default) execution manager's methods.
///
/// This supplies an interface to allocate and deallocate qubits, reset a
/// qubit, measure a qubit, and apply the gates defined by CUDA-Q.

std::int64_t __nvqpp__cudaq_em_allocate() {
  return cudaq::getExecutionManager()->allocateQudit();
}

void __nvqpp__cudaq_em_apply(const char *gateName, std::int64_t numParams,
                             const double *params,
                             const std::span<std::size_t> &ctrls,
                             const std::span<std::size_t> &targets,
                             bool isAdjoint) {
  std::vector<double> pv{params, params + numParams};
  auto fromSpan = [&](const std::span<std::size_t> &qubitSpan)
      -> std::vector<cudaq::QuditInfo> {
    std::vector<cudaq::QuditInfo> result;
    for (std::size_t qb : qubitSpan)
      result.emplace_back(2u, qb);
    return result;
  };
  std::vector<cudaq::QuditInfo> cv = fromSpan(ctrls);
  std::vector<cudaq::QuditInfo> tv = fromSpan(targets);
  cudaq::getExecutionManager()->apply(gateName, pv, cv, tv, isAdjoint);
}

std::int32_t __nvqpp__cudaq_em_measure(const std::span<std::size_t> &targets,
                                       const char *tagName) {
  cudaq::QuditInfo qubit{2u, targets[0]};
  std::string tag{tagName};
  return cudaq::getExecutionManager()->measure(qubit, tag);
}

void __nvqpp__cudaq_em_reset(const std::span<std::size_t> &targets) {
  cudaq::QuditInfo qubit{2u, targets[0]};
  cudaq::getExecutionManager()->reset(qubit);
}

void __nvqpp__cudaq_em_return(const std::span<std::size_t> &targets) {
  cudaq::QuditInfo qubit{2u, targets[0]};
  cudaq::getExecutionManager()->returnQudit(qubit);
}
} // extern "C"
