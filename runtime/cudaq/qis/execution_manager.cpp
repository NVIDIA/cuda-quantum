/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_manager.h"
#include "common/ExecutionContext.h"
#include "common/PluginUtils.h"
#include "cudaq/algorithms/policy_cpos.h"
#include "cudaq/algorithms/policy_dispatch.h"
#include "nvqir/CircuitSimulator.h"

using namespace cudaq;

static ExecutionManager *execution_manager;

namespace nvqir {
CircuitSimulator *getCircuitSimulatorInternal();
}

void cudaq::setExecutionManagerInternal(ExecutionManager *em) {
  CUDAQ_INFO("external caller setting the execution manager.");
  execution_manager = em;
}

void cudaq::resetExecutionManagerInternal() {
  CUDAQ_INFO("external caller clearing the execution manager.");
  execution_manager = nullptr;
}

ExecutionManager *cudaq::getExecutionManagerInternal() {
  return execution_manager;
}

ExecutionManager *cudaq::detail::getExecutionManagerFromContext() {
  auto ctx = getExecutionContext();
  if (ctx)
    return ctx->executionManager;
  return nullptr;
}

void ExecutionManager::configureExecutionContext(ExecutionContext &ctx) {
  nvqir::getCircuitSimulatorInternal()->configureExecutionContext(ctx);
}

void ExecutionManager::finalizeExecutionContext(ExecutionContext &ctx) {
  policies::withPolicy(ctx.name, [&](auto policy) {
    policies::visitResult(
        [&]() { return cudaq::finalize_execution_manager(*this, policy, ctx); },
        [&](sample_result &&r) { ctx.result = std::move(r); },
        [&](policies::void_result &&r) {});
  });
}
