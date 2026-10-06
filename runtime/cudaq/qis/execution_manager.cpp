/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "execution_manager.h"
#include "common/AnalysisScope.h"
#include "common/ExecutionContext.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/resourcecounter/ResourceCounterScope.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/policy_cpos.h"
#include "cudaq/algorithms/policy_dispatch.h"

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

void ExecutionManager::configureExecutionContext(const sample_policy &policy) {
  nvqir::getCircuitSimulatorInternal()->configureExecutionContext(policy);
}

void ExecutionManager::configureExecutionContext(const observe_policy &policy) {
  if (auto *ctx = getExecutionContext()) {
    configureExecutionContext(*ctx);
  }
  nvqir::getCircuitSimulatorInternal()->configureExecutionContext(policy);
}

void ExecutionManager::configureExecutionContext(const run_policy &policy) {
  nvqir::getCircuitSimulatorInternal()->configureExecutionContext(policy);
}

void ExecutionManager::configureExecutionContext(
    const msm_size_policy &policy) {
  nvqir::getCircuitSimulatorInternal()->configureExecutionContext(policy);
}

void ExecutionManager::configureExecutionContext(const msm_policy &policy) {
  nvqir::getCircuitSimulatorInternal()->configureExecutionContext(policy);
}

void ExecutionManager::configureExecutionContext(const dem_policy &policy) {
  nvqir::getCircuitSimulatorInternal()->configureExecutionContext(policy);
}

void ExecutionManager::configureExecutionContext(
    const ptsbe::sample_policy &policy) {
  nvqir::getCircuitSimulatorInternal()->configureExecutionContext(policy);
}

void ExecutionManager::configureExecutionContext(
    const estimate_policy &policy) {
  assert(cudaq::detail::AnalysisScope::is_active());
}

estimate_result
ExecutionManager::finalizeExecutionContext(const estimate_policy &policy) {
  assert(cudaq::detail::AnalysisScope::is_active());
  return nvqir::resource_counter::get_counts();
}

void ExecutionManager::configureExecutionContext(const orca::sample_policy &) {
  throw std::runtime_error(
      "Orca sampling is not supported by this execution manager.");
}

void ExecutionManager::finalizeExecutionContext(ExecutionContext &ctx) {
  // The execution context is no longer a result channel: every result-bearing
  // policy returns its result by value from the typed launch path. This
  // adapter therefore only serves the policies that have no result to deliver
  // (tracer, extract-state, resource counting, ...). A named result-bearing
  // policy arriving here means the caller went through the deprecated
  // set_exec_ctx / with_execution_context route and would silently get no
  // result, so fail loudly instead.
  policies::withPolicy(ctx.name, [&](auto policy) {
    if constexpr (std::is_same_v<decltype(policy), other_policies>) {
      cudaq::finalize_execution_manager(*this, policy, ctx);
    } else {
      throw std::runtime_error(
          "Execution context '" + ctx.name +
          "' names a result-bearing policy, which can no longer be finalized "
          "through the execution context. Launch it with cudaq::launch or "
          "cudaq::detail::launch and use the returned result instead.");
    }
  });
}
