/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ExecutionContext.h"

namespace {
/// @brief Thread-local storage for the current execution context.
thread_local cudaq::ExecutionContext *currentExecutionContext = nullptr;
thread_local bool persist_JITEngine = false;
thread_local std::optional<cudaq::JitEngine> jitEng = std::nullopt;
} // namespace

namespace nvqir {
bool isUsingResourceCounterSimulator();
} // namespace nvqir

namespace cudaq {

ExecutionContext *getExecutionContext() { return currentExecutionContext; }

bool isInTracerMode() {
  return currentExecutionContext && currentExecutionContext->name == "tracer";
}

bool isInBatchMode() {
  return currentExecutionContext &&
         currentExecutionContext->totalIterations != 0;
}

bool isLastBatch() {
  return currentExecutionContext &&
         currentExecutionContext->totalIterations > 0 &&
         currentExecutionContext->batchIteration ==
             currentExecutionContext->totalIterations - 1;
}

std::size_t getCurrentQpuId() {
  return currentExecutionContext ? currentExecutionContext->qpuId : 0;
}

void detail::setExecutionContext(ExecutionContext *ctx) {
  currentExecutionContext = ctx;

  if (currentExecutionContext && persist_JITEngine && jitEng.has_value())
    currentExecutionContext->jitEng = jitEng.value();
}

void detail::resetExecutionContext() {
  if (currentExecutionContext && persist_JITEngine &&
      currentExecutionContext->jitEng.has_value())
    jitEng = currentExecutionContext->jitEng.value();

  currentExecutionContext = nullptr;
}

/// This will cause the JITEngine stored in the current execution context to be
/// used for future launches until disabled by `disablePersistentJITEngine`
void detail::enablePersistentJITEngine() { persist_JITEngine = true; }

void detail::disablePersistentJITEngine() {
  persist_JITEngine = false;
  jitEng.reset();
}

} // namespace cudaq
