/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ExecutionContext.h"
#include <cstdlib>
#include <cstring>

namespace {
class SavedCompilerArtifact {
public:
  bool hasJitEngine() const { return jitEng.has_value(); }
  const std::optional<cudaq::JitEngine> &getJitEngine() const { return jitEng; }
  void setJitEngine(const cudaq::JitEngine &engine) { jitEng = engine; }

  void reset() {
    jitEng.reset();
    clearArgs();
  }

  // We're responsible for freeing argMessageBuffer.
  void saveLaunchInfo(void *argMessageBuffer, size_t size) {
    assert(argMessageBuffer);
    assert(!argBuff);
    argBuff = argMessageBuffer;
    argSize = size;
  }

  bool isLaunchInfoSame(void *argMessageBuffer, size_t size) const {
    if (!argBuff || !argMessageBuffer)
      return false;
    if (size != argSize)
      return false;
    return memcmp(argMessageBuffer, argBuff, size) == 0;
  }

private:
  void clearArgs() {
    if (argBuff)
      std::free(argBuff);
    argBuff = nullptr;
    argSize = 0;
  }

  std::optional<cudaq::JitEngine> jitEng = std::nullopt;
  void *argBuff = nullptr;
  size_t argSize = 0;
};

/// @brief Thread-local storage for the current execution context.
thread_local cudaq::ExecutionContext *currentExecutionContext = nullptr;
thread_local bool persistJITEngine = false;
thread_local SavedCompilerArtifact savedArtifact;
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

  if (currentExecutionContext && persistJITEngine &&
      savedArtifact.hasJitEngine())
    currentExecutionContext->jitEng = savedArtifact.getJitEngine().value();
}

void detail::resetExecutionContext() {
  if (currentExecutionContext && persistJITEngine &&
      currentExecutionContext->jitEng.has_value())
    savedArtifact.setJitEngine(currentExecutionContext->jitEng.value());

  currentExecutionContext = nullptr;
}

/// This will cause the JITEngine stored in the current execution context to be
/// used for future launches until disabled by `disablePersistentJITEngine`
void detail::enablePersistentJITEngine() { persistJITEngine = true; }

void detail::disablePersistentJITEngine() {
  persistJITEngine = false;
  savedArtifact.reset();
}

bool detail::isPersistingJITEngine() { return persistJITEngine; }

// We're responsible for freeing argMessageBuffer
void detail::saveLaunchInfo(void *argMessageBuffer, size_t size) {
  if (!persistJITEngine)
    return;
  savedArtifact.saveLaunchInfo(argMessageBuffer, size);
}

bool detail::isLaunchInfoSame(void *argMessageBuffer, size_t size) {
  if (!persistJITEngine)
    return false;
  return savedArtifact.isLaunchInfoSame(argMessageBuffer, size);
}

} // namespace cudaq
