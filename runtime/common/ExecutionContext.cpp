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
#include <string>

namespace nvqir {
bool isUsingResourceCounterSimulator();
} // namespace nvqir

namespace {
/// @brief Thread-local storage for the current execution context.
thread_local cudaq::ExecutionContext *currentExecutionContext = nullptr;
} // namespace

namespace cudaq {

namespace compiler_artifact {
struct SavedCompilerArtifact {
public:
  bool hasJitEngine() const { return jitEng.has_value(); }
  const cudaq::JitEngine &getJitEngine() const { return jitEng.value(); }
  void setJitEngine(const cudaq::JitEngine &engine) { jitEng = engine; }

  void reset() {
    jitEng.reset();
    clearArgs();
  }

  // We're responsible for freeing argMessageBuffer.
  void saveInfo(std::string_view kernelName, void *argMessageBuffer,
                size_t size) {
    clearArgs();
    assert(argMessageBuffer);
    argSize = size;
    this->kernelName = kernelName;
    argBuff = argMessageBuffer;
  }

  bool isKernelSame(std::string_view kernelName) const {
    return this->kernelName == kernelName;
  }

  bool cmpInfo(void *argMessageBuffer, size_t size) const {
    assert(argBuff && argMessageBuffer);
    if (size != argSize)
      return false;
    return memcmp(argMessageBuffer, argBuff, size) == 0;
  }

private:
  void clearArgs() {
    if (argBuff)
      free(argBuff);
    argSize = 0;
  }

  std::optional<cudaq::JitEngine> jitEng = std::nullopt;
  std::string kernelName;
  void *argBuff;
  size_t argSize = 0;
};

thread_local bool persistJITEngine = false;
thread_local SavedCompilerArtifact savedArtifact;

/// This will cause the JITEngine stored in the current execution context to be
/// used for future launches until disabled by `disablePersistentJITEngine`
void enablePersistentJITEngine() { persistJITEngine = true; }

void disablePersistentJITEngine() {
  persistJITEngine = false;
  savedArtifact.reset();
}

bool isPersistingJITEngine() { return persistJITEngine; }

// We're responsible for freeing argMessageBuffer
void saveArtifactInfo(std::string_view kernelName, void *argMessageBuffer,
                      size_t size) {
  if (!persistJITEngine)
    return;
  savedArtifact.saveInfo(kernelName, argMessageBuffer, size);
}

bool isKernelSame(std::string_view kernelName) {
  if (!persistJITEngine)
    return false;
  return savedArtifact.isKernelSame(kernelName);
}

bool isArtifactReusable(void *argMessageBuffer, size_t size) {
  if (!persistJITEngine)
    return false;
  return savedArtifact.cmpInfo(argMessageBuffer, size);
}
} // namespace compiler_artifact

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

  if (currentExecutionContext && compiler_artifact::persistJITEngine &&
      compiler_artifact::savedArtifact.hasJitEngine())
    currentExecutionContext->jitEng =
        compiler_artifact::savedArtifact.getJitEngine();
}

void detail::resetExecutionContext() {
  if (currentExecutionContext && compiler_artifact::persistJITEngine &&
      currentExecutionContext->jitEng.has_value())
    compiler_artifact::savedArtifact.setJitEngine(
        currentExecutionContext->jitEng.value());

  currentExecutionContext = nullptr;
}
} // namespace cudaq
