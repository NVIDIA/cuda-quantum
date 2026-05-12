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
thread_local bool reuseArtifact = false;

class SavedCompilerArtifact {
public:
  void saveArtifact(const std::string &kernelName, const JitEngine engine) {
    if (jitEng.has_value()) {
      throw std::runtime_error(
          "Attempted to overwrite saved compiler artifact.");
    }
    jitEng = engine;
    this->kernelName = kernelName;
  }

  void checkArtifactReuse(const std::string &kernelName,
                          const JitEngine engine) {
    if (!jitEng.has_value()) {
      saveArtifact(kernelName, engine);
      return;
    }

    if (kernelName != this->kernelName)
      throw std::runtime_error("Detected reuse of compiler artifact with "
                               "a different kernel.");
  }

  void reset() { jitEng.reset(); }

  std::optional<JitEngine> getArtifactJit(const std::string &kernelName) {
    if (!jitEng.has_value())
      return std::nullopt;
    if (kernelName != this->kernelName)
      throw std::runtime_error("Detected reuse of compiler artifact with "
                               "a different kernel.");
    return jitEng;
  }

  SavedCompilerArtifact() {}

  void saveEngineForReuse(ExecutionContext *ctx) {
    if (!reuseArtifact || !ctx)
      return;
    jitEng = ctx->jitEng;
    launchMode = ctx->name;
  }

  void reuseEngineIfPresent(ExecutionContext *ctx) {
    if (!reuseArtifact || !ctx || !jitEng.has_value())
      return;

    // Allow launchMode == "" when the artifact was saved before any execution
    // context was set (e.g., via precompile_module). In that case, accept any
    // context and record the mode for future checks.
    if (!launchMode.empty() && launchMode != ctx->name)
      throw std::runtime_error(
          "Detected reuse of compiler artifact with different launch mode");
    launchMode = ctx->name;
    ctx->jitEng = jitEng.value();
  }

private:
  std::optional<JitEngine> jitEng = std::nullopt;
  // This is actually going to be a pointer into the jitEng,
  // but we have to store it explicitly due to linking issues.
  std::string kernelName;
  std::string launchMode;
};

thread_local SavedCompilerArtifact savedArtifact;

/// This will cause the JITEngine stored in the current execution context to be
/// used for future launches until disabled by `disablePersistentJITEngine`
void enablePersistentJITEngine() { reuseArtifact = true; }

void disablePersistentJITEngine() {
  reuseArtifact = false;
  savedArtifact.reset();
}

bool isPersistingJITEngine() { return reuseArtifact; }

void checkArtifactReuse(const std::string kernelName, const JitEngine jit) {
  if (!reuseArtifact)
    return;

  savedArtifact.checkArtifactReuse(kernelName, jit);
}

void saveArtifact(const std::string kernelName, const JitEngine jit) {
  if (!reuseArtifact)
    return;

  savedArtifact.saveArtifact(kernelName, jit);
}

std::optional<JitEngine> getArtifactJit(const std::string &kernelName) {
  if (!reuseArtifact)
    return std::nullopt;
  return savedArtifact.getArtifactJit(kernelName);
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
  compiler_artifact::savedArtifact.reuseEngineIfPresent(ctx);
  currentExecutionContext = ctx;
}

void detail::resetExecutionContext() {
  compiler_artifact::savedArtifact.saveEngineForReuse(currentExecutionContext);
  currentExecutionContext = nullptr;
}
} // namespace cudaq
