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
class SavedCompilerArtifact {
public:
  const std::optional<cudaq::JitEngine> &getJitEngine() const { return jitEng; }
  void setJitEngine(const cudaq::JitEngine engine) { jitEng = engine; }

  void checkArtifactReuse(const std::string &kernelName,
                          const std::vector<void *> &args,
                          const cudaq::JitEngine &engine,
                          std::function<void *()> argsCreatorThunk) {
    if (!jitEng.has_value()) {
      jitEng = engine;
      this->argsCreator = reinterpret_cast<int64_t (*)(const void *, void **)>(
          argsCreatorThunk());
      this->kernelName = kernelName;
      auto [resSize, scopedArgBuffer] = processArgs(args);
      this->argSize = resSize;
      this->argBuff = std::move(scopedArgBuffer);
      return;
    }

    if (kernelName != this->kernelName)
      throw std::runtime_error("Detected reuse of compiler artifact with "
                               "a different kernel.");

    auto [resSize, scopedArgBuffer] = processArgs(args);

    auto validate = [this, resSize, &scopedArgBuffer]() {
      if (resSize != this->argSize)
        return false;
      return memcmp(this->argBuff.get(), scopedArgBuffer.get(), resSize) == 0;
    };

    if (!validate())
      throw std::runtime_error("Detected reuse of compiler artifact with "
                               "diverging explicit arguments.");
  }

  void reset() {
    jitEng.reset();
    argsCreator = nullptr;
    argBuff.reset();
    argSize = 0;
  }

  SavedCompilerArtifact() : argBuff(nullptr, free) {}

private:
  std::optional<cudaq::JitEngine> jitEng = std::nullopt;
  // This is actually going to be a pointer into the jitEng,
  // but we have to store it explicitly due to linking issues.
  int64_t (*argsCreator)(const void *, void **);
  std::string kernelName;
  std::unique_ptr<void, decltype(&free)> argBuff;
  size_t argSize = 0;

  std::tuple<size_t, std::unique_ptr<void, decltype(&free)>>
  processArgs(const std::vector<void *> &args) {
    assert(jitEng.has_value());
    void *resBuffer;
    auto resSize = argsCreator(args.data(), &resBuffer);
    std::unique_ptr<void, decltype(&free)> scopedArgBuffer(resBuffer, free);
    return std::tuple(resSize, std::move(scopedArgBuffer));
  }
};

thread_local bool reuseArtifact = false;
thread_local SavedCompilerArtifact savedArtifact;

/// This will cause the JITEngine stored in the current execution context to be
/// used for future launches until disabled by `disablePersistentJITEngine`
void enablePersistentJITEngine() { reuseArtifact = true; }

void disablePersistentJITEngine() {
  reuseArtifact = false;
  savedArtifact.reset();
}

bool isPersistingJITEngine() { return reuseArtifact; }

void checkArtifactReuse(const std::string kernelName,
                        const std::vector<void *> &args, const JitEngine jit,
                        std::function<void *()> argsCreatorThunk) {
  if (!reuseArtifact)
    return;

  savedArtifact.checkArtifactReuse(kernelName, args, jit, argsCreatorThunk);
}

void saveEngineForReuse(ExecutionContext *ctx) {
  if (reuseArtifact && ctx && ctx->jitEng.has_value())
    savedArtifact.setJitEngine(ctx->jitEng.value());
}

void reuseEngineIfPresent(ExecutionContext *ctx) {
  if (!reuseArtifact || !ctx)
    return;
  auto engine = savedArtifact.getJitEngine();
  if (engine.has_value())
    ctx->jitEng = engine.value();
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

  compiler_artifact::reuseEngineIfPresent(ctx);
}

void detail::resetExecutionContext() {
  compiler_artifact::saveEngineForReuse(currentExecutionContext);

  currentExecutionContext = nullptr;
}
} // namespace cudaq
