/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/quantum_platform.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
#include "common/RuntimeTarget.h"
#include "cudaq/platform/qpu.h"
#include <iostream>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>

LLVM_INSTANTIATE_REGISTRY(cudaq::QPU::RegistryType)

namespace cudaq {

// These functions are defined elsewhere, but
// we are going to use them here.
std::string get_quake(const std::string &);

static quantum_platform *platform;
static constexpr std::string_view GetQuantumPlatformSymbol =
    "getQuantumPlatform";

void setQuantumPlatformInternal(quantum_platform *p) {
  info("external caller setting the platform.");
  platform = p;
}

/// @brief Get the provided platform plugin
/// @return
quantum_platform *getQuantumPlatformInternal() {
  if (platform)
    return platform;
  platform =
      getUniquePluginInstance<quantum_platform>(GetQuantumPlatformSymbol);
  return platform;
}

void quantum_platform::set_noise(const noise_model *model, std::size_t qpu_id) {
  validateQpuId(qpu_id);
  auto &platformQPU = platformQPUs[qpu_id];
  platformQPU->setNoiseModel(model);
}

const noise_model *quantum_platform::get_noise(std::size_t qpu_id) {
  if (auto *ctx = executionContext.get())
    return ctx->noiseModel;

  validateQpuId(qpu_id);
  auto &platformQPU = platformQPUs[qpu_id];
  return platformQPU->getNoiseModel();
}

void quantum_platform::reset_noise(std::size_t qpu_id) {
  validateQpuId(qpu_id);
  set_noise(nullptr, qpu_id);
}

std::future<sample_result>
quantum_platform::enqueueAsyncTask(const std::size_t qpu_id,
                                   KernelExecutionTask &task) {
  std::promise<sample_result> promise;
  auto f = promise.get_future();
  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), t = task]() mutable {
        auto counts = t();
        p.set_value(counts);
      });

  platformQPUs[qpu_id]->enqueue(wrapped);
  return f;
}

void quantum_platform::enqueueAsyncTask(const std::size_t qpu_id,
                                        std::function<void()> &f) {
  platformQPUs[qpu_id]->enqueue(f);
}

void quantum_platform::validateQpuId(std::size_t qpuId) const {
  if (platformQPUs.empty())
    throw std::runtime_error("No QPUs are available for this target.");
  if (qpuId >= platformQPUs.size()) {
    throw std::invalid_argument(
        "Invalid QPU ID: " + std::to_string(qpuId) +
        ". Number of QPUs: " + std::to_string(platformQPUs.size()));
  }
}

std::size_t quantum_platform::get_current_qpu() const {
  if (auto *ctx = executionContext.get())
    return ctx->qpuId;
  return 0;
}

// Specify the execution context for this platform.
// This delegates to the targeted QPU
void quantum_platform::set_exec_ctx(ExecutionContext *ctx) {
  std::size_t qid = ctx->qpuId;
  validateQpuId(qid);

  executionContext.set(ctx);
  auto &platformQPU = platformQPUs[qid];
  try {
    platformQPU->setExecutionContext(ctx);
  } catch (...) {
    executionContext.set(nullptr);
    throw;
  }
}

/// Reset the execution context for this platform.
void quantum_platform::reset_exec_ctx() {
  auto ctx = executionContext.get();
  if (ctx == nullptr)
    return;

  std::size_t qid = ctx->qpuId;
  auto &platformQPU = platformQPUs[qid];

  try {
    platformQPU->resetExecutionContext();
  } catch (...) {
    executionContext.set(nullptr);
    throw;
  }
  executionContext.set(nullptr);
}

std::optional<QubitConnectivity> quantum_platform::connectivity() {
  return platformQPUs.front()->getConnectivity();
}

bool quantum_platform::is_simulator(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->isSimulator();
}

bool quantum_platform::is_remote(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->isRemote();
}

bool quantum_platform::is_emulated(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->isEmulated();
}

std::size_t quantum_platform::get_num_qubits(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->getNumQubits();
}

bool quantum_platform::supports_conditional_feedback(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->supportsConditionalFeedback();
}

bool quantum_platform::supports_explicit_measurements(
    std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->supportsExplicitMeasurements();
}

void quantum_platform::launchVQE(const std::string kernelName,
                                 const void *kernelArgs, gradient *gradient,
                                 const spin_op &H, optimizer &optimizer,
                                 const int n_params, const std::size_t shots,
                                 std::size_t qpu_id) {
  validateQpuId(qpu_id);
  auto &qpu = platformQPUs[qpu_id];
  qpu->launchVQE(kernelName, kernelArgs, gradient, H, optimizer, n_params,
                 shots);
}

RemoteCapabilities
quantum_platform::get_remote_capabilities(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->getRemoteCapabilities();
}

KernelThunkResultType quantum_platform::launchKernel(
    const std::string &kernelName, KernelThunkType kernelFunc, void *args,
    std::uint64_t voidStarSize, std::uint64_t resultOffset,
    const std::vector<void *> &rawArgs, std::size_t qpu_id) {
  validateQpuId(qpu_id);
  auto &qpu = platformQPUs[qpu_id];
  return qpu->launchKernel(kernelName, kernelFunc, args, voidStarSize,
                           resultOffset, rawArgs);
}

void quantum_platform::launchKernel(const std::string &kernelName,
                                    const std::vector<void *> &rawArgs,
                                    std::size_t qpu_id) {
  validateQpuId(qpu_id);
  auto &qpu = platformQPUs[qpu_id];
  qpu->launchKernel(kernelName, rawArgs);
}

KernelThunkResultType quantum_platform::launchModule(
    const std::string &kernelName, mlir::ModuleOp module,
    const std::vector<void *> &rawArgs, mlir::Type resTy, std::size_t qpu_id) {
  validateQpuId(qpu_id);
  auto &qpu = platformQPUs[qpu_id];
  return qpu->launchModule(kernelName, module, rawArgs, resTy);
}

void *quantum_platform::specializeModule(const std::string &kernelName,
                                         mlir::ModuleOp module,
                                         const std::vector<void *> &rawArgs,
                                         mlir::Type resTy, void *cachedEngine,
                                         std::size_t qpu_id) {
  validateQpuId(qpu_id);
  auto &qpu = platformQPUs[qpu_id];
  return qpu->specializeModule(kernelName, module, rawArgs, resTy,
                               cachedEngine);
}

void quantum_platform::onRandomSeedSet(std::size_t seed) {
  // Send on the notification to all QPUs.
  for (auto &qpu : platformQPUs)
    qpu->onRandomSeedSet(seed);
}

void quantum_platform::resetLogStream() { platformLogStream = nullptr; }

std::ostream *quantum_platform::getLogStream() { return platformLogStream; }

void quantum_platform::setLogStream(std::ostream &logStream) {
  platformLogStream = &logStream;
}

cudaq::CodeGenConfig quantum_platform::get_codegen_config() {
  if (runtimeTarget &&
      !runtimeTarget->config.getCodeGenSpec(runtimeTarget->runtimeConfig)
           .empty()) {
    auto config = cudaq::parseCodeGenTranslation(
        runtimeTarget->config.getCodeGenSpec(runtimeTarget->runtimeConfig));
    return config;
  }

  // The target config doesn't specify a codegen setting
  CodeGenConfig config = {.profile = "qir-adaptive",
                          .isQIRProfile = true,
                          .version = QirVersion::version_1_0,
                          .qir_major_version = 1,
                          .qir_minor_version = 0,
                          .isAdaptiveProfile = true,
                          .isBaseProfile = false,
                          .integerComputations = true,
                          .floatComputations = true,
                          .outputLog = !is_remote(),
                          .eraseStackBounding = false,
                          .eraseRecordCalls = false,
                          .allowAllInstructions = true};

  return config;
}

const RuntimeTarget *quantum_platform::get_runtime_target() const {
  return runtimeTarget.get();
}
} // namespace cudaq

cudaq::KernelThunkResultType
cudaq::altLaunchKernel(const char *kernelName,
                       cudaq::KernelThunkType kernelFunc, void *kernelArgs,
                       std::uint64_t argsSize, std::uint64_t resultOffset) {
  ScopedTraceWithContext("altLaunchKernel", kernelName, argsSize);
  auto &platform = *getQuantumPlatformInternal();
  std::string kernName = kernelName;
  std::size_t qpu_id = platform.get_current_qpu();
  return platform.launchKernel(kernName, kernelFunc, kernelArgs, argsSize,
                               resultOffset, {}, qpu_id);
}

cudaq::KernelThunkResultType
cudaq::streamlinedLaunchKernel(const char *kernelName,
                               const std::vector<void *> &rawArgs) {
  std::size_t argsSize = rawArgs.size();
  ScopedTraceWithContext("streamlinedLaunchKernel", kernelName, argsSize);
  auto &platform = *getQuantumPlatformInternal();
  std::string kernName = kernelName;
  std::size_t qpu_id = platform.get_current_qpu();
  platform.launchKernel(kernName, rawArgs, qpu_id);
  // NB: The streamlined launch will never return results. Use alt or hybrid if
  // the kernel returns results.
  return {};
}

// FIXME: make this an inline function in nvqpp_interface.h. Requires ModuleOp
// definition be available in that .h file though.
cudaq::KernelThunkResultType
cudaq::streamlinedLaunchModule(const char *kernelName, mlir::ModuleOp moduleOp,
                               const std::vector<void *> &rawArgs,
                               mlir::Type resTy) {
  std::string name = kernelName;
  return streamlinedLaunchModule(name, moduleOp, rawArgs, resTy);
}

cudaq::KernelThunkResultType cudaq::streamlinedLaunchModule(
    const std::string &kernelName, mlir::ModuleOp moduleOp,
    const std::vector<void *> &rawArgs, mlir::Type resTy) {
  ScopedTraceWithContext("streamlinedLaunchModule", kernelName, rawArgs.size());

  auto &platform = *getQuantumPlatformInternal();
  std::size_t qpu_id = platform.get_current_qpu();
  return platform.launchModule(kernelName, moduleOp, rawArgs, resTy, qpu_id);
}

void *cudaq::streamlinedSpecializeModule(const std::string &kernelName,
                                         mlir::ModuleOp moduleOp,
                                         const std::vector<void *> &rawArgs,
                                         mlir::Type resTy, void *cachedEngine) {
  ScopedTraceWithContext("streamlinedSpecializeModule", kernelName,
                         rawArgs.size());

  auto &platform = *getQuantumPlatformInternal();
  auto qpu_id = platform.get_current_qpu();
  return platform.specializeModule(kernelName, moduleOp, rawArgs, resTy,
                                   cachedEngine, qpu_id);
}

cudaq::KernelThunkResultType
cudaq::hybridLaunchKernel(const char *kernelName, cudaq::KernelThunkType kernel,
                          void *args, std::uint64_t argsSize,
                          std::uint64_t resultOffset,
                          const std::vector<void *> &rawArgs) {
  ScopedTraceWithContext("hybridLaunchKernel", kernelName);
  auto &platform = *getQuantumPlatformInternal();
  const std::string kernName = kernelName;
  std::size_t qpu_id = platform.get_current_qpu();
  if (platform.is_remote()) {
    // This path should never call a kernel that returns results.
    platform.launchKernel(kernName, rawArgs, qpu_id);
    return {};
  }
  return platform.launchKernel(kernName, kernel, args, argsSize, resultOffset,
                               rawArgs, qpu_id);
}

namespace cudaq {

// Per-thread execution context storage implementation. Temporary - will be
// removed when executionContext is eliminated.
struct detail::PerThreadExecCtx::Impl {
  mutable std::shared_mutex mutex;
  std::unordered_map<std::size_t, ExecutionContext *> contexts;
};

detail::PerThreadExecCtx::PerThreadExecCtx() : impl(std::make_unique<Impl>()) {}

detail::PerThreadExecCtx::~PerThreadExecCtx() = default;

ExecutionContext *detail::PerThreadExecCtx::get() const {
  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  std::shared_lock<std::shared_mutex> lock(impl->mutex);
  auto it = impl->contexts.find(tid);
  return it != impl->contexts.end() ? it->second : nullptr;
}

void detail::PerThreadExecCtx::set(ExecutionContext *ctx) {
  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  std::unique_lock<std::shared_mutex> lock(impl->mutex);
  if (ctx)
    impl->contexts[tid] = ctx;
  else
    impl->contexts.erase(tid);
}

} // namespace cudaq
