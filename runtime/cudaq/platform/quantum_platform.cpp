/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/quantum_platform.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
#include "common/RuntimeTarget.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/qis/qudit.h"
#include "nvqpp_config.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>

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

void quantum_platform::set_noise(const noise_model *model) {
  auto &platformQPU = platformQPUs[platformCurrentQPU];
  platformQPU->setNoiseModel(model);
}

const noise_model *quantum_platform::get_noise() {
  if (executionContext)
    return executionContext->noiseModel;

  auto &platformQPU = platformQPUs[platformCurrentQPU];
  return platformQPU->getNoiseModel();
}

void quantum_platform::reset_noise() { set_noise(nullptr); }

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
  set_current_qpu(qpu_id);
  platformQPUs[qpu_id]->enqueue(f);
}

void quantum_platform::validateQpuId(int qpuId) const {
  if (platformQPUs.empty())
    throw std::runtime_error("No QPUs are available for this target.");
  if (qpuId < 0 || qpuId >= platformNumQPUs) {
    throw std::invalid_argument(
        "Invalid QPU ID: " + std::to_string(qpuId) +
        ". Number of QPUs: " + std::to_string(platformNumQPUs));
  }
}

void quantum_platform::set_current_qpu(const std::size_t device_id) {
  if (device_id >= platformNumQPUs) {
    throw std::invalid_argument(
        "QPU device id " + std::to_string(device_id) +
        " is not valid (greater than number of available QPUs: " +
        std::to_string(platformNumQPUs) + ").");
  }
  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  {
    std::unique_lock lock(threadToQpuIdMutex);
    platformCurrentQPU = device_id;
    auto iter = threadToQpuId.find(tid);
    if (iter != threadToQpuId.end())
      iter->second = device_id;
    else
      threadToQpuId.emplace(tid, device_id);
  }
}

std::size_t quantum_platform::get_current_qpu() { return platformCurrentQPU; }

// Specify the execution context for this platform.
// This delegates to the targeted QPU
void quantum_platform::set_exec_ctx(ExecutionContext *ctx, std::size_t qid) {
  executionContext = ctx;
  validateQpuId(qid);
  auto &platformQPU = platformQPUs[qid];
  platformQPU->setExecutionContext(ctx);
}

/// Reset the execution context for this platform.
void quantum_platform::reset_exec_ctx(std::size_t qid) {
  validateQpuId(qid);
  auto &platformQPU = platformQPUs[qid];
  platformQPU->resetExecutionContext();
  executionContext = nullptr;
}

std::optional<QubitConnectivity> quantum_platform::connectivity() {
  return platformQPUs.front()->getConnectivity();
}

bool quantum_platform::is_simulator(const std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->isSimulator();
}

bool quantum_platform::is_remote(const std::size_t qpu_id) {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->isRemote();
}

bool quantum_platform::is_emulated(const std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->isEmulated();
}

bool quantum_platform::supports_conditional_feedback(
    const std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->supportsConditionalFeedback();
}

bool quantum_platform::supports_explicit_measurements(
    const std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->supportsExplicitMeasurements();
}

void quantum_platform::launchVQE(const std::string kernelName,
                                 const void *kernelArgs, gradient *gradient,
                                 const spin_op &H, optimizer &optimizer,
                                 const int n_params, const std::size_t shots) {
  std::size_t qpu_id = 0;

  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  {
    std::shared_lock lock(threadToQpuIdMutex);
    auto iter = threadToQpuId.find(tid);
    if (iter != threadToQpuId.end())
      qpu_id = iter->second;
  }

  auto &qpu = platformQPUs[qpu_id];
  qpu->launchVQE(kernelName, kernelArgs, gradient, H, optimizer, n_params,
                 shots);
}

RemoteCapabilities
quantum_platform::get_remote_capabilities(const std::size_t qpu_id) const {
  if (platformQPUs.empty())
    throw std::runtime_error("No QPUs are available for this target.");
  return platformQPUs[qpu_id]->getRemoteCapabilities();
}

KernelThunkResultType quantum_platform::launchKernel(
    const std::string &kernelName, KernelThunkType kernelFunc, void *args,
    std::uint64_t voidStarSize, std::uint64_t resultOffset,
    const std::vector<void *> &rawArgs) {
  std::size_t qpu_id = 0;

  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  {
    std::shared_lock lock(threadToQpuIdMutex);
    auto iter = threadToQpuId.find(tid);
    if (iter != threadToQpuId.end())
      qpu_id = iter->second;
  }
  auto &qpu = platformQPUs[qpu_id];
  return qpu->launchKernel(kernelName, kernelFunc, args, voidStarSize,
                           resultOffset, rawArgs);
}

void quantum_platform::launchKernel(const std::string &kernelName,
                                    const std::vector<void *> &rawArgs) {
  std::size_t qpu_id = 0;

  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  {
    std::shared_lock lock(threadToQpuIdMutex);
    auto iter = threadToQpuId.find(tid);
    if (iter != threadToQpuId.end())
      qpu_id = iter->second;
  }
  auto &qpu = platformQPUs[qpu_id];
  qpu->launchKernel(kernelName, rawArgs);
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

KernelThunkResultType altLaunchKernel(const char *kernelName,
                                      KernelThunkType kernelFunc,
                                      void *kernelArgs, std::uint64_t argsSize,
                                      std::uint64_t resultOffset) {
  ScopedTraceWithContext("altLaunchKernel", kernelName, argsSize);
  auto &platform = *getQuantumPlatformInternal();
  std::string kernName = kernelName;
  return platform.launchKernel(kernName, kernelFunc, kernelArgs, argsSize,
                               resultOffset, {});
}

KernelThunkResultType
streamlinedLaunchKernel(const char *kernelName,
                        const std::vector<void *> &rawArgs) {
  std::size_t argsSize = rawArgs.size();
  ScopedTraceWithContext("streamlinedLaunchKernel", kernelName, argsSize);
  auto &platform = *getQuantumPlatformInternal();
  std::string kernName = kernelName;
  platform.launchKernel(kernName, rawArgs);
  // NB: The streamlined launch will never return results. Use alt or hybrid if
  // the kernel returns results.
  return {};
}

KernelThunkResultType hybridLaunchKernel(const char *kernelName,
                                         KernelThunkType kernel, void *args,
                                         std::uint64_t argsSize,
                                         std::uint64_t resultOffset,
                                         const std::vector<void *> &rawArgs) {
  ScopedTraceWithContext("hybridLaunchKernel", kernelName);
  auto &platform = *getQuantumPlatformInternal();
  const std::string kernName = kernelName;
  if (platform.is_remote(platform.get_current_qpu())) {
    // This path should never call a kernel that returns results.
    platform.launchKernel(kernName, rawArgs);
    return {};
  }
  return platform.launchKernel(kernName, kernel, args, argsSize, resultOffset,
                               rawArgs);
}

} // namespace cudaq
