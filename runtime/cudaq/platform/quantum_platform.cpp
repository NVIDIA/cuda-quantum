/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/quantum_platform.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
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

namespace cudaq {

// These functions are defined elsewhere, but
// we are going to use them here.
std::string get_quake(const std::string &);

static quantum_platform *platform;
inline static constexpr std::string_view GetQuantumPlatformSymbol =
    "getQuantumPlatform";

void setQuantumPlatformInternal(quantum_platform *p) {
  cudaq::info("external caller setting the platform.");
  platform = p;
}

/// @brief Get the provided platform plugin
/// @return
quantum_platform *getQuantumPlatformInternal() {
  if (platform)
    return platform;
  platform = cudaq::getUniquePluginInstance<quantum_platform>(
      GetQuantumPlatformSymbol);
  return platform;
}

void quantum_platform::set_noise(const noise_model *model) {
  auto &platformQPU = platformQPUs[platformCurrentQPU];
  platformQPU->setNoiseModel(model);
}

void quantum_platform::reset_noise() { set_noise(nullptr); }

std::future<sample_result>
quantum_platform::enqueueAsyncTask(const std::size_t qpu_id,
                                   KernelExecutionTask &task) {
  std::promise<sample_result> promise;
  auto f = promise.get_future();
  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), t = std::move(task)]() mutable {
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

void quantum_platform::set_current_qpu(const std::size_t device_id) {
  if (device_id >= platformNumQPUs) {
    throw std::invalid_argument(
        "QPU device id is not valid (greater than number of available QPUs).");
  }

  platformCurrentQPU = device_id;
  threadToQpuId.emplace(
      std::hash<std::thread::id>{}(std::this_thread::get_id()), device_id);
}

std::size_t quantum_platform::get_current_qpu() { return platformCurrentQPU; }

// Specify the execution context for this platform.
// This delegates to the targeted QPU
void quantum_platform::set_exec_ctx(cudaq::ExecutionContext *ctx,
                                    std::size_t qid) {
  executionContext = ctx;
  auto &platformQPU = platformQPUs[qid];
  platformQPU->setExecutionContext(ctx);
}

/// Reset the execution context for this platform.
void quantum_platform::reset_exec_ctx(std::size_t qid) {
  auto &platformQPU = platformQPUs[qid];
  platformQPU->resetExecutionContext();
  executionContext = nullptr;
}

std::optional<QubitConnectivity> quantum_platform::connectivity() {
  return platformQPUs.front()->getConnectivity();
}

bool quantum_platform::is_simulator(const std::size_t qpu_id) const {
  return platformQPUs[qpu_id]->isSimulator();
}

bool quantum_platform::is_remote(const std::size_t qpu_id) {
  return platformQPUs[qpu_id]->isRemote();
}

bool quantum_platform::is_emulated(const std::size_t qpu_id) const {
  return platformQPUs[qpu_id]->isEmulated();
}

bool quantum_platform::supports_conditional_feedback(
    const std::size_t qpu_id) const {
  return platformQPUs[qpu_id]->supportsConditionalFeedback();
}

void quantum_platform::launchKernel(std::string kernelName,
                                    void (*kernelFunc)(void *), void *args,
                                    std::uint64_t voidStarSize,
                                    std::uint64_t resultOffset) {
  std::size_t qpu_id = 0;

  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  auto iter = threadToQpuId.find(tid);
  if (iter != threadToQpuId.end())
    qpu_id = iter->second;

  auto &qpu = platformQPUs[qpu_id];
  qpu->launchKernel(kernelName, kernelFunc, args, voidStarSize, resultOffset);
}

} // namespace cudaq

void cudaq::altLaunchKernel(const char *kernelName, void (*kernelFunc)(void *),
                            void *kernelArgs, std::uint64_t argsSize,
                            std::uint64_t resultOffset) {
  ScopedTrace trace("altLaunchKernel", kernelName, argsSize);
  auto &platform = *cudaq::getQuantumPlatformInternal();
  std::string kernName = kernelName;
  platform.launchKernel(kernName, kernelFunc, kernelArgs, argsSize,
                        resultOffset);
}
