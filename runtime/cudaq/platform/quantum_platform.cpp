/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifdef __clang__
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wsuggest-override"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "cudaq/platform/quantum_platform.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
#include "nvqpp_config.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/qis/qudit.h"
#include <fmt/core.h>
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

thread_local static quantum_platform *platform;
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

void quantum_platform::set_noise(noise_model *model) {
  auto &platformQPU = platformQPUs[platformCurrentQPU];
  platformQPU->setNoiseModel(model);
}

std::future<sample_result>
quantum_platform::enqueueAsyncTask(const std::size_t qpu_id,
                                   KernelExecutionTask &task) {
  set_current_qpu(qpu_id);

  std::promise<sample_result> promise;
  auto f = promise.get_future();
  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), t = std::move(task)]() mutable {
        auto counts = t();
        p.set_value(counts);
      });

  platformQPUs[platformCurrentQPU]->enqueue(wrapped);
  return f;
}

void quantum_platform::set_current_qpu(const std::size_t device_id) {
  if (device_id >= platformNumQPUs) {
    throw std::invalid_argument(
        "QPU device id is not valid (greater than number of available QPUs).");
  }

  platformCurrentQPU = device_id;
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

bool quantum_platform::supports_conditional_feedback(
    const std::size_t qpu_id) const {
  return platformQPUs[qpu_id]->supportsConditionalFeedback();
}

void quantum_platform::launchKernel(std::string kernelName,
                                    void (*kernelFunc)(void *), void *args,
                                    std::uint64_t voidStarSize,
                                    std::uint64_t resultOffset) {
  auto &qpu = platformQPUs[platformCurrentQPU];
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
