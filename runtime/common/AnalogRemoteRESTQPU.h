/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/AnalogEmulation.h"
#include "common/BaseRemoteRESTQPU.h"
#include "cudaq/platform/qpu_utils.h"
#include <memory>

namespace cudaq {

/// @brief Base QPU class for analog platforms like `quera` and `pasqal`.
/// Provides common functionality and implementation.
class AnalogRemoteRESTQPU : public BaseRemoteRESTQPU {
protected:
  /// @brief Local emulation engine, loaded by `setTargetBackend` when
  /// emulating.
  std::unique_ptr<analog::Engine> engine;

  /// @brief Targets that implement `emulateJob` return true.
  virtual bool supportsEmulation() const { return false; }

  /// @brief Emulate a kernel payload. Implementations build the job response in
  /// the vendor's format and return `serverHelper->processResults` of it, so
  /// emulated results are parsed exactly as remote ones.
  virtual sample_result emulateJob(const std::string & /*payload*/,
                                   std::size_t /*shots*/, std::size_t /*seed*/,
                                   analog::Engine & /*engine*/) {
    throw std::logic_error("Local emulation is not implemented.");
  }

  static std::runtime_error emulationUnavailable() {
    return std::runtime_error(
        "Local emulation is not available for this target. It requires a "
        "target with emulation support and the CUDA-Q dynamics backend "
        "library.");
  }

  /// @brief Return the payload of an analog Hamiltonian kernel launch.
  static std::string analogPayload(const CompiledModule &module,
                                   const KernelArgs &args) {
    const auto &kernelName = module.getName();
    if (!cudaq::detail::isAnalogHamiltonianKernel(kernelName))
      throw std::runtime_error(
          "Arbitrary kernel execution is not supported on this target.");

    CUDAQ_INFO("Launching analog kernel ({})", kernelName);
    const auto packed = args.getPacked();
    if (!packed)
      throw std::runtime_error(
          "Analog Hamiltonian launch requires a packed JSON payload.");
    return std::string(reinterpret_cast<const char *>(packed->data.data()),
                       packed->data.size());
  }

  /// @brief Submit an analog payload to the remote backend.
  detail::future launchRemote(const sample_policy &policy,
                              const CompiledModule &module,
                              const std::string &payload) {
    std::vector<cudaq::KernelExecution> codes;
    codes.push_back(KernelExecution{.name = module.getName(), .code = payload});
    executor->setShots(policy.options.shots);
    return executor->execute(codes);
  }

public:
  /// @brief Check if this is a remote target
  virtual bool isRemote() override { return !emulate; }

  /// @brief Check if this is an emulated target
  virtual bool isEmulated() override { return emulate; }

  using BaseRemoteRESTQPU::getCompileTarget;
  using BaseRemoteRESTQPU::launchKernel;

  CompileTarget getCompileTarget(const RuntimeTarget * = nullptr) override {
    return {.overrideAOTCompilation = false};
  }

  /// @brief Set the target backend, loading the local emulation engine when
  /// `emulate` is requested.
  void setTargetBackend(const std::string &backend) override {
    BaseRemoteRESTQPU::setTargetBackend(backend);
    engine.reset();
    if (!emulate)
      return;
    if (supportsEmulation())
      engine = analog::loadEngine("dynamics");
    if (!engine)
      throw emulationUnavailable();
  }

  /// @brief Remote only: emulated targets run asynchronous launches on the
  /// platform's execution queue, as digital emulation does.
  async_sample_result launchKernel(const async_sample_policy &policy,
                                   const CompiledModule &module,
                                   KernelArgs args) override {
    if (emulate)
      throw std::runtime_error(
          "Asynchronous launches are not supported for emulated targets.");
    return async_sample_result(
        launchRemote(policy.inner, module, analogPayload(module, args)));
  }

  /// @brief Launch a kernel with the given arguments, remotely or locally
  /// when emulating. Only analog Hamiltonian kernels are supported.
  sample_result launchKernel(const sample_policy &policy,
                             const CompiledModule &module,
                             KernelArgs args) override {
    const auto payload = analogPayload(module, args);
    if (!emulate)
      return launchRemote(policy, module, payload).get();
    if (!engine)
      throw emulationUnavailable();
    // Use the thread-specific seed, as digital emulation does; 0 is unset.
    return emulateJob(payload, policy.options.shots, cudaq::get_random_seed(),
                      *engine);
  }
};

} // namespace cudaq
