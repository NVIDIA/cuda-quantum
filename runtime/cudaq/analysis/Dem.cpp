/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/dem.h"
#include "common/CompiledModule.h"
#include "common/ExecutionContext.h"
#include "common/KernelArgs.h"
#include "common/NoiseModel.h"
#include "nvqir/AnalysisScope.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/RecordedCircuit.h"
#include "nvqir/dem/DemScope.h"
#include "stim.h"
#include "cudaq/platform.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq::details {

namespace {

std::string runDemFromKernelImpl(const std::string &kernelName,
                                 cudaq::quantum_platform &,
                                 const cudaq::noise_model *noise,
                                 const std::function<void()> &kernel,
                                 std::string plugin_name);

/// Serialize concurrent `dem_from_kernel` calls across threads.
///
/// `nvqir::AnalysisScope` claims a thread-local slot, but the simulator
/// pointer it resolves through `dlsym(getCircuitSimulator_<plugin>)` is a
/// process-wide singleton owned by the NVQIR plugin. The mutex therefore guards
/// the kernel-execution + circuit-readback pair plus the on-entry simulator
/// reset done by `nvqir::dem::make_scope`'s `on_enter` hook; once we have a
/// local `stim::Circuit` copy, `ErrorAnalyzer` operates on it lock-free so
/// concurrent analyses do not serialize on the error analysis step.
std::mutex &demMutex() {
  static std::mutex m;
  return m;
}

void launchLocallyForDem(const cudaq::AnyModule &module,
                         cudaq::KernelArgs args) {
  if (std::holds_alternative<cudaq::SourceModule>(module)) {
    const auto &src = std::get<cudaq::SourceModule>(module);
    auto rawFn = src.getFunctionPtr();
    if (!rawFn)
      throw std::runtime_error(
          "`cudaq::dem_from_kernel`: missing compiled kernel function for "
          "kernel '" +
          src.getName() + "'.");
    auto packed = args.getPacked();
    void *argData = packed ? packed->data.data() : nullptr;
    rawFn->getFn()(argData, /*isRemote=*/false);
    return;
  }

  const auto &compiled = std::get<cudaq::CompiledModule>(module);
  auto jit = compiled.getJit();
  if (!jit)
    throw std::runtime_error("`cudaq::dem_from_kernel`: compiled module '" +
                             compiled.getName() +
                             "' does not contain a JIT artifact.");
  auto rawArgs = args.getTypeErased().value_or(std::span<void *const>{});
  auto funcPtr = jit->getFn();
  const auto &resultInfo = compiled.getResultInfo();
  if (!compiled.isFullySpecialized()) {
    auto argsCreator = compiled.getArgsCreator();
    void *buff = nullptr;
    argsCreator(static_cast<const void *>(rawArgs.data()), &buff);
    reinterpret_cast<cudaq::KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    if (resultInfo.hasResult()) {
      auto offset = compiled.getReturnOffset().value();
      std::memcpy(rawArgs.back(), static_cast<char *>(buff) + offset,
                  resultInfo.getBufferSize());
    }
    std::free(buff);
    return;
  }
  if (resultInfo.hasResult()) {
    void *buff = const_cast<void *>(rawArgs.back());
    reinterpret_cast<cudaq::KernelThunkResultType (*)(void *, bool)>(funcPtr)(
        buff, /*client_server=*/false);
    return;
  }
  funcPtr();
}

template <typename Callable>
void withDemExecutionContext(cudaq::ExecutionContext &ctx,
                             Callable &&callable) {
  auto *outerContext = cudaq::getExecutionContext();
  cudaq::detail::setExecutionContext(&ctx);
  auto cleanup = [&]() {
    cudaq::detail::resetExecutionContext();
    if (outerContext)
      cudaq::detail::setExecutionContext(outerContext);
  };
  cudaq::detail::try_finally(std::forward<Callable>(callable), cleanup);
}

std::string runDemFromKernelImpl(const std::string &kernelName,
                                 cudaq::quantum_platform &,
                                 const cudaq::noise_model *noise,
                                 const std::function<void()> &kernel,
                                 std::string plugin_name) {
  cudaq::ExecutionContext ctx("dem");
  ctx.kernelName = kernelName;
  ctx.qpuId = cudaq::getCurrentQpuId();
  ctx.asyncExec = false;
  if (noise)
    ctx.noiseModel = noise;

  // The recorded circuit lives on the shared `NVQIR` plugin singleton, so
  // `kernel()` and the readback below MUST be serialized across DEM
  // calls. The `ErrorAnalyzer` step operates on a local `stim::Circuit`
  // copy.
  stim::Circuit recorded;
  {
    if (nvqir::AnalysisScope::is_active())
      throw std::runtime_error("`cudaq::dem_from_kernel`: launching an "
                               "analysis primitive from inside another "
                               "analysis primitive is not supported.");

    std::lock_guard<std::mutex> serialize(demMutex());

    // RAII: scope releases the thread-local override on every exit path,
    // including exceptions thrown from the kernel.
    auto demScope = nvqir::dem::make_scope(std::move(plugin_name));

    auto &recorder =
        dynamic_cast<nvqir::RecordedCircuit &>(demScope.simulator());

    // Propagate `ctx.noiseModel` onto the analysis simulator's member
    // via the standard `CircuitSimulator::configureExecutionContext`
    // API.
    demScope.simulator().configureExecutionContext(ctx);

    ctx.executeKernelApi = [](const cudaq::AnyModule &module,
                              const cudaq::KernelArgs &args) {
      launchLocallyForDem(module, args);
    };
    // Set the thread-local context without running the active platform's
    // configure/finalize hooks. DEM drives the Stim analysis simulator directly
    // through `demScope`; the launch hook only redirects kernels to local JIT /
    // raw-function execution so remote targets do not send analysis work to
    // their transport.
    cudaq::detail::try_finally(
        [&]() {
          withDemExecutionContext(ctx, [&]() {
            kernel();
            recorded = *recorder.circuit();
          });
        },
        [&]() { ctx.executeKernelApi = nullptr; });
  }

  stim::DetectorErrorModel stimDem =
      stim::ErrorAnalyzer::circuit_to_detector_error_model(
          recorded,
          /*decompose_errors=*/false,
          /*fold_loops=*/false,
          /*allow_gauge_detectors=*/false,
          /*approximate_disjoint_errors_threshold=*/0,
          /*ignore_decomposition_failures=*/false,
          /*block_decomposition_from_introducing_remnant_edges=*/false);
  return stimDem.str();
}

} // namespace

/// `extern "C"` entry point resolved by `cudaq::details::runDemFromKernel`
/// via `dlopen` + `dlsym` on `libcudaq-analysis.so`.
extern "C" std::string cudaq_runDemFromKernel(
    const std::string &kernelName, cudaq::quantum_platform &platform,
    const cudaq::noise_model *noise, const std::function<void()> &wrappedKernel,
    const std::string &plugin_name) {
  return runDemFromKernelImpl(kernelName, platform, noise, wrappedKernel,
                              plugin_name);
}

} // namespace cudaq::details
