/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/Executor.h"
#include "common/KernelExecution.h"
#include "common/Resources.h"
#include "common/ServerHelper.h"
#include "nvqir/AnalysisScope.h"
#include "nvqir/resourcecounter/ResourceCounterScope.h"
#include "cudaq/Target/TargetConfig.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/platform/platform_iface.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

namespace nvqir {
// QIR helper to retrieve the output log.
std::string_view getQirOutputLog();
} // namespace nvqir

namespace cudaq {
void set_random_seed(std::size_t seed);
std::size_t get_random_seed();
class noise_model;

inline observe_result observeResultFromCounts(const observe_policy &policy,
                                              sample_result data) {
  double sum = 0.0;
  for (const auto &term : policy.spin) {
    if (term.is_identity())
      sum += term.evaluate_coefficient().real();
    else
      sum += data.expectation(term.get_term_id()) *
             term.evaluate_coefficient().real();
  }
  return observe_result(sum, policy.spin, data);
}

class BaseRemoteRESTQPU : public QPU {
protected:
  /// The number of shots
  std::optional<int> nShots;

  /// @brief the platform file path
  std::filesystem::path platformPath;

  /// @brief The name of the QPU being targeted
  std::string qpuName;

  // Pointer to the concrete Executor for this QPU
  std::unique_ptr<cudaq::Executor> executor;

  /// @brief Pointer to the forward declared ServerHelper, provides
  /// specific JSON payloads and POST/GET URL paths.
  cudaq::owning_ptr<cudaq::ServerHelper> serverHelper;

  /// @brief Mapping of general key-values for backend configuration.
  std::map<std::string, std::string> backendConfig;

  /// @brief Flag indicating whether we should emulate execution locally.
  bool emulate = false;

  /// @brief The target configuration
  cudaq::config::TargetConfig targetConfig;

public:
  // This class overrides `launchKernel(dem_policy)` (local DEM generation,
  // shared by all remote QPUs) but not the `sample`/`observe` overloads (those
  // are overridden in the leaf QPUs). Re-import QPU's `launchKernel` overloads
  // so this partial override does not trip nvcc "overloaded virtual function
  // only partially overridden" error.
  using QPU::launchKernel;

  /// @brief The constructor
  BaseRemoteRESTQPU() : QPU() {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
  }

  BaseRemoteRESTQPU(BaseRemoteRESTQPU &&) = delete;
  virtual ~BaseRemoteRESTQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  /// @brief Return true if the current backend is a simulator
  /// @return
  bool isSimulator() override { return emulate; }

  /// @brief Return true if the current backend supports explicit measurements
  bool supportsExplicitMeasurements() override { return false; }

  /// Provide the number of shots
  void setShots(int _nShots) override {
    nShots = _nShots;
    executor->setShots(static_cast<std::size_t>(_nShots));
  }

  /// Clear the number of shots
  void clearShots() override { nShots = std::nullopt; }
  virtual bool isRemote() override { return !emulate; }

  /// @brief Return true if locally emulating a remote QPU
  virtual bool isEmulated() override { return emulate; }

  /// @brief Set the noise model, only allow this for
  /// emulation.
  void setNoiseModel(const cudaq::noise_model *model) override {
    if (!emulate && model)
      throw std::runtime_error(
          "Noise modeling is not allowed on remote physical quantum backends.");

    noiseModel = model;
  }

  /// Store the execution context for launchKernel
  void
  configureExecutionContext(cudaq::ExecutionContext &context) const override {
    // Re-entry guard for analysis scopes. Without this, a host callback issued
    // by the analysis simulator (e.g. a `choice` function that calls
    // `cudaq::sample`) could launch a second kernel through this transport
    // while the outer scope is still active.
    if (nvqir::AnalysisScope::is_active() && context.name != "resource-count")
      throw std::runtime_error(
          "Illegal use of a resource counter on a remote QPU.");

    CUDAQ_INFO("Remote Rest QPU preparing execution context for {}",
               context.name);

    if (context.executionManager)
      context.executionManager->configureExecutionContext(context);
  }

  void
  finalizeExecutionContext(cudaq::ExecutionContext &context) const override {
    if (context.executionManager)
      context.executionManager->finalizeExecutionContext(context);
  }

  void beginExecution() override {
    auto executionContext = getExecutionContext();
    if (executionContext && executionContext->executionManager)
      executionContext->executionManager->beginExecution();
  }

  void endExecution() override {
    auto executionContext = getExecutionContext();
    if (executionContext && executionContext->executionManager)
      getExecutionContext()->executionManager->endExecution();
  }

  /// @brief This setTargetBackend override is in charge of reading the
  /// specific target backend configuration file (bundled as part of this
  /// CUDA-Q installation) and extract MLIR lowering pipelines and
  /// specific code generation output required by this backend (QIR/QASM2).
  void setTargetBackend(const std::string &backend) override {
    CUDAQ_INFO("Remote REST platform is targeting {}.", backend);

    // First we see if the given backend has extra config params
    auto mutableBackend = backend;
    if (mutableBackend.find(";") != std::string::npos) {
      auto split = cudaq::split(mutableBackend, ';');
      mutableBackend = split[0];
      // Must be key-value pairs, therefore an even number of values here
      if ((split.size() - 1) % 2 != 0)
        throw std::runtime_error(
            "Backend config must be provided as key-value pairs: " +
            std::to_string(split.size()));

      // Add to the backend configuration map
      for (std::size_t i = 1; i < split.size(); i += 2) {
        // No need to decode trivial true/false values
        if (split[i + 1].starts_with("base64_")) {
          split[i + 1].erase(0, 7); // erase "base64_"
          std::string decodedStr = detail::decodeBase64(split[i + 1]);
          CUDAQ_INFO("Decoded {} parameter from '{}' to '{}'", split[i],
                     split[i + 1], decodedStr);
          backendConfig.insert({split[i], decodedStr});
        } else {
          backendConfig.insert({split[i], split[i + 1]});
        }
      }
    }

    // Turn on emulation mode if requested
    auto iter = backendConfig.find("emulate");
    emulate = iter != backendConfig.end() && iter->second == "true";

    /// Once we know the backend, we should search for the configuration file
    /// from there we can get the URL/PORT and the required MLIR pass pipeline.
    std::string fileName = mutableBackend + std::string(".yml");
    auto configFilePath =
        detail::getTargetConfigPath(backend, platformPath / fileName);
    backendConfig.erase("__yml_path");
    CUDAQ_INFO("Config file path = {}", configFilePath.string());
    targetConfig = cudaq::config::loadTargetConfig(configFilePath);
    detail::loadTargetPluginLibraries(mutableBackend, configFilePath,
                                      targetConfig);

    // Set the qpu name
    qpuName = mutableBackend;
    // Create the ServerHelper for this QPU and give it the backend config
    detail::initServerHelperAndExecutor(qpuName, backendConfig, targetConfig,
                                        serverHelper, executor);
  }

  using QPU::getCompileTarget;
  std::unique_ptr<CompileTarget>
  getCompileTarget(const other_policies &, ExecutionContext *ctx) override {
    if (!ctx)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), cudaq::run(), or cudaq::contrib::draw().");

    auto target = std::make_unique<CompileTarget>(
        targetConfig, backendConfig, emulate,
        serverHelper->getPipelineSubstitutions(platformPath));
    target->pipelineConfig.replaceStateWithKernel = true;
    target->overrideAOTCompilation = true;
    if (ctx && ctx->name == "resource-count")
      target->emitResourceCounts = true;

    return target;
  }

  std::unique_ptr<CompileTarget>
  getCompileTarget(const sample_policy &policy) override {
    auto target = std::make_unique<CompileTarget>(
        targetConfig, backendConfig, emulate,
        serverHelper->getPipelineSubstitutions(platformPath));
    target->supportConditionalsOnMeasureResults = !emulate;
    target->pipelineConfig.addMeasurements = true;
    target->storeReorderIdx = true;
    target->pipelineConfig.replaceStateWithKernel = true;
    target->overrideAOTCompilation = true;
    return target;
  }

  std::unique_ptr<CompileTarget>
  getCompileTarget(const observe_policy &policy) override {
    auto target = std::make_unique<CompileTarget>(
        targetConfig, backendConfig, emulate,
        serverHelper->getPipelineSubstitutions(platformPath));
    target->overrideAOTCompilation = true;
    target->pauliTermSplitObservable = policy.spin;
    target->pipelineConfig.replaceStateWithKernel = true;
    return target;
  }

  std::unique_ptr<CompileTarget> getCompileTarget(const run_policy &) override {
    auto target = std::make_unique<CompileTarget>(
        targetConfig, backendConfig, emulate,
        serverHelper->getPipelineSubstitutions(platformPath));
    target->pipelineConfig.replaceStateWithKernel = true;
    target->overrideAOTCompilation = true;
    return target;
  }

  /// Build a local JIT artifact for DEM analysis. No provider target code is
  /// emitted or submitted while this policy is active.
  std::unique_ptr<CompileTarget> getCompileTarget(const dem_policy &) override {
    // Skip pipeline substitutions: this path never builds the lowering pipeline
    // and should not trigger server-helper side effects (e.g. IQM arch fetch).
    auto target =
        std::make_unique<CompileTarget>(targetConfig, backendConfig, emulate);
    target->pipelineConfig.replaceStateWithKernel = true;
    target->overrideAOTCompilation = true;
    target->emitJit = true;
    target->emitTargetCode = false;
    target->pipelineConfig.skipTargetLoweringPipeline = true;
    return target;
  }

  /// Generate the DEM locally while preserving the selected remote target.
  dem_result launchKernel(const dem_policy &policy,
                          const CompiledModule &module,
                          KernelArgs args) override {
    CUDAQ_INFO("BaseRemoteRESTQPU::launchKernel {} locally", policy.name);
    if (!module.getJit())
      throw std::runtime_error(
          "Remote QPU could not produce the local JIT artifact required for "
          "detector error model generation.");

    return cudaq::ExecutionManager::with_default_em(policy, [&] {
      [[maybe_unused]] auto kernelResult = runJITCompiledModule(module, args);
    });
  }

  void completeLaunchKernel(const std::string &kernelName,
                            std::vector<cudaq::KernelExecution> &&codes) {
    auto executionContext = cudaq::getExecutionContext();

    // Check to see if we are simply drawing the circuit. If so, perform the
    // trace here and then return.
    if (executionContext->name == "tracer" && codes.size() == 1) {
      assert(codes[0].jit);
      cudaq::ExecutionContext context("tracer");
      context.executionManager = cudaq::getDefaultExecutionManager();
      context.hasConditionalsOnMeasureResults =
          codes[0].hasConditionalsOnMeasureResults;
      cudaq::platform::with_execution_context(
          context, [&]() { codes[0].jit->run(kernelName); });
      executionContext->kernelTrace = std::move(context.kernelTrace);
      return;
    }

    if (executionContext->name == "resource-count") {
      assert(codes.size() == 1 && codes[0].jit && codes[0].resourceCounts);
      cudaq::ExecutionContext context("resource-count");
      context.executionManager = cudaq::getDefaultExecutionManager();
      context.hasConditionalsOnMeasureResults =
          codes[0].hasConditionalsOnMeasureResults;
      nvqir::resource_counter::prepopulate(
          std::move(codes[0].resourceCounts.value()));
      cudaq::platform::with_execution_context(
          context, [&]() { codes[0].jit->run(kernelName); });
      return;
    }

    return;
  }

  run_result completeLaunchKernel(const run_policy &policy,
                                  const std::string &kernelName,
                                  std::vector<cudaq::KernelExecution> &&codes) {
    // The synchronous run policy is only ever used for the emulation path;
    // remote execution is dispatched through async_run_policy and the executor.
    assert(emulate);

    // Seed the simulator RNG for reproducibility. If seed is 0, then it has
    // not been set.
    std::size_t seed = cudaq::get_random_seed();
    if (seed > 0)
      cudaq::set_random_seed(seed);

    // cudaq::run kernels should only generate one JIT'ed kernel, which we
    // invoke once per shot and let the execution manager collect the QIR
    // output log.
    assert(codes.size() == 1 && codes[0].jit);
    return cudaq::ExecutionManager::with_default_em(policy, [&] {
      // Run each shot with the thread-local execution context cleared.
      // CircuitSimulator::deallocateQubits skips deallocation while an
      // execution context is set, so if the context stays set across the whole
      // shot loop the per-shot qubits allocated by the kernel accumulate and
      // blow up the simulator state (OOM). Clearing it lets each shot fully
      // deallocate, matching the pre-policy behavior where the shot loop ran on
      // a separate thread with no execution context.
      auto *savedContext = cudaq::getExecutionContext();
      cudaq::detail::resetExecutionContext();
      cudaq::detail::try_finally(
          [&] {
            for (std::size_t shot = 0; shot < policy.shots; shot++)
              codes[0].jit->run(kernelName);
          },
          [&] {
            if (savedContext)
              cudaq::detail::setExecutionContext(savedContext);
          });
    });
  }

  async_run_result
  completeLaunchKernel(const async_run_policy &policy,
                       const std::string &kernelName,
                       std::vector<cudaq::KernelExecution> &&codes) {
    executor->setShots(policy.inner.shots);
    assert(!emulate);
    if (getEnvBool("DISABLE_REMOTE_SEND", false)) {
      auto rawOutput = std::make_shared<std::vector<char>>();
      std::promise<sample_result> promise;
      auto future = promise.get_future();
      promise.set_value({});
      return async_run_result(cudaq::detail::future(std::move(future)),
                              std::move(rawOutput));
    }

    auto rawOutput = std::make_shared<std::vector<char>>();
    auto future = executor->execute(
        codes, cudaq::detail::ExecutionContextType::run, rawOutput.get());
    return async_run_result(std::move(future), std::move(rawOutput));
  }

  async_sample_result
  completeLaunchKernel(const async_sample_policy &policy,
                       const std::string &kernelName,
                       std::vector<cudaq::KernelExecution> &&codes) {
    // Get the current execution context and number of shots
    std::size_t localShots = 1000;
    if (policy.inner.options.shots != std::numeric_limits<std::size_t>::max() &&
        policy.inner.options.shots != 0)
      localShots = policy.inner.options.shots;

    executor->setShots(localShots);

    // Execute the codes produced in quake lowering
    // Allow developer to disable remote sending (useful for debugging IR)
    assert(!emulate);
    if (getEnvBool("DISABLE_REMOTE_SEND", false))
      return {};
    // Cannot be observe and run at the same time
    const cudaq::detail::ExecutionContextType execType =
        cudaq::detail::ExecutionContextType::sample;

    auto future = executor->execute(codes, execType);
    return async_sample_result(std::move(future));
  }

  sample_result
  completeLaunchKernel(const sample_policy &policy,
                       const std::string &kernelName,
                       std::vector<cudaq::KernelExecution> &&codes) {

    // Get the current execution context and number of shots
    std::size_t localShots = 1000;
    if (policy.options.shots != std::numeric_limits<std::size_t>::max() &&
        policy.options.shots != 0)
      localShots = policy.options.shots;

    // If emulation requested, then just grab the function and invoke it with
    // the simulator
    assert(emulate);

    // Fetch the thread-specific seed outside and then pass it inside.
    std::size_t seed = cudaq::get_random_seed();

    // Launch the execution of the simulated jobs asynchronously
    std::vector<cudaq::ExecutionResult> results;

    // If seed is 0, then it has not been set.
    if (seed > 0)
      cudaq::set_random_seed(seed);

    // Otherwise, this is a non-adaptive sampling or observe.
    // We run the kernel(s) (multiple kernels if this is a multi-term
    // observe) one time each.
    for (std::size_t i = 0; i < codes.size(); i++) {
      cudaq::ExecutionContext context("sample", localShots);
      context.hasConditionalsOnMeasureResults =
          codes[i].hasConditionalsOnMeasureResults;
      sample_policy localPolicy;
      localPolicy.options.shots = localShots;
      localPolicy.reorderIdx = std::move(codes[i].mapping_reorder_idx);
      localPolicy.kernelName = kernelName;
      assert(codes[i].jit);
      auto result = detail::with_policy_and_ctx(localPolicy, context, [&]() {
        return cudaq::ExecutionManager::with_default_em(
            localPolicy, [&]() { codes[i].jit->run(kernelName); });
      });

      // For each register, add the context results into result.
      for (auto &regName : result.register_names()) {
        results.emplace_back(result.to_map(regName), regName);
        results.back().sequentialData = result.sequential_data(regName);
      }
    }
    return cudaq::sample_result(results);
  }

  async_observe_result
  completeLaunchKernel(const async_observe_policy &policy,
                       const std::string &kernelName,
                       std::vector<cudaq::KernelExecution> &&codes) {
    std::size_t localShots = 1000;
    if (policy.inner.options.shots > 0)
      localShots = static_cast<std::size_t>(policy.inner.options.shots);

    executor->setShots(localShots);

    assert(!emulate);
    if (getEnvBool("DISABLE_REMOTE_SEND", false))
      return {};

    auto future =
        executor->execute(codes, cudaq::detail::ExecutionContextType::observe);
    return async_observe_result(std::move(future), &policy.inner.spin);
  }

  observe_result
  completeLaunchKernel(const observe_policy &policy,
                       const std::string &kernelName,
                       std::vector<cudaq::KernelExecution> &&codes) {
    std::size_t localShots = 1000;
    if (policy.options.shots > 0)
      localShots = static_cast<std::size_t>(policy.options.shots);

    assert(emulate);

    std::size_t seed = cudaq::get_random_seed();
    if (seed > 0)
      cudaq::set_random_seed(seed);

    std::vector<cudaq::ExecutionResult> results;
    for (std::size_t i = 0; i < codes.size(); i++) {
      cudaq::ExecutionContext context("sample", localShots);
      context.hasConditionalsOnMeasureResults =
          codes[i].hasConditionalsOnMeasureResults;
      sample_policy localPolicy;
      localPolicy.options.shots = localShots;
      localPolicy.reorderIdx = std::move(codes[i].mapping_reorder_idx);
      localPolicy.kernelName = kernelName;
      assert(codes[i].jit);
      auto result = detail::with_policy_and_ctx(localPolicy, context, [&]() {
        return cudaq::ExecutionManager::with_default_em(
            localPolicy, [&]() { codes[i].jit->run(kernelName); });
      });

      // Use the code name instead of the global register.
      results.emplace_back(result.to_map(), codes[i].name);
      results.back().sequentialData = result.sequential_data();
    }
    return observeResultFromCounts(policy, cudaq::sample_result(results));
  }
};

} // namespace cudaq
