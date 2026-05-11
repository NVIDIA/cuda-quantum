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
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/platform/platform_iface.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/JIT.h"
#include <filesystem>
#include <fstream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

namespace nvqir {
// QIR helper to retrieve the output log.
std::string_view getQirOutputLog();
void setResourceCounts(cudaq::Resources &&);
bool isUsingResourceCounterSimulator();
} // namespace nvqir

namespace cudaq {
void set_random_seed(std::size_t seed);
std::size_t get_random_seed();
class noise_model;

class BaseRemoteRESTQPU : public QPU {
protected:
  using Compiler = cudaq_internal::compiler::Compiler;

  /// The number of shots
  std::optional<int> nShots;

  /// @brief the platform file path
  std::filesystem::path platformPath;

  /// @brief The name of the QPU being targeted
  std::string qpuName;

  /// @brief Name of code generation target (e.g. `qir-adaptive`, `qir-base`,
  /// `qasm2`, `iqm`)
  std::string codegenTranslation = "";

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
    // This check ensures that a kernel is not called whilst actively being
    // used for resource counting (implying that the kernel was somehow
    // invoked from inside the choice function). This check may want to
    // be expanded more broadly to ensure that the execution context is
    // always fully reset, implying the end of the invocation, being being
    // set again, signaling a new invocation.
    if (nvqir::isUsingResourceCounterSimulator() &&
        context.name != "resource-count")
      throw std::runtime_error(
          "Illegal use of resource counter simulator! (Did you attempt to run "
          "a kernel inside of a choice function?)");

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
    auto configFilePath = platformPath / fileName;
    CUDAQ_INFO("Config file path = {}", configFilePath.string());
    std::ifstream configFile(configFilePath.string());
    std::string configYmlContents((std::istreambuf_iterator<char>(configFile)),
                                  std::istreambuf_iterator<char>());
    detail::parseTargetConfigYml(configYmlContents, targetConfig);

    // Keep a local copy for capability queries like
    // supportsConditionalFeedback(). The Compiler computes and validates the
    // full codegen configuration for lowering.
    codegenTranslation = targetConfig.getCodeGenSpec(backendConfig);

    // Set the qpu name
    qpuName = mutableBackend;
    // Create the ServerHelper for this QPU and give it the backend config
    detail::initServerHelperAndExecutor(qpuName, backendConfig, targetConfig,
                                        serverHelper, executor);
  }

  /// @brief Launch the kernel. Extract the Quake code and lower to the
  /// representation required by the targeted backend. Handle all pertinent
  /// modifications for the execution context as well as asynchronous or
  /// synchronous invocation.
  KernelThunkResultType launchKernel(const SourceModule &src,
                                     KernelArgs args) override {
    const auto &kernelName = src.getName();
    CUDAQ_INFO("launching remote rest kernel ({})", kernelName);

    auto executionContext = cudaq::getExecutionContext();

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), cudaq::run(), or cudaq::contrib::draw().");

    auto [moduleOp, context] = Compiler::loadQuakeCodeByName(kernelName);

    // Get the Quake code, lowered according to config file.
    Compiler compiler(serverHelper.get(), backendConfig, targetConfig,
                      noiseModel, emulate);
    auto codes =
        compiler.lowerQuakeCode(executionContext, kernelName, moduleOp, args);
    completeLaunchKernel(kernelName, std::move(codes));

    // NB: Kernel should/will never return dynamic results.
    return {};
  }

  KernelThunkResultType launchModule(const CompiledModule &compiled,
                                     KernelArgs args) override {
    CUDAQ_INFO("launching remote rest kernel via module ({})",
               compiled.getName());

    Compiler compiler(serverHelper.get(), backendConfig, targetConfig,
                      noiseModel, emulate);
    auto codes = compiler.emitKernelExecutions(compiled);
    completeLaunchKernel(compiled.getName(), std::move(codes));
    return {};
  }

  CompiledModule compileModule(const SourceModule &src, KernelArgs args,
                               bool isEntryPoint) override {
    const auto &kernelName = src.getName();
    auto mlirArt = src.getMlir();
    if (!mlirArt)
      throw std::runtime_error(
          "BaseRemoteRESTQPU::compileModule requires an MLIR artifact on "
          "the SourceModule for kernel '" +
          kernelName + "'.");
    auto modulePtr = mlirArt->getOpaqueModulePtr();
    CUDAQ_INFO("specializing remote rest kernel via module ({})", kernelName);
    auto executionContext = cudaq::getExecutionContext();

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), cudaq::run(), or cudaq::contrib::draw().");

    Compiler compiler(serverHelper.get(), backendConfig, targetConfig,
                      noiseModel, emulate);
    return compiler.runPassPipeline(executionContext, kernelName, modulePtr,
                                    args);
  }

  void completeLaunchKernel(const std::string &kernelName,
                            std::vector<cudaq::KernelExecution> &&codes) {
    auto executionContext = cudaq::getExecutionContext();

    // After performing lowerQuakeCode, check to see if we are simply drawing
    // the circuit. If so, perform the trace here and then return.
    if (executionContext->name == "tracer" && codes.size() == 1) {
      cudaq::ExecutionContext context("tracer");
      context.executionManager = cudaq::getDefaultExecutionManager();
      assert(codes[0].jit);
      cudaq::platform::with_execution_context(
          context, [&]() { codes[0].jit->run(kernelName); });
      executionContext->kernelTrace = std::move(context.kernelTrace);
      return;
    }

    if (executionContext->name == "resource-count") {
      cudaq::ExecutionContext context("resource-count");
      context.executionManager = cudaq::getDefaultExecutionManager();
      assert(codes.size() == 1 && codes[0].jit && codes[0].resourceCounts);
      nvqir::setResourceCounts(std::move(codes[0].resourceCounts.value()));
      cudaq::platform::with_execution_context(
          context, [&]() { codes[0].jit->run(kernelName); });
      return;
    }

    // Get the current execution context and number of shots
    std::size_t localShots = 1000;
    if (executionContext->shots != std::numeric_limits<std::size_t>::max() &&
        executionContext->shots != 0)
      localShots = executionContext->shots;

    executor->setShots(localShots);
    const bool isObserve =
        executionContext && executionContext->name == "observe";
    const bool isRun = executionContext && executionContext->name == "run";

    // If emulation requested, then just grab the function and invoke it with
    // the simulator
    cudaq::details::future future;
    if (emulate) {

      // TODO: This assert demonstrates that we are never expected to return a
      // future in emulation mode. We are launching a new thread just to wait
      // for its execution to finish below. We need to make this work without
      // the thread as the executionContext is crossing the thread boundary
      // which is not thread safe in the general case.
      assert(!executionContext->asyncExec);

      // Fetch the thread-specific seed outside and then pass it inside.
      std::size_t seed = cudaq::get_random_seed();

      // Launch the execution of the simulated jobs asynchronously
      future = cudaq::details::future(std::async(
          std::launch::async,
          [&, codes, localShots, kernelName, seed, isObserve, isRun,
           reorderIdx =
               executionContext->reorderIdx]() mutable -> cudaq::sample_result {
            std::vector<cudaq::ExecutionResult> results;

            // If seed is 0, then it has not been set.
            if (seed > 0)
              cudaq::set_random_seed(seed);

            const bool hasConditionals =
                executionContext
                    ? executionContext->hasConditionalsOnMeasureResults
                    : false;

            if (hasConditionals && isObserve)
              throw std::runtime_error("error: spin_ops not yet supported with "
                                       "kernels containing conditionals");
            if (isRun) {
              // Validate the execution logic: cudaq::run kernels should only
              // generate one JIT'ed kernel.
              assert(codes.size() == 1 && codes[0].jit);
              executor->setShots(1); // run one shot at a time

              // If this is executed via cudaq::run, then you have to run the
              // code localShots times
              for (std::size_t shot = 0; shot < localShots; shot++)
                codes[0].jit->run(kernelName);

              // Get QIR output log
              const auto qirOutputLog = nvqir::getQirOutputLog();
              executionContext->invocationResultBuffer.assign(
                  qirOutputLog.begin(), qirOutputLog.end());

            } else {
              // Otherwise, this is a non-adaptive sampling or observe.
              // We run the kernel(s) (multiple kernels if this is a multi-term
              // observe) one time each.
              for (std::size_t i = 0; i < codes.size(); i++) {
                cudaq::ExecutionContext context("sample", localShots);
                context.reorderIdx = reorderIdx;
                context.executionManager = cudaq::getDefaultExecutionManager();
                context.kernelName = kernelName;
                context.warnedNamedMeasurements =
                    executionContext ? executionContext->warnedNamedMeasurements
                                     : false;
                assert(codes[i].jit);
                cudaq::platform::with_execution_context(
                    context, [&]() { codes[i].jit->run(kernelName); });

                if (isObserve) {
                  // Use the code name instead of the global register.
                  results.emplace_back(context.result.to_map(), codes[i].name);
                  results.back().sequentialData =
                      context.result.sequential_data();
                } else {
                  // For each register, add the context results into result.
                  for (auto &regName : context.result.register_names()) {
                    results.emplace_back(context.result.to_map(regName),
                                         regName);
                    results.back().sequentialData =
                        context.result.sequential_data(regName);
                  }
                }
              }
            }
            return cudaq::sample_result(results);
          }));

    } else {
      // Execute the codes produced in quake lowering
      // Allow developer to disable remote sending (useful for debugging IR)
      if (getEnvBool("DISABLE_REMOTE_SEND", false))
        return;
      // Cannot be observe and run at the same time
      assert(!isObserve || !isRun);
      const cudaq::details::ExecutionContextType execType =
          isRun       ? cudaq::details::ExecutionContextType::run
          : isObserve ? cudaq::details::ExecutionContextType::observe
                      : cudaq::details::ExecutionContextType::sample;

      future = executor->execute(codes, execType,
                                 &executionContext->invocationResultBuffer);
    }

    // Keep this asynchronous if requested
    if (executionContext->asyncExec) {
      executionContext->futureResult = future;
      return;
    }

    // Otherwise make this synchronous
    executionContext->result = future.get();
  }
};

} // namespace cudaq
