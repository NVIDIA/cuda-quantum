/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QDMIQPU.h"
#include "QDMIServerHelper.h"

#include "common/ExecutionContext.h"
#include "common/Executor.h"
#include "common/ServerHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace {
cudaq::observe_result
observeResultFromCounts(const cudaq::observe_policy &policy,
                        cudaq::sample_result data) {
  double sum = 0.0;
  for (const auto &term : policy.spin) {
    if (term.is_identity())
      sum += term.evaluate_coefficient().real();
    else
      sum += data.expectation(term.get_term_id()) *
             term.evaluate_coefficient().real();
  }
  return cudaq::observe_result(sum, policy.spin, data);
}

std::vector<cudaq::KernelExecution>
runCodegen(const cudaq::CompiledModule &module,
           std::unique_ptr<cudaq::CompileTarget> target) {
  if (module.getMlirArtifacts().empty())
    throw std::runtime_error("QPU does not support launching a "
                             "CompiledModule without MLIR artifacts.");

  cudaq_internal::compiler::Compiler compiler(std::move(target));
  return compiler.emitKernelExecutions(module);
}

class QDMICompileTarget : public cudaq::CompileTarget {
public:
  QDMICompileTarget(cudaq::ServerHelper *serverHelper,
                    cudaq::config::TargetConfig targetConfig,
                    std::map<std::string, std::string> runtimeConfig)
      : CompileTarget(std::move(targetConfig), std::move(runtimeConfig),
                      /*emulate=*/false),
        serverHelper(serverHelper) {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
  }

  void updatePassPipeline(std::string &passPipeline) const override {
    serverHelper->updatePassPipeline(platformPath, passPipeline);
  }

private:
  cudaq::ServerHelper *serverHelper;
  std::filesystem::path platformPath;
};
} // namespace

namespace cudaq {

QDMIQPU::QDMIQPU() : QPU() {
  std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
  platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
}

QDMIQPU::~QDMIQPU() = default;

void QDMIQPU::enqueue(QuantumTask &task) { execution_queue->enqueue(task); }

void QDMIQPU::setShots(int shots) {
  nShots = shots;
  if (executor && shots > 0)
    executor->setShots(static_cast<std::size_t>(shots));
}

void QDMIQPU::clearShots() { nShots = std::nullopt; }

void QDMIQPU::setNoiseModel(const noise_model *model) {
  if (model)
    throw std::runtime_error(
        "Noise modeling is not supported by the QDMI backend.");
  noiseModel = nullptr;
}

void QDMIQPU::configureExecutionContext(ExecutionContext &context) const {
  CUDAQ_INFO("QDMI QPU preparing execution context for {}", context.name);
  if (context.executionManager)
    context.executionManager->configureExecutionContext(context);
}

void QDMIQPU::finalizeExecutionContext(ExecutionContext &context) const {
  if (context.executionManager)
    context.executionManager->finalizeExecutionContext(context);
}

void QDMIQPU::beginExecution() {
  auto *executionContext = getExecutionContext();
  if (executionContext && executionContext->executionManager)
    executionContext->executionManager->beginExecution();
}

void QDMIQPU::endExecution() {
  auto *executionContext = getExecutionContext();
  if (executionContext && executionContext->executionManager)
    executionContext->executionManager->endExecution();
}

void QDMIQPU::setTargetBackend(const std::string &backend) {
  CUDAQ_INFO("QDMI platform is targeting {}.", backend);

  backendConfig.clear();
  targetConfig = config::TargetConfig{};

  auto targetName = backend;
  if (targetName.find(";") != std::string::npos) {
    auto split = cudaq::split(targetName, ';');
    targetName = split[0];
    if ((split.size() - 1) % 2 != 0)
      throw std::runtime_error(
          "Backend config must be provided as key-value pairs.");

    for (std::size_t i = 1; i < split.size(); i += 2) {
      if (split[i + 1].starts_with("base64_")) {
        split[i + 1].erase(0, 7);
        backendConfig.insert({split[i], detail::decodeBase64(split[i + 1])});
      } else {
        backendConfig.insert({split[i], split[i + 1]});
      }
    }
  }

  if (auto iter = backendConfig.find("emulate");
      iter != backendConfig.end() && iter->second == "true")
    throw std::runtime_error(
        "QDMI backend does not support CUDA-Q emulation mode.");

  const auto configFilePath = platformPath / (targetName + ".yml");
  std::ifstream configFile(configFilePath.string());
  if (!configFile)
    throw std::runtime_error("Could not open QDMI target configuration.");

  const std::string configYmlContents(
      (std::istreambuf_iterator<char>(configFile)),
      std::istreambuf_iterator<char>());
  detail::parseTargetConfigYml(configYmlContents, targetConfig);

  qpuName = targetName;
  detail::initServerHelperAndExecutor(qpuName, backendConfig, targetConfig,
                                      serverHelper, executor);

  auto *helper = dynamic_cast<QDMIServerHelper *>(serverHelper.get());
  if (!helper)
    throw std::runtime_error("QDMI QPU requires QDMIServerHelper.");

  numQubits = helper->getQubitCount();
  connectivity = helper->getConnectivity();
}

std::unique_ptr<CompileTarget>
QDMIQPU::getCompileTarget(const sample_policy &policy) {
  auto target = std::make_unique<QDMICompileTarget>(
      serverHelper.get(), targetConfig, backendConfig);
  target->supportConditionalsOnMeasureResults = false;
  target->pipelineConfig.addMeasurements = true;
  target->storeReorderIdx = true;
  target->pipelineConfig.replaceStateWithKernel = true;
  target->overrideAOTCompilation = true;
  return target;
}

std::unique_ptr<CompileTarget>
QDMIQPU::getCompileTarget(const observe_policy &policy) {
  auto target = std::make_unique<QDMICompileTarget>(
      serverHelper.get(), targetConfig, backendConfig);
  target->supportConditionalsOnMeasureResults = false;
  target->overrideAOTCompilation = true;
  target->pauliTermSplitObservable = policy.spin;
  target->pipelineConfig.replaceStateWithKernel = true;
  return target;
}

std::unique_ptr<CompileTarget> QDMIQPU::getCompileTarget(const other_policies &,
                                                         ExecutionContext *) {
  throw std::runtime_error(
      "QDMI backend supports cudaq::sample() and cudaq::observe().");
}

sample_result QDMIQPU::launchKernel(const sample_policy &policy,
                                    const CompiledModule &module, KernelArgs) {
  auto codes = runCodegen(module, getCompileTarget(policy));

  std::size_t localShots = 1000;
  if (nShots && *nShots > 0)
    localShots = static_cast<std::size_t>(*nShots);
  if (policy.options.shots > 0)
    localShots = static_cast<std::size_t>(policy.options.shots);
  executor->setShots(localShots);

  auto *executionContext = cudaq::getExecutionContext();
  auto future = executor->execute(
      codes, detail::ExecutionContextType::sample,
      executionContext ? &executionContext->invocationResultBuffer : nullptr);
  return future.get();
}

async_sample_result QDMIQPU::launchKernel(const async_sample_policy &policy,
                                          const CompiledModule &module,
                                          KernelArgs) {
  auto codes = runCodegen(module, getCompileTarget(policy.inner));

  std::size_t localShots = 1000;
  if (nShots && *nShots > 0)
    localShots = static_cast<std::size_t>(*nShots);
  if (policy.inner.options.shots > 0)
    localShots = static_cast<std::size_t>(policy.inner.options.shots);
  executor->setShots(localShots);

  auto *executionContext = cudaq::getExecutionContext();
  auto future = executor->execute(
      codes, detail::ExecutionContextType::sample,
      executionContext ? &executionContext->invocationResultBuffer : nullptr);
  return async_sample_result(std::move(future));
}

observe_result QDMIQPU::launchKernel(const observe_policy &policy,
                                     const CompiledModule &module, KernelArgs) {
  auto codes = runCodegen(module, getCompileTarget(policy));

  std::size_t localShots = 1000;
  if (nShots && *nShots > 0)
    localShots = static_cast<std::size_t>(*nShots);
  if (policy.options.shots > 0)
    localShots = static_cast<std::size_t>(policy.options.shots);
  executor->setShots(localShots);

  auto *executionContext = cudaq::getExecutionContext();
  auto future = executor->execute(
      codes, detail::ExecutionContextType::observe,
      executionContext ? &executionContext->invocationResultBuffer : nullptr);
  return observeResultFromCounts(policy, future.get());
}

async_observe_result QDMIQPU::launchKernel(const async_observe_policy &policy,
                                           const CompiledModule &module,
                                           KernelArgs) {
  auto codes = runCodegen(module, getCompileTarget(policy.inner));

  std::size_t localShots = 1000;
  if (nShots && *nShots > 0)
    localShots = static_cast<std::size_t>(*nShots);
  if (policy.inner.options.shots > 0)
    localShots = static_cast<std::size_t>(policy.inner.options.shots);
  executor->setShots(localShots);

  auto *executionContext = cudaq::getExecutionContext();
  auto future = executor->execute(
      codes, detail::ExecutionContextType::observe,
      executionContext ? &executionContext->invocationResultBuffer : nullptr);
  return async_observe_result(std::move(future), &policy.inner.spin);
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::QDMIQPU, qdmi)
