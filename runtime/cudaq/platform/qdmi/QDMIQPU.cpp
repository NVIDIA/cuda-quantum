/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QDMIQPU.h"
#include "QDMIPlatformDevice.h"

#include "common/ExecutionContext.h"
#include "common/Future.h"
#include "common/KernelExecution.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq/runtime/logger/logger.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace {
cudaq::CountsDictionary
toCountsDictionary(const std::map<std::string, std::size_t> &counts) {
  cudaq::CountsDictionary result;
  result.reserve(counts.size());
  for (const auto &[rawBits, count] : counts) {
    auto bits = rawBits;
    std::reverse(bits.begin(), bits.end());
    result[std::move(bits)] = count;
  }
  return result;
}

std::vector<std::string> toShotData(std::vector<std::string> shots) {
  for (auto &bits : shots)
    std::reverse(bits.begin(), bits.end());
  return shots;
}

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

template <typename ShotType>
std::size_t resolveShots(std::optional<int> configuredShots,
                         ShotType policyShots) {
  if (policyShots > 0)
    return policyShots;
  if (configuredShots && *configuredShots > 0)
    return static_cast<std::size_t>(*configuredShots);
  return 1000;
}

cudaq::sample_result
submitJobs(std::shared_ptr<cudaq::QDMIPlatformDevice> platformDevice,
           std::vector<cudaq::KernelExecution> codes,
           cudaq::detail::ExecutionContextType execType,
           std::size_t shotCount) {
  if (!platformDevice)
    throw std::runtime_error("QDMI QPU is not configured.");
  if (execType == cudaq::detail::ExecutionContextType::run)
    throw std::runtime_error("QDMI backend does not support cudaq::run.");

  cudaq::sample_result result;
  bool hasResult = false;

  for (const auto &code : codes) {
    auto job = platformDevice->fomacDevice.submitJob(
        code.code, platformDevice->programFormat, shotCount,
        platformDevice->jobCustom1, platformDevice->jobCustom2,
        platformDevice->jobCustom3, platformDevice->jobCustom4,
        platformDevice->jobCustom5);
    if (!job.wait())
      throw std::runtime_error("QDMI job timed out.");

    const auto status = job.check();
    if (status == QDMI_JOB_STATUS_FAILED)
      throw std::runtime_error("QDMI job failed.");
    if (status == QDMI_JOB_STATUS_CANCELED)
      throw std::runtime_error("QDMI job was canceled.");
    if (status != QDMI_JOB_STATUS_DONE)
      throw std::runtime_error("QDMI job did not complete.");

    const bool observe =
        execType == cudaq::detail::ExecutionContextType::observe;
    const auto registerName = observe ? code.name : cudaq::GlobalRegisterName;
    cudaq::ExecutionResult executionResult(toCountsDictionary(job.getCounts()),
                                           registerName);
    try {
      executionResult.sequentialData = toShotData(job.getShots());
    } catch (const std::exception &e) {
      CUDAQ_DBG("QDMI shot data is unavailable: {}", e.what());
    }

    cudaq::sample_result jobResult(std::move(executionResult));
    if (!code.mapping_reorder_idx.empty())
      jobResult.reorder(code.mapping_reorder_idx, registerName);
    if (hasResult) {
      result += jobResult;
    } else {
      result = std::move(jobResult);
      hasResult = true;
    }
  }

  return result;
}
} // namespace

namespace cudaq {

QDMIQPU::QDMIQPU() : QPU() {}

QDMIQPU::QDMIQPU(std::shared_ptr<QDMIPlatformDevice> device,
                 config::TargetConfig targetConfig,
                 std::map<std::string, std::string> backendConfig)
    : QPU() {
  configure(std::move(device), std::move(targetConfig),
            std::move(backendConfig));
}

QDMIQPU::~QDMIQPU() = default;

void QDMIQPU::configure(std::shared_ptr<QDMIPlatformDevice> device,
                        config::TargetConfig targetConfig,
                        std::map<std::string, std::string> backendConfig) {
  platformDevice = std::move(device);
  this->targetConfig = std::move(targetConfig);
  this->backendConfig = std::move(backendConfig);

  if (!platformDevice)
    return;

  numQubits = platformDevice->qubitCount;
  connectivity = platformDevice->connectivity;
}

void QDMIQPU::enqueue(QuantumTask &task) { execution_queue->enqueue(task); }

bool QDMIQPU::isSimulator() {
  // FoMaC/QDMI does not expose this kind.
  return false;
}

bool QDMIQPU::isRemote() {
  // FoMaC/QDMI does not expose this kind.
  return true;
}

bool QDMIQPU::isEmulated() {
  // QDMI does not expose CUDA-Q emulation.
  return false;
}

void QDMIQPU::setShots(int shots) { nShots = shots; }

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

void QDMIQPU::setTargetBackend(const std::string &) {
  throw std::runtime_error("QDMI QPUs are configured by the QDMI platform.");
}

std::unique_ptr<CompileTarget>
QDMIQPU::getCompileTarget(const sample_policy &) {
  auto target = std::make_unique<CompileTarget>(targetConfig, backendConfig,
                                                /*emulate=*/false);
  target->supportConditionalsOnMeasureResults = false;
  target->pipelineConfig.addMeasurements = true;
  target->storeReorderIdx = true;
  target->pipelineConfig.replaceStateWithKernel = true;
  target->overrideAOTCompilation = true;
  return target;
}

std::unique_ptr<CompileTarget>
QDMIQPU::getCompileTarget(const observe_policy &policy) {
  auto target = std::make_unique<CompileTarget>(targetConfig, backendConfig,
                                                /*emulate=*/false);
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
  const auto shotCount = resolveShots(nShots, policy.options.shots);
  return submitJobs(platformDevice, std::move(codes),
                    detail::ExecutionContextType::sample, shotCount);
}

observe_result QDMIQPU::launchKernel(const observe_policy &policy,
                                     const CompiledModule &module, KernelArgs) {
  auto codes = runCodegen(module, getCompileTarget(policy));
  const auto shotCount = resolveShots(nShots, policy.options.shots);
  return observeResultFromCounts(
      policy, submitJobs(platformDevice, std::move(codes),
                         detail::ExecutionContextType::observe, shotCount));
}

} // namespace cudaq
