/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/quantum_platform.h"
#include "algorithms/policies.h"
#include "common/CompiledModule.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/PluginUtils.h"
#include "common/RuntimeTarget.h"
#include "cudaq/Target/TargetConfig.h"
#include "cudaq/algorithms/policy_dispatch.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/runtime/logger/logger.h"
#include <exception>
#include <string>

using namespace cudaq_internal::compiler;

CUDAQ_INSTANTIATE_REGISTRY(cudaq::QPU::RegistryType)

// Bridge so the Python extension can register QPU subtypes (e.g. RemoteRESTQPU)
// into this DSO's registry. Same pattern as cudaq_add_module_launcher_node.
extern "C" void cudaq_add_qpu_node(void *node_ptr) {
  using Node = cudaq::Registry<cudaq::QPU>::node;
  cudaq::Registry<cudaq::QPU>::add_node(static_cast<Node *>(node_ptr));
}

namespace cudaq {

// These functions are defined elsewhere, but
// we are going to use them here.
std::string get_quake(const std::string &);

static quantum_platform *platform;
static constexpr std::string_view GetQuantumPlatformSymbol =
    "getQuantumPlatform";

static void (*platformInitCallback)() = nullptr;

extern "C" void setQuantumPlatformInitCallback(void (*callback)()) {
  platformInitCallback = callback;
}

void setQuantumPlatformInternal(quantum_platform *p) {
  info("external caller setting the platform.");
  platform = p;
}

/// @brief Get the provided platform plugin
/// @return
quantum_platform *getQuantumPlatformInternal() {
  if (platform)
    return platform;

  if (platformInitCallback) {
    auto callback = platformInitCallback;
    platformInitCallback = nullptr;
    callback();
    if (platform)
      return platform;
  }

  platform =
      getUniquePluginInstance<quantum_platform>(GetQuantumPlatformSymbol);
  return platform;
}

void quantum_platform::disableRuntimeEndpointOverride(std::size_t qpuId,
                                                      std::string what) const {
  if (hasRuntimeEndpointOverride(qpuId)) {
    throw std::runtime_error(
        what + " is not supported when manually setting a runtime endpoint.");
  }
}

bool quantum_platform::hasRuntimeEndpointOverride(std::size_t qpuId) const {
  return qpuId < platformQPUs.size() && !platformQPUs[qpuId];
}

void quantum_platform::set_noise(const noise_model *model, std::size_t qpu_id) {
  validateQpuId(qpu_id);
  disableRuntimeEndpointOverride(qpu_id, "Using noise models");
  auto &platformQPU = platformQPUs[qpu_id];
  platformQPU->setNoiseModel(model);
}

const noise_model *quantum_platform::get_noise(std::size_t qpu_id) {
  ExecutionContext *executionContext = getExecutionContext();
  if (executionContext != nullptr)
    return executionContext->noiseModel;

  if (hasRuntimeEndpointOverride(qpu_id)) {
    CUDAQ_WARN(
        "quantum_platform::get_noise is currently not supported for custom "
        "runtime endpoints");
    return nullptr;
  }

  validateQpuId(qpu_id);
  auto &platformQPU = platformQPUs[qpu_id];
  return platformQPU->getNoiseModel();
}

void quantum_platform::reset_noise(std::size_t qpu_id) {
  disableRuntimeEndpointOverride(qpu_id, "Using noise models");
  validateQpuId(qpu_id);
  set_noise(nullptr, qpu_id);
}

static cudaq::CompileTarget
getDefaultPythonCompileTargetImpl(quantum_platform *platform = nullptr) {
  if (!platform)
    platform = getQuantumPlatformInternal();

  const bool enablePythonCodegenDump =
      cudaq::getEnvBool("CUDAQ_PYTHON_CODEGEN_DUMP", false);
  if (enablePythonCodegenDump) {
    CUDAQ_WARN("CUDAQ_PYTHON_CODEGEN_DUMP is no longer supported. Use "
               "CUDAQ_MLIR_PRINT_EACH_PASS=argsynth instead.");
  }
  cudaq::CompileTarget ct;
  auto *rt = platform->get_runtime_target();
  if (!rt) {
    ct.pipelineConfig.skipTargetLoweringPipeline = true;
  } else {
    ct = cudaq::CompileTarget(rt->config, rt->runtimeConfig,
                              platform->is_emulated());
  }

  bool isLocalSimulator = !(platform->is_remote() || platform->is_emulated());

  ct.fullySpecialize = !isLocalSimulator;
  ct.isLocalSimulator = isLocalSimulator;
  ct.supportDeviceCalls = true;
  ct.argumentSynthChangeSemantics = false;
  ct.pipelineConfig.codegenTranslation = "qir:";
  ct.emitJit = true;
  return ct;
}

cudaq::CompileTarget getDefaultCompileTarget(const sample_policy &) {
  auto ct = getDefaultPythonCompileTargetImpl();
  ct.overrideAOTCompilation = false;
  return ct;
}
cudaq::CompileTarget getDefaultCompileTarget(const observe_policy &) {
  auto ct = getDefaultPythonCompileTargetImpl();
  ct.overrideAOTCompilation = false;
  return ct;
}
cudaq::CompileTarget getDefaultCompileTarget(const run_policy &) {
  auto ct = getDefaultPythonCompileTargetImpl();
  ct.overrideAOTCompilation = false;
  return ct;
}
cudaq::CompileTarget getDefaultCompileTarget(const dem_policy &) {
  auto ct = getDefaultPythonCompileTargetImpl();
  ct.overrideAOTCompilation = false;
  ct.emitJit = true;
  ct.emitTargetCode = false;
  ct.pipelineConfig.skipTargetLoweringPipeline = true;
  return ct;
}
cudaq::CompileTarget getDefaultCompileTarget(const other_policies &,
                                             ExecutionContext *context) {
  auto ct = getDefaultPythonCompileTargetImpl();
  ct.overrideAOTCompilation = false;
  ct.emitResourceCounts = context && context->name == "resource-count";
  return ct;
}

std::future<sample_result>
quantum_platform::enqueueAsyncTask(const std::size_t qpu_id,
                                   KernelExecutionTask &task) {
  std::promise<sample_result> promise;
  auto f = promise.get_future();
  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), t = task]() mutable {
        try {
          p.set_value(t());
        } catch (...) {
          p.set_exception(std::current_exception());
        }
      });

  platformQPUs[qpu_id]->enqueue(wrapped);
  return f;
}

void quantum_platform::enqueueAsyncTask(const std::size_t qpu_id,
                                        std::function<void()> &f) {
  platformQPUs[qpu_id]->enqueue(f);
}

void quantum_platform::validateQpuId(std::size_t qpuId,
                                     bool acceptRuntimeEndpoints) const {
  if (platformQPUs.empty())
    throw std::runtime_error("No QPUs are available for this target.");
  if (qpuId >= platformQPUs.size() &&
      (!acceptRuntimeEndpoints || qpuId >= runtimeEndpoints.size())) {
    std::size_t numQpus = platformQPUs.size();
    if (acceptRuntimeEndpoints)
      numQpus = std::max(numQpus, runtimeEndpoints.size());
    throw std::invalid_argument("Invalid QPU ID: " + std::to_string(qpuId) +
                                ". Number of QPUs: " + std::to_string(numQpus));
  }
}

// [remove at]: runtime refactor release
// Deprecated: Use with_execution_context instead.
void quantum_platform::set_exec_ctx(ExecutionContext *ctx) {
  configureExecutionContext(*ctx);
  detail::setExecutionContext(ctx);
  beginExecution();
}

// [remove at]: runtime refactor release
// Deprecated: Use with_execution_context instead.
void quantum_platform::reset_exec_ctx() {
  auto *ctx = getExecutionContext();
  if (ctx == nullptr)
    return;

  detail::try_finally([this, ctx] { finalizeExecutionContext(*ctx); },
                      [this] {
                        endExecution();
                        detail::resetExecutionContext();
                      });
}

// Specify the execution context for this platform.
// This delegates to the targeted QPU
void quantum_platform::configureExecutionContext(ExecutionContext &ctx) const {
  std::size_t qid = ctx.qpuId;
  disableRuntimeEndpointOverride(qid, "Policy '" + ctx.name + "'");
  validateQpuId(qid);
  auto &platformQPU = platformQPUs[qid];
  platformQPU->configureExecutionContext(ctx);
}

void quantum_platform::beginExecution() {
  auto qid = cudaq::getCurrentQpuId();
  disableRuntimeEndpointOverride(qid, "Unsupported policy");
  auto &platformQPU = platformQPUs[qid];

  platformQPU->beginExecution();
}

void quantum_platform::endExecution() {
  auto qid = cudaq::getCurrentQpuId();
  disableRuntimeEndpointOverride(qid, "Unsupported policy");
  auto &platformQPU = platformQPUs[qid];

  platformQPU->endExecution();
}

/// Reset the execution context for this platform.
void quantum_platform::finalizeExecutionContext(ExecutionContext &ctx) const {
  std::size_t qid = ctx.qpuId;
  disableRuntimeEndpointOverride(qid, "Unsupported policy");
  auto &platformQPU = platformQPUs[qid];
  platformQPU->finalizeExecutionContext(ctx);
}

std::optional<QubitConnectivity> quantum_platform::connectivity() {
  return platformQPUs.front()->getConnectivity();
}

bool quantum_platform::is_simulator(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  disableRuntimeEndpointOverride(qpu_id, "is_simulator");
  return platformQPUs[qpu_id]->isSimulator();
}

bool quantum_platform::is_remote(std::size_t qpu_id) const {
  if (hasRuntimeEndpointOverride(qpu_id)) {
    CUDAQ_WARN(
        "quantum_platform::is_remote is currently not supported for custom "
        "runtime endpoints");
    return false;
  }
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->isRemote();
}

bool quantum_platform::is_emulated(std::size_t qpu_id) const {
  if (hasRuntimeEndpointOverride(qpu_id)) {
    CUDAQ_WARN("quantum_platform::is_emulated is currently not supported for "
               "custom runtime endpoints");
    return false;
  }
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->isEmulated();
}

std::size_t quantum_platform::get_num_qubits(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  disableRuntimeEndpointOverride(qpu_id, "get_num_qubits");
  return platformQPUs[qpu_id]->getNumQubits();
}

bool quantum_platform::supports_explicit_measurements(
    std::size_t qpu_id) const {
  if (hasRuntimeEndpointOverride(qpu_id)) {
    CUDAQ_WARN("quantum_platform::supports_explicit_measurements is currently "
               "not supported for custom runtime endpoints");
    return false;
  }
  validateQpuId(qpu_id);
  return platformQPUs[qpu_id]->supportsExplicitMeasurements();
}

void quantum_platform::launchVQE(const std::string kernelName,
                                 const void *kernelArgs, gradient *gradient,
                                 const spin_op &H, optimizer &optimizer,
                                 const int n_params, const std::size_t shots,
                                 std::size_t qpu_id) {
  validateQpuId(qpu_id);
  disableRuntimeEndpointOverride(qpu_id, "Policy VQE");
  auto &qpu = platformQPUs[qpu_id];
  qpu->launchVQE(kernelName, kernelArgs, gradient, H, optimizer, n_params,
                 shots);
}

RemoteCapabilities
quantum_platform::get_remote_capabilities(std::size_t qpu_id) const {
  validateQpuId(qpu_id);
  disableRuntimeEndpointOverride(qpu_id, "get_remote_capabilities");
  return platformQPUs[qpu_id]->getRemoteCapabilities();
}

KernelThunkResultType
quantum_platform::unifiedLaunchModule(const AnyModule &module, KernelArgs args,
                                      std::size_t qpu_id) {
  validateQpuId(qpu_id);
  disableRuntimeEndpointOverride(qpu_id, "Unsupported policy");
  auto &qpu = platformQPUs[qpu_id];
  return qpu->unifiedLaunchModule(module, args);
}

void quantum_platform::ensureRuntimeEndpointExists(std::size_t qpuId,
                                                   bool allowNullopt) {
  validateQpuId(qpuId, /*acceptRuntimeEndpoints=*/true);
  std::scoped_lock lock(runtimeEndpointsMutex);
  if (runtimeEndpoints.size() <= qpuId)
    runtimeEndpoints.resize(platformQPUs.size());
  if (!allowNullopt && !runtimeEndpoints[qpuId].has_value()) {
    runtimeEndpoints[qpuId] = RuntimeEndpoint::wrapQPU(*platformQPUs[qpuId]);
  }
}

void quantum_platform::resetRuntimeEndpoints() {
  std::scoped_lock lock(runtimeEndpointsMutex);
  runtimeEndpoints.clear();
}

QPU &quantum_platform::addQPU(std::unique_ptr<QPU> qpu) {
  if (!qpu)
    throw std::invalid_argument("Cannot add a null QPU to the platform.");

  platformQPUs.emplace_back(std::move(qpu));
  return *platformQPUs.back();
}

void quantum_platform::clearQPUs() {
  resetRuntimeEndpoints();
  platformQPUs.clear();
}

QPU &quantum_platform::getQPU(std::size_t qpuId) {
  validateQpuId(qpuId);
  disableRuntimeEndpointOverride(qpuId, "Accessing the QPU");
  return *platformQPUs[qpuId];
}

void quantum_platform::setCompileTarget(std::optional<CompileTarget> target) {
  compileTarget = std::move(target);
}

RuntimeEndpoint &quantum_platform::getRuntimeEndpoint(std::size_t qpuId) {
  ensureRuntimeEndpointExists(qpuId);
  return *runtimeEndpoints[qpuId];
}

void quantum_platform::setRuntimeEndpoint(RuntimeEndpoint endpoint,
                                          std::size_t qpuId) {
  ensureRuntimeEndpointExists(qpuId, /*allowNullopt=*/true);

  if (!compileTarget.has_value()) {
    CUDAQ_WARN("Overriding compile target with default (local simulator)");
    compileTarget = getDefaultPythonCompileTargetImpl(this);
  }

  std::scoped_lock lock(runtimeEndpointsMutex);
  runtimeEndpoints[qpuId] = std::move(endpoint);
  platformQPUs[qpuId] = nullptr;
}

void quantum_platform::onRandomSeedSet(std::size_t seed) {
  // Send on the notification to all QPUs.
  for (auto &qpu : platformQPUs) {
    if (!qpu)
      continue;
    qpu->onRandomSeedSet(seed);
  }
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

bool quantum_platform::is_library_mode() const {
  if (libraryModeOverride > 0)
    return true;
  const auto *rt = get_runtime_target();
  if (!rt || !rt->config.BackendConfig)
    return false;
  auto &bc = rt->config.BackendConfig.value();
  return !bc.LibraryModeExecutionManager.empty();
}
} // namespace cudaq

cudaq::KernelThunkResultType
cudaq::altLaunchKernel(const char *kernelName,
                       cudaq::KernelThunkType kernelFunc, void *kernelArgs,
                       std::uint64_t argsSize, std::uint64_t resultOffset) {
  ScopedTraceWithContext("altLaunchKernel", kernelName, argsSize);
  auto &platform = *getQuantumPlatformInternal();
  std::string kernName = kernelName;
  KernelArgs args{KernelArgs::PackedArgs{kernelArgs, argsSize, resultOffset}};
  SourceModule src{kernName, kernelFunc};
  auto ctx = cudaq::getExecutionContext();
  if (ctx && ctx->executeKernelApi) {
    ctx->executeKernelApi(src, args);
    return {};
  }
  std::size_t qpu_id = cudaq::getCurrentQpuId();
  return platform.unifiedLaunchModule(src, args, qpu_id);
}

cudaq::KernelThunkResultType
cudaq::streamlinedLaunchKernel(const char *kernelName,
                               const std::vector<void *> &rawArgs) {
  std::size_t argsSize = rawArgs.size();
  ScopedTraceWithContext("streamlinedLaunchKernel", kernelName, argsSize);
  std::string kernName = kernelName;
  KernelArgs args{rawArgs};
  SourceModule src{kernName};
  auto ctx = cudaq::getExecutionContext();
  if (ctx && ctx->executeKernelApi) {
    ctx->executeKernelApi(src, args);
    return {};
  }
  auto &platform = *getQuantumPlatformInternal();
  std::size_t qpu_id = cudaq::getCurrentQpuId();
  [[maybe_unused]] auto r = platform.unifiedLaunchModule(src, args, qpu_id);
  // NB: The streamlined launch will never return results. Use alt or hybrid if
  // the kernel returns results.
  return {};
}

cudaq::KernelThunkResultType
cudaq::streamlinedLaunchModule(const CompiledModule &compiled,
                               const std::vector<void *> &rawArgs) {
  ScopedTraceWithContext("streamlinedLaunchModule", compiled.getName(),
                         rawArgs.size());

  auto ctx = cudaq::getExecutionContext();
  if (ctx && ctx->executeKernelApi) {
    ctx->executeKernelApi(compiled, {rawArgs});
    return {};
  }
  auto &platform = *getQuantumPlatformInternal();
  std::size_t qpu_id = getCurrentQpuId();
  return platform.unifiedLaunchModule(compiled, {rawArgs}, qpu_id);
}

cudaq::KernelThunkResultType
cudaq::hybridLaunchKernel(const char *kernelName, cudaq::KernelThunkType kernel,
                          void *args, std::uint64_t argsSize,
                          std::uint64_t resultOffset,
                          const std::vector<void *> &rawArgs) {
  ScopedTraceWithContext("hybridLaunchKernel", kernelName);
  auto &platform = *getQuantumPlatformInternal();
  const std::string kernName = kernelName;
  std::size_t qpu_id = cudaq::getCurrentQpuId();
  SourceModule src{kernName, kernel};

  KernelArgs kargs = platform.is_remote(qpu_id)
                         ? KernelArgs{rawArgs}
                         : KernelArgs{{args, argsSize, resultOffset}, rawArgs};

  auto ctx = cudaq::getExecutionContext();
  if (ctx && ctx->executeKernelApi) {
    ctx->executeKernelApi(src, kargs);
    return {};
  }

  if (platform.is_remote(qpu_id)) {
    // This path should never call a kernel that returns results.
    [[maybe_unused]] auto r = platform.unifiedLaunchModule(src, kargs, qpu_id);
    return {};
  }
  return platform.unifiedLaunchModule(src, kargs, qpu_id);
}
