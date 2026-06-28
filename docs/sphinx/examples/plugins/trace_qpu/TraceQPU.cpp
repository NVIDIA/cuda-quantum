/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/SampleResult.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include <fstream>
#include <map>

namespace cudaq {

class TraceQPU : public QPU {
  std::map<std::string, std::string> backendConfig;
  std::string traceFile;

  void appendTrace(const std::string &kernelName) const {
    if (traceFile.empty())
      return;

    std::ofstream out(traceFile, std::ios::app);
    if (!out)
      throw std::runtime_error("trace_qpu could not open trace file: " +
                               traceFile);

    auto *context = getExecutionContext();
    out << "kernel=" << kernelName;
    if (context)
      out << " context=" << context->name << " shots=" << context->shots;
    out << '\n';
  }

public:
  TraceQPU() { numQubits = 64; }

  void enqueue(QuantumTask &task) override { execution_queue->enqueue(task); }

  bool isSimulator() override { return true; }

  void setTargetBackend(const std::string &backend) override {
    auto parts = cudaq::split(backend, ';');
    for (std::size_t i = 1; i + 1 < parts.size(); i += 2) {
      auto value = parts[i + 1];
      if (value.starts_with("base64_"))
        value = detail::decodeBase64(value.substr(7));
      backendConfig[parts[i]] = value;
    }

    if (auto iter = backendConfig.find("trace_file");
        iter != backendConfig.end())
      traceFile = iter->second;
  }

  KernelThunkResultType unifiedLaunchModule(const AnyModule &module,
                                            KernelArgs args) override {
    std::string kernelName = "unknown";
    if (std::holds_alternative<SourceModule>(module))
      kernelName = std::get<SourceModule>(module).getName();
    else
      kernelName = std::get<CompiledModule>(module).getName();

    CUDAQ_INFO("trace_qpu launch: {}", kernelName);
    appendTrace(kernelName);

    if (auto *context = getExecutionContext()) {
      const auto shots = context->shots == 0 ? std::size_t{1} : context->shots;
      CountsDictionary counts{{"0", shots}};
      context->result = sample_result(ExecutionResult(counts));
      if (context->name == "observe")
        context->expectationValue = 1.0;
    }

    return {nullptr, 0};
  }

  sample_result launchKernel(const sample_policy &policy,
                             const CompiledModule &module,
                             KernelArgs args) override {
    unifiedLaunchModule(module, args);
    CountsDictionary counts{{"0", policy.options.shots}};
    return sample_result(ExecutionResult(counts));
  }

  observe_result launchKernel(const observe_policy &policy,
                              const CompiledModule &module,
                              KernelArgs args) override {
    unifiedLaunchModule(module, args);
    if (policy.options.shots > 0) {
      CountsDictionary counts{
          {"0", static_cast<std::size_t>(policy.options.shots)}};
      return observe_result(1.0, policy.spin,
                            sample_result(ExecutionResult(counts)));
    }
    return observe_result(1.0, policy.spin);
  }

  std::unique_ptr<CompileTarget>
  getCompileTarget(const other_policies &policy,
                   ExecutionContext *context) override {
    return getDefaultCompileTarget(policy, context);
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::TraceQPU, trace_qpu)
