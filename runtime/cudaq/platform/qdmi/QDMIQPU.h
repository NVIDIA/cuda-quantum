/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"
#include "cudaq/platform/qpu.h"

#include <map>
#include <memory>
#include <optional>
#include <string>

namespace cudaq {

struct QDMIPlatformDevice;

class QDMIQPU : public QPU {
public:
  QDMIQPU();
  QDMIQPU(std::shared_ptr<QDMIPlatformDevice> device,
          config::TargetConfig targetConfig,
          std::map<std::string, std::string> backendConfig);
  ~QDMIQPU() override;

  void enqueue(QuantumTask &task) override;
  bool isSimulator() override;
  bool supportsExplicitMeasurements() override { return false; }
  void setShots(int shots) override;
  void clearShots() override;
  bool isRemote() override;
  bool isEmulated() override;
  void setNoiseModel(const noise_model *model) override;
  void configureExecutionContext(ExecutionContext &context) const override;
  void finalizeExecutionContext(ExecutionContext &context) const override;
  void beginExecution() override;
  void endExecution() override;
  void setTargetBackend(const std::string &backend) override;

  std::unique_ptr<CompileTarget>
  getCompileTarget(const sample_policy &policy) override;
  std::unique_ptr<CompileTarget>
  getCompileTarget(const observe_policy &policy) override;
  std::unique_ptr<CompileTarget>
  getCompileTarget(const other_policies &policy,
                   ExecutionContext *context) override;

  sample_result launchKernel(const sample_policy &policy,
                             const CompiledModule &module,
                             KernelArgs args) override;
  async_sample_result launchKernel(const async_sample_policy &policy,
                                   const CompiledModule &module,
                                   KernelArgs args) override;
  observe_result launchKernel(const observe_policy &policy,
                              const CompiledModule &module,
                              KernelArgs args) override;
  async_observe_result launchKernel(const async_observe_policy &policy,
                                    const CompiledModule &module,
                                    KernelArgs args) override;

private:
  void configure(std::shared_ptr<QDMIPlatformDevice> device,
                 config::TargetConfig targetConfig,
                 std::map<std::string, std::string> backendConfig);

  std::optional<int> nShots;
  std::shared_ptr<QDMIPlatformDevice> platformDevice;
  std::map<std::string, std::string> backendConfig;
  config::TargetConfig targetConfig;
};

} // namespace cudaq
