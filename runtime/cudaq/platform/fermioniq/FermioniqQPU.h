/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"
#include <optional>

namespace cudaq {

/// @brief The `FermioniqBaseQPU` is a QPU that allows users to
// submit kernels to the Fermioniq simulator.
class FermioniqQPU : public BaseRemoteRESTQPU {
public:
  ~FermioniqQPU() override;

  virtual bool isRemote() override { return true; }

  /// @brief Return true if locally emulating a remote QPU
  virtual bool isEmulated() override { return false; }

  /// @brief Set the noise model, only allow this for
  /// emulation.
  virtual void setNoiseModel(const cudaq::noise_model *model) override {
    if (model) {
      throw std::runtime_error("Noise modeling is not allowed on this backend");
    }
  }

  /// Reset the execution context
  virtual void
  finalizeExecutionContext(ExecutionContext &context) const override {
    // set the pre-computed expectation value.
    if (context.name == "observe") {
      auto expectation = context.result.expectation(GlobalRegisterName);
      context.expectationValue = expectation;
    }
  }

  KernelThunkResultType unifiedLaunchModule(const AnyModule &module,
                                            KernelArgs args) override;

  using BaseRemoteRESTQPU::getCompileTarget;
  std::unique_ptr<CompileTarget>
  getCompileTarget(ExecutionContext *ctx) override {
    auto target = BaseRemoteRESTQPU::getCompileTarget(ctx);
    // This target handles observable evaluation server-side.
    // We don't want to split up the circuit into several ansatz
    // sub circuit.
    if (ctx->name == "observe") {
      target->pauliTermSplitObservable = std::nullopt;
    }
    return target;
  }

  sample_result launchKernel(sample_policy &policy, const AnyModule &module,
                             KernelArgs args) override;

  async_sample_result launchKernel(async_sample_policy &policy,
                                   const AnyModule &module,
                                   KernelArgs args) override;
};

} // namespace cudaq
