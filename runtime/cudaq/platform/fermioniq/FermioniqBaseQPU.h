/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"

namespace cudaq {

/// @brief The `FermioniqBaseQPU` is a QPU that allows users to
// submit kernels to the Fermioniq simulator.
class FermioniqBaseQPU : public BaseRemoteRESTQPU {
public:
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
  virtual void resetExecutionContext() override {
    // set the pre-computed expectation value.
    if (executionContext->name == "observe") {
      auto expectation =
          executionContext->result.expectation(GlobalRegisterName);

      executionContext->expectationValue = expectation;
    }
    executionContext = nullptr;
  }

  KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    CUDAQ_INFO("FermioniqBaseQPU launching kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), or cudaq::contrib::draw().");

    // When the user issues an observe call, we don't want to use the default
    // cuda-quantum behaviour that splits up the circuit into several ansatz
    // sub circuit.
    // So before calling lowerQuakeCode, we create a temporary "sample"
    // executionContext.
    // Once the codes are generated, we reset it.
    cudaq::ExecutionContext defaultContext("sample", 1);
    auto *originalContext = executionContext;
    if (executionContext->name == "observe")
      executionContext = &defaultContext;

    auto codes = rawArgs.empty() ? lowerQuakeCode(kernelName, args, {})
                                 : lowerQuakeCode(kernelName, nullptr, rawArgs);
    if (codes.size() != 1) {
      throw std::runtime_error("Provider only allows 1 circuit at a time.");
    }

    executionContext = originalContext;

    if (executionContext->name == "observe") {
      auto spin = executionContext->spin.value();
      auto user_data = nlohmann::json::object();
      auto obs = nlohmann::json::array();

      for (const auto &term : spin) {
        auto spin_op = nlohmann::json::object();

        auto terms = nlohmann::json::array();

        auto termStr = term.get_term_id();

        terms.push_back(termStr);

        auto coeff = term.evaluate_coefficient();
        auto coeff_str = cudaq_fmt::format("{}{}{}j", coeff.real(),
                                           coeff.imag() < 0.0 ? "-" : "+",
                                           std::fabs(coeff.imag()));

        terms.push_back(coeff_str);

        obs.push_back(terms);
      }

      user_data["observable"] = obs;

      codes[0].user_data = user_data;
    }

    completeLaunchKernel(kernelName, std::move(codes));

    return {};
  }

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    launchKernel(kernelName, nullptr, nullptr, 0, 0, rawArgs);
  }
};
} // namespace cudaq
