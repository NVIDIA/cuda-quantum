/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
  void setNoiseModel(const cudaq::noise_model *model) override {
    throw std::runtime_error("Noise modeling is not allowed on this backend");
  }

  /// Reset the execution context
  void resetExecutionContext() override {
    // set the pre-computed expectation value.
    if (executionContext->name == "observe") {
      auto expectation =
          executionContext->result.expectation(GlobalRegisterName);

      executionContext->expectationValue = expectation;
    }
    executionContext = nullptr;
  }

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    cudaq::info("launching remote rest kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), or cudaq::draw().");

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

    auto codes = lowerQuakeCode(kernelName, nullptr, rawArgs);
    if (codes.size() != 1) {
      throw std::runtime_error("Provider only allows 1 circuit at a time.");
    }

    executionContext = originalContext;

    if (executionContext->name == "observe") {
      auto spin = executionContext->spin.value();
      auto user_data = nlohmann::json::object();
      auto obs = nlohmann::json::array();

      spin->for_each_term([&](spin_op &term) {
        auto spin_op = nlohmann::json::object();

        auto terms = nlohmann::json::array();

        auto termStr = term.to_string(false);

        terms.push_back(termStr);

        auto coeff = term.get_coefficient();
        auto coeff_str =
            fmt::format("{}{}{}j", coeff.real(), coeff.imag() < 0.0 ? "-" : "+",
                        std::fabs(coeff.imag()));

        terms.push_back(coeff_str);

        obs.push_back(terms);
      });

      user_data["observable"] = obs;

      codes[0].user_data = user_data;
    }

    completeLaunchKernel(kernelName, std::move(codes));
  }
};
} // namespace cudaq
