/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"

#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/spin_op.h"
#include "cudaq/utils/cudaq_utils.h"
#include "qpp.h"
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <stack>

namespace {

class SimpleQuditExecutionManager : public cudaq::BasicExecutionManager {
private:
  qpp::ket state;

  std::unordered_map<std::string, std::function<void(const Instruction &)>>
      instructions;

  std::vector<cudaq::QuditInfo> sampleQudits;

protected:
  void allocateQudit(const cudaq::QuditInfo &q) override {
    if (state.size() == 0) {
      // qubit will give [1,0], qutrit will give [1,0,0]
      state = qpp::ket::Zero(q.levels);
      state(0) = 1.0;
      return;
    }

    qpp::ket zeroState = qpp::ket::Zero(q.levels);
    zeroState(0) = 1.0;
    state = qpp::kron(state, zeroState);
  }

  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    for (auto &q : qudits)
      allocateQudit(q);
  }

  void deallocateQudit(const cudaq::QuditInfo &q) override {}
  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {}

  void handleExecutionContextChanged() override {}

  void handleExecutionContextEnded() override {
    if (executionContext && executionContext->name == "sample") {
      std::vector<std::size_t> ids;
      for (auto &s : sampleQudits) {
        ids.push_back(s.id);
      }
      auto sampleResult =
          qpp::sample(1000, state, ids, sampleQudits.begin()->levels);

      for (auto [result, count] : sampleResult) {
        std::cout << fmt::format("Sample {} : {}", result, count) << "\n";
      }
    }
  }

  void executeInstruction(const Instruction &instruction) override {
    auto operation = instructions[std::get<0>(instruction)];
    operation(instruction);
  }

  int measureQudit(const cudaq::QuditInfo &q) override {
    if (executionContext && executionContext->name == "sample") {
      sampleQudits.push_back(q);
      return 0;
    }

    // If here, then we care about the result bit, so compute it.
    const auto measurement_tuple = qpp::measure(
        state, qpp::cmat::Identity(q.levels, q.levels), {q.id},
        /*qudit dimension=*/q.levels, /*destructive measmt=*/false);
    const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
    const auto &collapsed_state = post_meas_states[measurement_result];
    state = Eigen::Map<const qpp::ket>(collapsed_state.data(),
                                       collapsed_state.size());

    cudaq::info("Measured qubit {} -> {}", q.id, measurement_result);
    return measurement_result;
  }

  void measureSpinOp(const cudaq::spin_op &) override {}

public:
  SimpleQuditExecutionManager() {
    instructions.emplace("plusGate", [&](const Instruction &inst) {
      qpp::cmat u(3, 3);
      u << 0, 0, 1, 1, 0, 0, 0, 1, 0;
      auto &[gateName, params, controls, qudits, op] = inst;
      auto target = qudits[0];
      cudaq::info("Applying plusGate on {}<{}>", target.id, target.levels);
      state = qpp::applyCTRL(state, u, {}, {target.id}, target.levels);
    });
  }
  virtual ~SimpleQuditExecutionManager() = default;

  cudaq::SpinMeasureResult measure(cudaq::spin_op &op) override {
    return cudaq::SpinMeasureResult();
  }

  void resetQudit(const cudaq::QuditInfo &id) override {}
};

} // namespace

CUDAQ_REGISTER_EXECUTION_MANAGER(SimpleQuditExecutionManager)
