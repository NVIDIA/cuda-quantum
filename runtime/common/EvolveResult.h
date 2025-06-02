/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ObserveResult.h"
#include "cudaq/operators.h"
#include "cudaq/qis/state.h"
#include <memory>
#include <optional>

namespace cudaq {

/// @brief The evolve_result encapsulates all data generated from a
/// cudaq"::"evolve call. This includes information about the state
/// and any computed expectation values during and after evolution,
/// depending on the arguments passed to the call.
class evolve_result {
public:
  // The state of the system. If only final state is retained, this vector will
  // have exactly one element.
  std::optional<std::vector<state>> states = std::nullopt;

  // The computed expectation values. If only final expectation values are
  // retained, this vector will have exactly one element.
  std::optional<std::vector<std::vector<observe_result>>> expectation_values =
      std::nullopt;

  // The result of sampling of an analog Hamiltonian simulation on a QPU
  std::optional<sample_result> sampling_result = std::nullopt;

  // Construct from single final state.
  evolve_result(state state)
      : states(std::make_optional<std::vector<cudaq::state>>(
            std::vector<cudaq::state>{std::move(state)})) {}

  // Construct from single final observe result.
  evolve_result(state state, const std::vector<observe_result> &expectations)
      : states(std::make_optional<std::vector<cudaq::state>>(
            std::vector<cudaq::state>{std::move(state)})),
        expectation_values(
            std::make_optional<std::vector<std::vector<observe_result>>>(
                std::vector<std::vector<observe_result>>{expectations})) {}

  evolve_result(state state, const std::vector<double> &expectations)
      : states(std::make_optional<std::vector<cudaq::state>>(
            std::vector<cudaq::state>{std::move(state)})) {
    std::vector<observe_result> result;
    const spin_op emptyOp = spin_op::empty();
    for (auto e : expectations) {
      result.push_back(observe_result(e, emptyOp));
    }

    expectation_values =
        std::make_optional<std::vector<std::vector<observe_result>>>(
            std::vector<std::vector<observe_result>>{result});
  }

  // Construct from system states.
  evolve_result(const std::vector<state> &states)
      : states(std::make_optional<std::vector<state>>(states)) {}

  // Construct from intermediate system states and observe results.
  evolve_result(const std::vector<state> &states,
                const std::vector<std::vector<observe_result>> &expectations)
      : states(std::make_optional<std::vector<state>>(states)),
        expectation_values(
            std::make_optional<std::vector<std::vector<observe_result>>>(
                expectations)) {}

  evolve_result(const std::vector<state> &states,
                const std::vector<std::vector<double>> &expectations)
      : states(std::make_optional<std::vector<state>>(states)) {
    std::vector<std::vector<observe_result>> result;
    const spin_op emptyOp = spin_op::empty();
    for (const auto &vec : expectations) {
      std::vector<observe_result> subResult;
      for (auto e : vec) {
        subResult.push_back(observe_result(e, emptyOp));
      }
      result.push_back(subResult);
    }
    expectation_values = result;
  }

  // Construct result with a final state and intermediate expectations
  evolve_result(state finalState,
                const std::vector<std::vector<double>> &expectations)
      : states(std::make_optional<std::vector<cudaq::state>>(
            std::vector<cudaq::state>{std::move(finalState)})) {
    std::vector<std::vector<observe_result>> result;
    const spin_op emptyOp = spin_op::empty();
    for (const auto &vec : expectations) {
      std::vector<observe_result> subResult;
      for (auto e : vec) {
        subResult.push_back(observe_result(e, emptyOp));
      }
      result.push_back(subResult);
    }
    expectation_values = result;
  }
  evolve_result(const sample_result &sr) : sampling_result(sr) {}
};
} // namespace cudaq
