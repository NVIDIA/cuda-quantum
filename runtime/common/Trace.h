/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/execution_manager.h"
#include <string>
#include <vector>

namespace cudaq {

/// @brief A trace is a circuit representation of the executed computation, as
/// seen by the execution manager. (Here, a circuit is represented as a list
/// of instructions on qudits). Since the execution manager cannot "see" control
/// flow, the trace of a kernel with control flow represents a single execution
/// path, and thus two calls to the same kernel might produce traces.
class Trace {
public:
  struct Instruction {
    std::string name;
    std::vector<double> params;
    std::vector<QuditInfo> controls;
    std::vector<QuditInfo> targets;

    Instruction(std::string_view name, std::vector<double> params,
                std::vector<QuditInfo> controls, std::vector<QuditInfo> targets)
        : name(name), params(params), controls(controls), targets(targets) {}
  };

  void appendInstruction(std::string_view name, std::vector<double> params,
                         std::vector<QuditInfo> controls,
                         std::vector<QuditInfo> targets);

  auto getNumQudits() const { return numQudits; }

  auto begin() const { return instructions.begin(); }

  auto end() const { return instructions.end(); }

private:
  std::size_t numQudits = 0;
  std::vector<Instruction> instructions;
};

} // namespace cudaq
