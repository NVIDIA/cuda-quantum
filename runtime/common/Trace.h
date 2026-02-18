/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/execution_manager.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {

struct QuditInfo;

/// @brief Instruction kind in the trace. Default is Gate for backwards
/// compatibility.
enum class TraceInstructionType {
  Gate,
  Noise,
};

/// @brief Name used in the trace for apply_noise (inline noise) instructions.
constexpr std::string_view TRACE_APPLY_NOISE_NAME = "apply_noise";

/// @brief A trace is a circuit representation of the executed computation, as
/// seen by the execution manager. (Here, a circuit is represented as a list
/// of instructions on qudits). Since the execution manager cannot "see" control
/// flow, the trace of a kernel with control flow represents a single execution
/// path, and thus two calls to the same kernel might produce traces.
///
/// Instructions are typed (Gate or Noise). Noise instructions represent
/// apply_noise; the channel is resolved via noise_model::get_channel(
/// noise_channel_key, `params`).
class Trace {
public:
  struct Instruction {
    TraceInstructionType type = TraceInstructionType::Gate;
    std::string name;
    std::vector<double> params;
    std::vector<QuditInfo> controls;
    std::vector<QuditInfo> targets;
    std::optional<std::intptr_t> noise_channel_key;

    Instruction(std::string_view name, std::vector<double> params,
                std::vector<QuditInfo> controls, std::vector<QuditInfo> targets,
                std::optional<std::intptr_t> noise_key = std::nullopt,
                TraceInstructionType type = TraceInstructionType::Gate)
        : type(type), name(name), params(params), controls(controls),
          targets(targets), noise_channel_key(noise_key) {}
  };

  void appendInstruction(std::string_view name, std::vector<double> params,
                         std::vector<QuditInfo> controls,
                         std::vector<QuditInfo> targets);

  /// @brief Append an apply_noise instruction (for PTSBE trace capture).
  void appendNoiseInstruction(std::intptr_t noise_channel_key,
                              std::vector<double> params,
                              std::vector<QuditInfo> controls,
                              std::vector<QuditInfo> targets);

  auto getNumQudits() const { return numQudits; }

  auto getNumInstructions() const { return instructions.size(); }

  auto begin() const { return instructions.begin(); }

  auto end() const { return instructions.end(); }

private:
  std::size_t numQudits = 0;
  std::vector<Instruction> instructions;
};

} // namespace cudaq
