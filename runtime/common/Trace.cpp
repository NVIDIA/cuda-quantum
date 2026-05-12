/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Trace.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>

void cudaq::Trace::appendInstruction(std::string_view name,
                                     std::vector<double> params,
                                     std::vector<QuditInfo> controls,
                                     std::vector<QuditInfo> targets) {
  assert(!targets.empty() && "An instruction must have at least one target");
  auto findMaxID = [](const std::vector<QuditInfo> &qudits) -> std::size_t {
    return std::max_element(qudits.cbegin(), qudits.cend(),
                            [](auto &a, auto &b) { return a.id < b.id; })
        ->id;
  };
  std::size_t maxID = findMaxID(targets);
  if (!controls.empty())
    maxID = std::max(maxID, findMaxID(controls));
  numQudits = std::max(numQudits, maxID + 1);
  instructions.emplace_back(name, params, controls, targets, std::nullopt,
                            TraceInstructionType::Gate);
}

void cudaq::Trace::appendNoiseInstruction(std::intptr_t noise_channel_key,
                                          std::string_view channel_name,
                                          std::vector<double> params,
                                          std::vector<QuditInfo> controls,
                                          std::vector<QuditInfo> targets) {
  if (targets.empty())
    throw std::invalid_argument(
        "appendNoiseInstruction: noise channel must have at least one target");
  auto findMaxID = [](const std::vector<QuditInfo> &qudits) -> std::size_t {
    return std::max_element(qudits.cbegin(), qudits.cend(),
                            [](auto &a, auto &b) { return a.id < b.id; })
        ->id;
  };
  std::size_t maxID = findMaxID(targets);
  if (!controls.empty())
    maxID = std::max(maxID, findMaxID(controls));
  numQudits = std::max(numQudits, maxID + 1);
  instructions.emplace_back(channel_name, params, std::move(controls),
                            std::move(targets), noise_channel_key,
                            TraceInstructionType::Noise);
}

void cudaq::Trace::appendMeasurement(std::string_view name,
                                     std::vector<QuditInfo> targets,
                                     std::optional<std::string> register_name) {
  assert(!targets.empty() && "A measurement must have at least one target");
  auto findMaxID = [](const std::vector<QuditInfo> &qudits) -> std::size_t {
    return std::max_element(qudits.cbegin(), qudits.cend(),
                            [](auto &a, auto &b) { return a.id < b.id; })
        ->id;
  };
  numQudits = std::max(numQudits, findMaxID(targets) + 1);
  instructions.emplace_back(name, std::vector<double>{},
                            std::vector<QuditInfo>{}, std::move(targets),
                            std::nullopt, TraceInstructionType::Measurement,
                            std::move(register_name));
}
