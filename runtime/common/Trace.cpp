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

void cudaq::Trace::appendInstruction(
    std::string_view name, std::vector<double> params,
    std::vector<QuditInfo> controls, std::vector<QuditInfo> targets,
    const std::vector<std::int32_t> &controlValues) {
  if (!controlValues.empty() && controlValues.size() != controls.size())
    throw std::invalid_argument(
        "required control value count does not match control count");
  if (std::any_of(controlValues.begin(), controlValues.end(),
                  [](auto value) { return value != 0 && value != 1; }))
    throw std::invalid_argument("required control value must be 0 or 1");
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
                            TraceInstructionType::Gate, std::nullopt,
                            controlValues);
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
