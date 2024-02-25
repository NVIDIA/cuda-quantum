/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Trace.h"
#include <algorithm>
#include <cassert>

namespace cudaq {

void Trace::appendInstruction(std::string_view name,
                              const std::vector<double> &params,
                              const std::vector<std::size_t> &controls,
                              const std::vector<std::size_t> &targets) {
  assert(!targets.empty() && "A instruction must have at least one target");
  auto findMaxID = [](const std::vector<std::size_t> &qudits) -> std::size_t {
    return *std::max_element(qudits.cbegin(), qudits.cend());
  };
  std::size_t maxID = findMaxID(targets);
  if (!controls.empty())
    maxID = std::max(maxID, findMaxID(controls));
  numQudits = std::max(numQudits, maxID + 1);
  instructions.emplace_back(name, params, controls, targets);
}

void Trace::appendInstruction(std::string_view name,
                              const std::vector<float> &params,
                              const std::vector<std::size_t> &controls,
                              const std::vector<std::size_t> &targets) {
  std::vector<double> converted_params;
  converted_params.reserve(params.size());
  std::transform(params.begin(), params.end(),
                 std::back_inserter(converted_params),
                 [](float p) { return static_cast<double>(p); });
  appendInstruction(name, converted_params, controls, targets);
}

} // namespace cudaq
