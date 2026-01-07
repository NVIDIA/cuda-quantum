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
  instructions.emplace_back(name, params, controls, targets);
}
