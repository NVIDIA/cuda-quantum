/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t
// clang-format on

#include <cassert>
#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

int main() {
  auto bellKernel = []() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  };

  // `estimate` returns the counts `estimate_resources` returns, wrapped in an
  // `estimate_result`.
  auto result = cudaq::estimate(bellKernel);
  assert(result.get_resources().count("h") == 1);
  assert(result.get_resources().count_controls("x", /*nControls=*/1) == 1);
  assert(result.get_resources().count() == 2);

  // Same for the choice overload.
  auto chosen = cudaq::estimate([]() { return true; }, bellKernel);
  assert(chosen.get_resources().count("h") == 1);
  assert(chosen.get_resources().count_controls("x", /*nControls=*/1) == 1);

  // `estimate_resources` is now a thin wrapper and must keep agreeing.
  auto resources = cudaq::estimate_resources(bellKernel);
  assert(resources.count() == result.get_resources().count());

  return 0;
}
