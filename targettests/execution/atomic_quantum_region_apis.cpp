/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ -c %s -o %t.o && nvq++ %t.o -o %t.staged && \
// RUN:   %t.staged | FileCheck %s
// clang-format on

#include <algorithm>
#include <cmath>
#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include <iostream>
#include <string>

struct atomic_h {
  void operator()(cudaq::qubit &q) __qpu__ __atomic_quantum_region__ { h(q); }
};

struct atomic_round_trip {
  void operator()() __qpu__ {
    cudaq::qubit q;
    atomic_h{}(q);
    cudaq::adjoint(atomic_h{}, q);
  }
};

struct measured_atomic_round_trip {
  int operator()() __qpu__ {
    cudaq::qubit q;
    atomic_h{}(q);
    cudaq::adjoint(atomic_h{}, q);
    bool result = mz(q);
    return result ? 1 : 0;
  }
};

static std::size_t countOccurrences(const std::string &text,
                                    const std::string &token) {
  std::size_t count = 0;
  std::size_t position = 0;
  while ((position = text.find(token, position)) != std::string::npos) {
    ++count;
    position += token.size();
  }
  return count;
}

int main() {
  constexpr std::size_t shots = 8;

  auto counts = cudaq::sample(shots, atomic_round_trip{});
  std::cout << "sample=" << counts.count("0") << '\n';

  auto result = cudaq::observe(atomic_round_trip{}, cudaq::spin_op::z(0));
  std::cout << "observe=" << std::round(result.expectation()) << '\n';

  auto values = cudaq::run(shots, measured_atomic_round_trip{});
  std::cout << "run=" << std::count(values.begin(), values.end(), 0) << '\n';

  auto state = cudaq::get_state(atomic_round_trip{});
  std::cout << "state=" << std::round(std::abs(state[0])) << ','
            << std::round(std::abs(state[1])) << '\n';

  // CHECK: sample=8
  // CHECK-NEXT: observe=1
  // CHECK-NEXT: run=8
  // CHECK-NEXT: state=1,0
}
