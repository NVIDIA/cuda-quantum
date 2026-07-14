/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace {

constexpr std::size_t shots = 4;

template <typename Kernel>
bool sampleAndCheck(std::string_view name, Kernel &&kernel,
                    std::string_view expected) {
  auto result = cudaq::sample(shots, std::forward<Kernel>(kernel));
  const auto counts = result.to_map(cudaq::GlobalRegisterName);
  const auto expectedIter = counts.find(std::string(expected));
  const bool passed = counts.size() == 1 && expectedIter != counts.end() &&
                      expectedIter->second == shots;

  if (passed) {
    std::cout << "PASS " << name << ": " << expected << '\n';
    return true;
  }

  std::cerr << "FAIL " << name << ": expected " << expected << '\n';
  result.dump();
  return false;
}

} // namespace

int main() {
  bool passed = true;

  auto terminalVector = []() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    x(q[2]);
    mz(q);
  };
  passed &= sampleAndCheck("terminal-vector", terminalVector, "101");

  auto implicitMeasurements = []() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    x(q[2]);
  };
  // q[1] is unused and is eliminated before implicit measurements are added.
  passed &= sampleAndCheck("implicit-measurements", implicitMeasurements, "11");

  auto terminalRouting = []() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    x(q[1]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[0], q[2]);
    x<cudaq::ctrl>(q[1], q[2]);
    mz(q);
  };
  passed &= sampleAndCheck("terminal-routing", terminalRouting, "101");

  auto implicitRouting = []() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    x(q[1]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[0], q[2]);
    x<cudaq::ctrl>(q[1], q[2]);
  };
  passed &= sampleAndCheck("implicit-routing", implicitRouting, "101");

  auto terminalUserSwap = []() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    swap(q[0], q[2]);
    mz(q);
  };
  passed &= sampleAndCheck("terminal-user-swap", terminalUserSwap, "001");

  auto implicitUserSwap = []() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    swap(q[0], q[2]);
  };
  // q[1] is unused and is eliminated before implicit measurements are added.
  passed &= sampleAndCheck("implicit-user-swap", implicitUserSwap, "01");

  // These reversed and partial cases extend the default-sampling ordering
  // convention, as validated in
  // `test_explicit_measurements.py::test_measurement_order`:
  // measured bits follow qubit allocation order, not `mz` execution order.
  auto reversedTerminalOrder = []() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    x(q[2]);
    mz(q[2]);
    mz(q[0]);
    mz(q[1]);
  };
  passed &=
      sampleAndCheck("reversed-terminal-order", reversedTerminalOrder, "101");

  auto partialTerminalMeasurement = []() __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    x(q[1]);
    mz(q[2]);
    mz(q[0]);
  };
  passed &= sampleAndCheck("partial-terminal-measurement",
                           partialTerminalMeasurement, "10");

  return passed ? 0 : 1;
}
