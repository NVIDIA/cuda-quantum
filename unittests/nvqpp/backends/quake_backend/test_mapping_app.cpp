/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

#include <algorithm>
#include <iostream>

// A scalar return requires no classical allocation, so canonicalization may
// erase the `cc.scope` around the `run` kernel.

struct bool_return_mapping {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    return true;
  }
};

struct int_return_mapping {
  int operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    return 2 + 3;
  }
};

struct float_return_mapping {
  float operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    return 1.5f;
  }
};

// The fake server requires `mapping` in the entry-point name before checking
// for mapping metadata. The branch operations leave q1 in a deterministic
// state so the returned measurement also verifies which path executed.
struct simple_false_feedforward_mapping {
  bool operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    const bool measurementResult = mz(q0);
    if (measurementResult) {
      s(q1);
      x(q1);
    } else {
      h(q1);
      h(q1);
    }
    return mz(q1);
  }
};

struct simple_true_feedforward_mapping {
  bool operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    x(q0);
    const bool measurementResult = mz(q0);
    if (measurementResult) {
      s(q1);
      x(q1);
    } else {
      h(q1);
      h(q1);
    }
    return mz(q1);
  }
};

struct inline_measurement_feedforward_mapping {
  bool operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    if (mz(q0)) {
      s(q1);
      x(q1);
    } else {
      h(q1);
      h(q1);
    }
    return mz(q1);
  }
};

int main() {
  const auto boolResults = cudaq::run(1, bool_return_mapping{});
  if (boolResults.size() != 1 || !boolResults.front()) {
    std::cerr << "Scalar bool return failed.\n";
    return 1;
  }

  const auto intResults = cudaq::run(1, int_return_mapping{});
  if (intResults.size() != 1 || intResults.front() != 5) {
    std::cerr << "Scalar int return failed.\n";
    return 1;
  }

  const auto floatResults = cudaq::run(1, float_return_mapping{});
  if (floatResults.size() != 1 || floatResults.front() != 1.5f) {
    std::cerr << "Scalar float return failed.\n";
    return 1;
  }

  const auto falseFeedforwardResults =
      cudaq::run(3, simple_false_feedforward_mapping{});
  if (falseFeedforwardResults.size() != 3 ||
      std::ranges::any_of(falseFeedforwardResults,
                          [](bool result) { return result; })) {
    std::cerr << "False measurement feedforward failed.\n";
    return 1;
  }

  const auto trueFeedforwardResults =
      cudaq::run(3, simple_true_feedforward_mapping{});
  if (trueFeedforwardResults.size() != 3 ||
      std::ranges::any_of(trueFeedforwardResults,
                          [](bool result) { return !result; })) {
    std::cerr << "True measurement feedforward failed.\n";
    return 1;
  }

  const auto inlineFeedforwardResults =
      cudaq::run(3, inline_measurement_feedforward_mapping{});
  if (inlineFeedforwardResults.size() != 3 ||
      std::ranges::any_of(inlineFeedforwardResults,
                          [](bool result) { return result; })) {
    std::cerr << "Inline measurement feedforward failed.\n";
    return 1;
  }

  std::cout << "Mapped scalar returns and measurement feedforward passed.\n";
  return 0;
}
