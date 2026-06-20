/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: mklq-backend,mklq-apple-silicon-target
// RUN: nvq++ --target mklq-cpu %s -o %t.cpu && %t.cpu | FileCheck %s --check-prefixes=CHECK,CPU
// RUN: nvq++ --target mklq-metal %s -o %t.metal && %t.metal | FileCheck %s --check-prefixes=CHECK,METAL

#include <cudaq.h>

#include <algorithm>
#include <complex>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(MKLQ_APPLE_SILICON_CPU_BASELINE) ==                                \
    defined(MKLQ_METAL_EXPERIMENTAL_CPU_ORACLE)
#error "expected exactly one MKL-Q target marker"
#endif

struct bell {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    mz(q);
  }
};

struct ghz {
  void operator()() __qpu__ {
    cudaq::qvector q(3);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[2]);
    mz(q);
  }
};

struct rotation_identity {
  void operator()(double theta) __qpu__ {
    cudaq::qubit q;
    x(q);
    rx(theta, q);
    rx(-theta, q);
    ry(theta, q);
    ry(-theta, q);
    rz(theta, q);
    rz(-theta, q);
    mz(q);
  }
};

struct y_gate {
  void operator()() __qpu__ {
    cudaq::qubit q;
    y(q);
    mz(q);
  }
};

struct cz_interference {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x(q[1]);
    z<cudaq::ctrl>(q[1], q[0]);
    h(q[0]);
    mz(q);
  }
};

struct swap_transfer {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    swap(q[0], q[1]);
    mz(q);
  }
};

struct bell_reset {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    reset(q[0]);
    mz(q[1]);
  }
};

struct bell_state {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  }
};

struct deterministic_feedback {
  bool operator()() __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    if (mz(q[0]))
      x(q[1]);
    return mz(q[1]);
  }
};

struct reset_after_mid_measurement {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    (void)mz(q);
    reset(q);
    x(q);
    return mz(q);
  }
};

static bool hasOnly(cudaq::sample_result &counts,
                    std::initializer_list<std::string> expected) {
  for (auto &[bits, count] : counts) {
    if (count == 0)
      continue;

    if (std::find(expected.begin(), expected.end(), bits) == expected.end()) {
      std::cerr << "unexpected bit string: " << bits << '\n';
      return false;
    }
  }

  for (auto &bits : expected) {
    if (counts.count(bits) == 0) {
      std::cerr << "missing expected bit string: " << bits << '\n';
      return false;
    }
  }

  return true;
}

static std::size_t countFor(cudaq::sample_result &counts,
                            const std::string &expected) {
  for (auto &[bits, count] : counts)
    if (bits == expected)
      return count;
  return 0;
}

static bool hasBalanced(cudaq::sample_result &counts,
                        std::initializer_list<std::string> expected,
                        std::size_t shots) {
  if (!hasOnly(counts, expected))
    return false;

  const auto lower = shots / 4;
  const auto upper = (shots * 3) / 4;
  for (auto &bits : expected) {
    const auto count = countFor(counts, bits);
    if (count < lower || count > upper) {
      std::cerr << "imbalanced bit string: " << bits << " count=" << count
                << '\n';
      return false;
    }
  }

  return true;
}

static bool hasBalancedRepeated(std::size_t shots) {
  std::size_t zeroCount = 0;
  std::size_t oneCount = 0;
  for (std::size_t i = 0; i < shots; ++i) {
    auto counts = cudaq::sample(1, bell_reset{});
    for (auto &[bits, count] : counts) {
      if (bits == "0")
        zeroCount += count;
      else if (bits == "1")
        oneCount += count;
      else {
        std::cerr << "unexpected reset bit string: " << bits << '\n';
        return false;
      }
    }
  }

  const auto lower = shots / 4;
  const auto upper = (shots * 3) / 4;
  if (zeroCount < lower || zeroCount > upper || oneCount < lower ||
      oneCount > upper) {
    std::cerr << "imbalanced reset counts: 0=" << zeroCount << " 1=" << oneCount
              << '\n';
    return false;
  }

  return true;
}

static bool allTrue(const std::vector<bool> &results,
                    std::string_view context) {
  for (auto result : results)
    if (!result) {
      std::cerr << context << " produced false\n";
      return false;
    }
  return true;
}

static bool stateIndexOutOfRangeRaises() {
  auto state = cudaq::get_state(bell_state{});
  try {
    (void)state[4];
  } catch (const std::runtime_error &error) {
    return std::string(error.what()).find("state index out of range") !=
           std::string::npos;
  }

  std::cerr << "state[4] did not raise on a 2-qubit state\n";
  return false;
}

static bool nonPowerOfTwoStateDataRaises() {
  try {
    std::vector<std::complex<double>> amplitudes{{1.0, 0.0}, {0.0, 0.0},
                                                 {0.0, 0.0}};
    cudaq::state invalidState(amplitudes);
  } catch (const std::runtime_error &error) {
    return std::string(error.what()).find("power-of-two state dimension") !=
           std::string::npos;
  }

  std::cerr << "non-power-of-two state data did not raise\n";
  return false;
}

static bool observeBellStateMatchesParityExpectations() {
  auto zz = cudaq::observe(
                bell_state{},
                cudaq::spin_op::z(0) * cudaq::spin_op::z(1))
                .expectation();
  if (std::abs(zz - 1.0) > 1.0e-6) {
    std::cerr << "unexpected Bell ZZ expectation: " << zz << '\n';
    return false;
  }

  auto xx = cudaq::observe(
                bell_state{},
                cudaq::spin_op::x(0) * cudaq::spin_op::x(1))
                .expectation();
  if (std::abs(xx - 1.0) > 1.0e-6) {
    std::cerr << "unexpected Bell XX expectation: " << xx << '\n';
    return false;
  }

  return true;
}

static bool runMidCircuitFeedbackMatchesExpectations() {
  constexpr std::size_t shots = 16;
  auto feedback = cudaq::run(shots, deterministic_feedback{});
  if (feedback.size() != shots) {
    std::cerr << "unexpected feedback result count: " << feedback.size()
              << '\n';
    return false;
  }
  if (!allTrue(feedback, "deterministic feedback"))
    return false;

  auto reset = cudaq::run(shots, reset_after_mid_measurement{});
  if (reset.size() != shots) {
    std::cerr << "unexpected reset result count: " << reset.size() << '\n';
    return false;
  }
  return allTrue(reset, "reset after mid-circuit measurement");
}

int main() {
  constexpr std::size_t bellShots = 1024;
  auto bellCounts = cudaq::sample(bellShots, bell{});
  if (!hasBalanced(bellCounts, {"00", "11"}, bellShots))
    return 1;

  constexpr std::size_t ghzShots = 1024;
  auto ghzCounts = cudaq::sample(ghzShots, ghz{});
  if (!hasBalanced(ghzCounts, {"000", "111"}, ghzShots))
    return 2;

  constexpr std::size_t resetShots = 128;
  if (!hasBalancedRepeated(resetShots))
    return 3;

  auto rotationCounts = cudaq::sample(128, rotation_identity{}, 0.37);
  if (!hasOnly(rotationCounts, {"1"}))
    return 4;

  auto yCounts = cudaq::sample(128, y_gate{});
  if (!hasOnly(yCounts, {"1"}))
    return 5;

  auto czCounts = cudaq::sample(128, cz_interference{});
  if (!hasOnly(czCounts, {"11"}))
    return 6;

  auto swapCounts = cudaq::sample(128, swap_transfer{});
  if (!hasOnly(swapCounts, {"01"}))
    return 7;

  if (!stateIndexOutOfRangeRaises())
    return 8;

#ifdef MKLQ_APPLE_SILICON_CPU_BASELINE
  if (!nonPowerOfTwoStateDataRaises())
    return 9;
#endif

  if (!observeBellStateMatchesParityExpectations())
    return 10;

  if (!runMidCircuitFeedbackMatchesExpectations())
    return 11;

#ifdef MKLQ_APPLE_SILICON_CPU_BASELINE
  std::cout << "mklq-target-marker:cpu\n";
#else
  std::cout << "mklq-target-marker:metal\n";
#endif

  std::cout << "mklq-observe-smoke-ok\n";
  std::cout << "mklq-mid-circuit-smoke-ok\n";
  std::cout << "mklq-runtime-smoke-ok\n";
  return 0;
}

// CPU: mklq-target-marker:cpu
// METAL: mklq-target-marker:metal
// CHECK: mklq-observe-smoke-ok
// CHECK: mklq-mid-circuit-smoke-ok
// CHECK: mklq-runtime-smoke-ok
