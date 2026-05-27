/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <cudaq/algorithms/dem.h>
#include <cstdio>
#include <exception>
#include <string>

// ---------------------------------------------------------------------------
// Test kernels
// ---------------------------------------------------------------------------

// Trivial kernel with no QEC declarations. DEM should be empty.
struct trivialKernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  }
};

// Bit-flip noise on a single measured qubit + scalar detector.
struct singleNoisyDetector {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::apply_noise<cudaq::x_error>(0.1, q);
    auto m = mz(q);
    cudaq::detector(m);
  }
};

// Three measurements, one detector spanning all three, one observable on
// the first measurement.
struct threeMzMultiDetector {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1, q2;
    x(q0);
    x(q1);
    cudaq::apply_noise<cudaq::x_error>(0.05, q0);
    cudaq::apply_noise<cudaq::x_error>(0.05, q1);
    cudaq::apply_noise<cudaq::x_error>(0.05, q2);
    auto m0 = mz(q0);
    auto m1 = mz(q1);
    auto m2 = mz(q2);
    cudaq::detector(m0, m1, m2);
    cudaq::logical_observable(m0);
  }
};

// Multi-round memory experiment with cross-round detectors and a final
// logical observable.
struct memoryExperimentTwoRounds {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1, q2;
    cudaq::apply_noise<cudaq::x_error>(0.03, q0);
    cudaq::apply_noise<cudaq::x_error>(0.03, q1);
    cudaq::apply_noise<cudaq::x_error>(0.03, q2);
    // Round 0
    auto m0_r0 = mz(q0);
    auto m1_r0 = mz(q1);
    auto m2_r0 = mz(q2);
    // Round 1
    cudaq::apply_noise<cudaq::x_error>(0.03, q0);
    cudaq::apply_noise<cudaq::x_error>(0.03, q1);
    cudaq::apply_noise<cudaq::x_error>(0.03, q2);
    auto m0_r1 = mz(q0);
    auto m1_r1 = mz(q1);
    auto m2_r1 = mz(q2);
    // Cross-round detectors
    cudaq::detector(m0_r0, m0_r1);
    cudaq::detector(m1_r0, m1_r1);
    cudaq::detector(m2_r0, m2_r1);
    cudaq::logical_observable(m0_r1, m1_r1, m2_r1);
  }
};

// Vectorized form: `mz(qvector)` returns a vector of handles and
// `cudaq::detectors(prev, curr)` declares N detectors element-wise.
struct vectorizedDetectors {
  void operator()() __qpu__ {
    cudaq::qvector qvec(3);
    cudaq::apply_noise<cudaq::x_error>(0.04, qvec[0]);
    cudaq::apply_noise<cudaq::x_error>(0.04, qvec[1]);
    cudaq::apply_noise<cudaq::x_error>(0.04, qvec[2]);
    auto m_prev = mz(qvec);
    cudaq::apply_noise<cudaq::x_error>(0.04, qvec[0]);
    cudaq::apply_noise<cudaq::x_error>(0.04, qvec[1]);
    cudaq::apply_noise<cudaq::x_error>(0.04, qvec[2]);
    auto m_curr = mz(qvec);
    cudaq::detectors(m_prev, m_curr);
    cudaq::logical_observable(m_curr[0], m_curr[1], m_curr[2]);
  }
};

__qpu__ void demoKernel() {
  cudaq::qubit q0, q1, q2;
  x(q0);
  x(q1);
  x(q2);
  auto m0 = mz(q0);
  auto m1 = mz(q1);
  auto m2 = mz(q2);
  cudaq::detector(m0, m1, m2);
  cudaq::logical_observable(m0);
}

__qpu__ void demoKernel2(int nRounds) {
  cudaq::qubit q0, q1, q2;
  x(q0);
  x(q1);
  x(q2);
  auto m0 = mz(q0);
  auto m1 = mz(q1);
  auto m2 = mz(q2);
  for (int round = 0; round < nRounds; ++round) {
    auto m0_new = mz(q0);
    auto m1_new = mz(q1);
    auto m2_new = mz(q2);
    cudaq::detector(m0_new, m0);
    cudaq::detector(m1_new, m1);
    cudaq::detector(m2_new, m2);
    m0 = m0_new;
    m1 = m1_new;
    m2 = m2_new;
  }
  cudaq::logical_observable(m0, m1, m2);
}

// Non-Clifford gate: should throw "Clifford only" diagnostic because
// Stim cannot simulate non-Clifford gates.
struct nonClifford {
  void operator()() __qpu__ {
    cudaq::qubit q;
    ry(0.3, q);
    auto m = mz(q);
    cudaq::detector(m);
  }
};

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

// Count occurrences of @p needle in @p haystack. Used to tally how many
// `error(` lines the returned `.dem` text contains.
static std::size_t countOccurrences(const std::string &haystack,
                                    const std::string &needle) {
  std::size_t count = 0, pos = 0;
  while ((pos = haystack.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

// Detectors and observables appear in the `.dem` file as TARGETS in
// `error(...)` lines, not as standalone instructions. `D5` is a reference
// to detector 5; `L2` is a reference to logical observable 2. Find the
// highest index per kind and return `max + 1` as the count (the .dem
// format does not pre-declare detector/observable counts — they are
// implied by the max index used).
static std::size_t maxIndexAfter(const std::string &haystack, char prefix) {
  long maxIdx = -1;
  for (std::size_t i = 0; i + 1 < haystack.size(); ++i) {
    // A target letter is preceded by whitespace and followed by a digit.
    if (haystack[i] != prefix || !std::isdigit(haystack[i + 1]))
      continue;
    if (i > 0) {
      char prev = haystack[i - 1];
      if (prev != ' ' && prev != '\t' && prev != '\n')
        continue;
    }
    long idx = 0;
    std::size_t j = i + 1;
    while (j < haystack.size() && std::isdigit(haystack[j])) {
      idx = idx * 10 + (haystack[j] - '0');
      ++j;
    }
    if (idx > maxIdx)
      maxIdx = idx;
  }
  return maxIdx < 0 ? 0 : static_cast<std::size_t>(maxIdx + 1);
}

static const cudaq::noise_model g_emptyNoise{};

template <typename Kernel, typename... Args>
static void runCase(const char *label, Kernel &&kernel, Args &&...args) {
  try {
    std::string demText = cudaq::dem_from_kernel(
        std::forward<Kernel>(kernel), &g_emptyNoise,
        std::forward<Args>(args)...);
    // Tally distinct DEM instructions.
    std::printf(
        "%s errors=%zu detectors=%zu observables=%zu\n", label,
        countOccurrences(demText, "error("), maxIndexAfter(demText, 'D'),
        maxIndexAfter(demText, 'L'));
  } catch (const std::exception &e) {
    std::printf("%s THREW: %s\n", label, e.what());
  }
}

static void runTrivial() {
  try {
    std::string demText = cudaq::dem_from_kernel(trivialKernel{});
    std::printf("TRIVIAL errors=%zu detectors=%zu observables=%zu\n",
                countOccurrences(demText, "error("),
                maxIndexAfter(demText, 'D'), maxIndexAfter(demText, 'L'));
  } catch (const std::exception &e) {
    std::printf("TRIVIAL THREW: %s\n", e.what());
  }
}

template <typename Kernel, typename... Args>
static void runNoNoiseCase(const char *label, Kernel &&kernel, Args &&...args) {
  try {
    std::string demText = cudaq::dem_from_kernel(
        std::forward<Kernel>(kernel), std::forward<Args>(args)...);
    std::printf(
        "%s errors=%zu detectors=%zu observables=%zu\n", label,
        countOccurrences(demText, "error("), maxIndexAfter(demText, 'D'),
        maxIndexAfter(demText, 'L'));
  } catch (const std::exception &e) {
    std::printf("%s THREW: %s\n", label, e.what());
  }
}

int main() {
  // Trivial: kernel without QEC ops yields an empty DEM.
  runTrivial();

  // Detector / logical-observable source examples without any noise model.
  runNoNoiseCase("DEMO_NO_NOISE", demoKernel);
  runNoNoiseCase("DEMO_ROUNDS_NO_NOISE", demoKernel2, 2);

  // Single noisy qubit, single detector: one X_ERROR mechanism, one
  // detector reference. No observable.
  runCase("SINGLE_NOISY", singleNoisyDetector{});

  // Three noisy measurements with multi-handle detector and observable.
  runCase("THREE_MZ", threeMzMultiDetector{});

  // Memory experiment, two rounds, three cross-round detectors plus one
  // multi-target observable.
  runCase("MEM_EXP_2R", memoryExperimentTwoRounds{});

  // Vectorized stdvec form producing the same shape as MEM_EXP_2R.
  runCase("VECTORIZED", vectorizedDetectors{});

  // Non-Clifford gate must surface as a Stim diagnostic.
  runNoNoiseCase("NON_CLIFFORD", nonClifford{});

  return 0;
}

// CHECK: TRIVIAL errors=0 detectors=0 observables=0
// CHECK: DEMO_NO_NOISE errors=0 detectors=1 observables=1
// CHECK: DEMO_ROUNDS_NO_NOISE errors=0 detectors=6 observables=1
// CHECK: SINGLE_NOISY errors=1 detectors=1 observables=0
// CHECK: THREE_MZ errors=2 detectors=1 observables=1
// CHECK: MEM_EXP_2R errors=4 detectors=3 observables=1
// CHECK: VECTORIZED errors=4 detectors=3 observables=1
// CHECK: NON_CLIFFORD THREW: {{.*}}Clifford{{.*}}
