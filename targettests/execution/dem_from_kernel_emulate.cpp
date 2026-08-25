/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target anyon      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq       --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_20.txt %t | FileCheck %s
// RUN: nvq++ --target oqc        --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target qbraid     --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cctype>
#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithms/dem.h>
#include <exception>
#include <string>
#include <vector>

struct singleDetector {
  void operator()() __qpu__ {
    cudaq::qubit q;
    auto m = mz(q);
    cudaq::detector(m);
  }
};

struct threeMzMultiDetector {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1, q2;
    x(q0);
    x(q1);
    auto m0 = mz(q0);
    auto m1 = mz(q1);
    auto m2 = mz(q2);
    cudaq::detector(m0, m1, m2);
    cudaq::logical_observable(m0);
  }
};

struct correlatedXXHyperedge {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    x<cudaq::ctrl>(q0, q1);
    auto m0 = mz(q0);
    auto m1 = mz(q1);
    cudaq::detector(m0);
    cudaq::detector(m0);
    cudaq::detector(m1);
    cudaq::detector(m1);
  }
};

static std::size_t countOccurrences(const std::string &haystack,
                                    const std::string &needle) {
  std::size_t count = 0, pos = 0;
  while ((pos = haystack.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

static std::size_t maxIndexAfter(const std::string &haystack, char prefix) {
  long maxIdx = -1;
  for (std::size_t i = 0; i + 1 < haystack.size(); ++i) {
    if (haystack[i] != prefix ||
        !std::isdigit(static_cast<unsigned char>(haystack[i + 1])))
      continue;
    if (i > 0) {
      char prev = haystack[i - 1];
      if (prev != ' ' && prev != '\t' && prev != '\n')
        continue;
    }
    long idx = 0;
    std::size_t j = i + 1;
    while (j < haystack.size() &&
           std::isdigit(static_cast<unsigned char>(haystack[j]))) {
      idx = idx * 10 + (haystack[j] - '0');
      ++j;
    }
    if (idx > maxIdx)
      maxIdx = idx;
  }
  return maxIdx < 0 ? 0 : static_cast<std::size_t>(maxIdx + 1);
}

static const cudaq::noise_model g_emptyNoise{};

template <typename Kernel>
static void runCase(const char *label, Kernel &&kernel) {
  try {
    std::string demText =
        cudaq::dem_from_kernel(std::forward<Kernel>(kernel), &g_emptyNoise);
    std::printf("%s detectors=%zu observables=%zu\n", label,
                maxIndexAfter(demText, 'D'), maxIndexAfter(demText, 'L'));
  } catch (const std::exception &e) {
    std::printf("%s THREW: %s\n", label, e.what());
  }
}

// Unlike dem_from_kernel, cudaq::sample routes through
// createEmulationTargetPrepPipeline, which must strip QEC ops.
template <typename Kernel>
static void runSampleCase(const char *label, Kernel &&kernel) {
  try {
    auto counts = cudaq::sample(std::forward<Kernel>(kernel));
    std::printf("%s most_probable=%s\n", label, counts.most_probable().c_str());
  } catch (const std::exception &e) {
    std::printf("%s THREW: %s\n", label, e.what());
  }
}

template <typename Kernel>
static void runDecomposeCase(const char *label, Kernel &&kernel) {
  try {
    std::vector<cudaq::real> pauli2Probs(15, 0.0);
    pauli2Probs[4] = 0.25; // XX
    cudaq::noise_model noise;
    noise.add_channel("x", {0, 1}, cudaq::pauli2(pauli2Probs));

    std::string demRaw = cudaq::dem_from_kernel(kernel, &noise);
    std::string demDecomposed = cudaq::dem_from_kernel(
        kernel, &noise,
        /*options=*/cudaq::dem_options{.decompose_errors = true});
    std::printf("%s_RAW hyperedge=%d caret=%d\n", label,
                demRaw.find("D0 D1 D2 D3") != std::string::npos ? 1 : 0,
                demRaw.find('^') != std::string::npos ? 1 : 0);
    std::printf("%s_DECOMPOSED hyperedge=%d caret=%d\n", label,
                demDecomposed.find("D0 D1 D2 D3") != std::string::npos ? 1 : 0,
                demDecomposed.find('^') != std::string::npos ? 1 : 0);
  } catch (const std::exception &e) {
    std::printf("%s THREW: %s\n", label, e.what());
  }
}

int main() {
  runCase("SINGLE", singleDetector{});
  runCase("THREE_MZ", threeMzMultiDetector{});
  runDecomposeCase("CORRELATED_XX", correlatedXXHyperedge{});
  runSampleCase("SAMPLE_QEC_KERNEL", singleDetector{});
  return 0;
}

// CHECK: SINGLE detectors=1 observables=0
// CHECK: THREE_MZ detectors=1 observables=1
// CHECK: CORRELATED_XX_RAW hyperedge=1 caret=0
// CHECK: CORRELATED_XX_DECOMPOSED hyperedge=0 caret=1
// CHECK: SAMPLE_QEC_KERNEL most_probable=0
