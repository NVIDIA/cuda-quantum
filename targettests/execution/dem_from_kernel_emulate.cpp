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

int main() {
  runCase("SINGLE", singleDetector{});
  runCase("THREE_MZ", threeMzMultiDetector{});
  return 0;
}

// CHECK: SINGLE detectors=1 observables=0
// CHECK: THREE_MZ detectors=1 observables=1
