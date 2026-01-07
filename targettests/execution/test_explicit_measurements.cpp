/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target stim %s -o %t && CUDAQ_LOG_LEVEL=info %t | grep "Creating new Stim frame simulator" | wc -l | FileCheck %s
// RUN: nvq++ --target anyon      --emulate %s -o %t && %t 2>&1 | FileCheck %s -check-prefix=FAIL
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t 2>&1 | FileCheck %s -check-prefix=FAIL ; fi
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t 2>&1 | FileCheck %s -check-prefix=FAIL
// RUN: nvq++ --target ionq       --emulate %s -o %t && %t 2>&1 | FileCheck %s -check-prefix=FAIL
// RUN: nvq++ --target iqm --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt %t 2>&1 | FileCheck %s -check-prefix=FAIL
// RUN: nvq++ --target oqc        --emulate %s -o %t && %t 2>&1 | FileCheck %s -check-prefix=FAIL
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t 2>&1 | FileCheck %s -check-prefix=FAIL
// clang-format on

#include <cudaq.h>

auto explicit_kernel = [](int n_qubits, int n_rounds) __qpu__ {
  cudaq::qvector q(n_qubits);
  for (int round = 0; round < n_rounds; round++) {
    h(q[0]);
    for (int i = 1; i < n_qubits; i++)
      x<cudaq::ctrl>(q[i - 1], q[i]);
    mz(q);
    for (int i = 0; i < n_qubits; i++)
      reset(q[i]);
  }
};

int main() {
  int n_qubits = 2;
  int n_rounds = 4;
  std::size_t num_shots = 5;
  cudaq::sample_options options{.shots = num_shots,
                                .explicit_measurements = true};
  auto results = cudaq::sample(options, explicit_kernel, n_qubits, n_rounds);
  results.dump();
  return 0;
}

// CHECK: 1

// FAIL: not supported on this target
