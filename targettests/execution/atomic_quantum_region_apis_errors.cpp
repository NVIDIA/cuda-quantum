/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: not nvq++ -DATOMIC_FREE_MEASUREMENT %s -o %t.free 2>&1 | FileCheck %s --check-prefix=MEASUREMENT
// RUN: not nvq++ -DATOMIC_FUNCTOR_ALLOCATION %s -o %t.functor 2>&1 | FileCheck %s --check-prefix=ALLOCATION
// RUN: not nvq++ -DATOMIC_LAMBDA_MEASUREMENT %s -o %t.lambda 2>&1 | FileCheck %s --check-prefix=MEASUREMENT
// RUN: not nvq++ -DATOMIC_FREE_MEASUREMENT -c %s -o %t.o 2>&1 | FileCheck %s --check-prefix=MEASUREMENT
// RUN: if %braket_avail; then not nvq++ --target braket --emulate -DATOMIC_FREE_MEASUREMENT %s -o %t.braket 2>&1 | FileCheck %s --check-prefix=MEASUREMENT; fi
// RUN: if %oqc_avail; then not nvq++ --target oqc --emulate -DATOMIC_FREE_MEASUREMENT %s -o %t.oqc 2>&1 | FileCheck %s --check-prefix=MEASUREMENT; fi
// RUN: not nvq++ --target quantinuum --emulate -DATOMIC_FREE_MEASUREMENT %s -o %t.quantinuum 2>&1 | FileCheck %s --check-prefix=MEASUREMENT
//
// The check runs before any lowering, so the diagnostic names this source file
// and the offending line rather than a temporary Quake file.
// MEASUREMENT-COUNT-1: atomic_quantum_region_apis_errors.cpp:{{[0-9]+}}:{{[0-9]+}}: error: 'quake.mz' op measurement operations are not supported inside an atomic quantum region; measure outside the region
// MEASUREMENT-NOT: measurement operations are not supported inside an atomic quantum region
//
// ALLOCATION-COUNT-1: atomic_quantum_region_apis_errors.cpp:{{[0-9]+}}:{{[0-9]+}}: error: 'quake.alloca' op qubit allocations are not supported inside an atomic quantum region; allocate in the caller and pass the qubits as arguments
// ALLOCATION-NOT: qubit allocations are not supported inside an atomic quantum region
// clang-format on

#include <cudaq.h>

#if defined(ATOMIC_FREE_MEASUREMENT)

__qpu__ void measure_helper(cudaq::qubit &q) { mz(q); }

__qpu__ __atomic_quantum_region__ void invalid_region(cudaq::qubit &q) {
  measure_helper(q);
}

struct test_kernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    invalid_region(q);
    x(q);
  }
};

#elif defined(ATOMIC_FUNCTOR_ALLOCATION)

struct test_kernel {
  void operator()() __qpu__ __atomic_quantum_region__ {
    cudaq::qubit q;
    x(q);
  }
};

#elif defined(ATOMIC_LAMBDA_MEASUREMENT)

struct test_kernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    auto invalid_region = [](cudaq::qubit &target)
                              __qpu__ __atomic_quantum_region__ { mz(target); };
    invalid_region(q);
  }
};

#endif

int main() {
  cudaq::sample(test_kernel{});
  return 0;
}
