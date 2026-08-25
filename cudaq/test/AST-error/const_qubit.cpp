/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: ( cudaq-quake %s || true ) 2>&1 | FileCheck %s --implicit-check-not="Cannot operate on a qudit with Levels != 2" --implicit-check-not="no matching function for call to 'qubitToQuditInfo'" --implicit-check-not="no matching function for call to 'qubitIsNegative'"

// CHECK-COUNT-8: static assertion failed{{.*}}Cannot apply a quantum operation to a const qubit.
// CHECK: C++ source has errors. nvq++ cannot proceed.
// clang-format on

#include <cudaq.h>

struct for_in_vector {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    for (const auto &qubit : q) {
      x(qubit);
      x(qubit);
    }
  }
};

struct parameterized_gate {
  auto operator()() __qpu__ {
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      rx(0.5, qubit);
  }
};

struct u3_gate {
  auto operator()() __qpu__ {
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      u3(0.1, 0.2, 0.3, qubit);
  }
};

struct swap_gate {
  auto operator()() __qpu__ {
    cudaq::qubit target;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      swap(qubit, target);
  }
};

struct controlled_gate {
  auto operator()() __qpu__ {
    cudaq::qubit target;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      x<cudaq::ctrl>(qubit, target);
  }
};

struct controlled_parameterized_gate {
  auto operator()() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      rx<cudaq::ctrl>(0.5, control, qubit);
  }
};

struct controlled_u3_gate {
  auto operator()() __qpu__ {
    cudaq::qubit control;
    cudaq::qubit target;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      u3<cudaq::ctrl>(0.1, 0.2, 0.3, control, qubit, target);
  }
};

struct controlled_swap_gate {
  auto operator()() __qpu__ {
    cudaq::qubit control;
    cudaq::qubit target;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      swap<cudaq::ctrl>(control, qubit, target);
  }
};
