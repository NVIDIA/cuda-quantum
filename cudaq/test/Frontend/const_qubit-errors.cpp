/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Diagnostic verification treats the expected Clang errors as success, so
// cudaq-quake continues Quake AST traversal on the invalid AST. Ignore the
// resulting secondary traversal errors.
// RUN: cudaq-quake -verify -Xcudaq -Xclang \
// RUN:   -Xcudaq -verify-ignore-unexpected=error %s -o /dev/null

#include <cudaq.h>

struct for_in_vector {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    for (const auto &qubit : q) {
      // expected-error@* {{Cannot apply a quantum operation to a const qubit.}}
      // expected-note@+1 {{in instantiation of function template}}
      x(qubit);
      x(qubit);
    }
  }
};

struct parameterized_gate {
  auto operator()() __qpu__ {
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      // expected-error@* {{Cannot apply a quantum operation to a const qubit.}}
      // expected-note@+1 {{in instantiation of function template}}
      rx(0.5, qubit);
  }
};

struct u3_gate {
  auto operator()() __qpu__ {
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      // expected-error@* {{Cannot apply a quantum operation to a const qubit.}}
      // expected-note@+1 {{in instantiation of function template}}
      u3(0.1, 0.2, 0.3, qubit);
  }
};

struct swap_gate {
  auto operator()() __qpu__ {
    cudaq::qubit target;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      // expected-error@* {{Cannot apply a quantum operation to a const qubit.}}
      // expected-note@+1 {{in instantiation of function template}}
      swap(qubit, target);
  }
};

struct controlled_gate {
  auto operator()() __qpu__ {
    cudaq::qubit target;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      // expected-error@* {{Cannot apply a quantum operation to a const qubit.}}
      // expected-note@+1 {{in instantiation of function template}}
      x<cudaq::ctrl>(qubit, target);
  }
};

struct controlled_parameterized_gate {
  auto operator()() __qpu__ {
    cudaq::qubit control;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      // expected-error@* {{Cannot apply a quantum operation to a const qubit.}}
      // expected-note@+1 {{in instantiation of function template}}
      rx<cudaq::ctrl>(0.5, control, qubit);
  }
};

struct controlled_u3_gate {
  auto operator()() __qpu__ {
    cudaq::qubit control;
    cudaq::qubit target;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      // expected-error@* {{Cannot apply a quantum operation to a const qubit.}}
      // expected-note@+1 {{in instantiation of function template}}
      u3<cudaq::ctrl>(0.1, 0.2, 0.3, control, qubit, target);
  }
};

struct controlled_swap_gate {
  auto operator()() __qpu__ {
    cudaq::qubit control;
    cudaq::qubit target;
    cudaq::qvector q(1);
    for (const auto &qubit : q)
      // expected-error@* {{Cannot apply a quantum operation to a const qubit.}}
      // expected-note@+1 {{in instantiation of function template}}
      swap<cudaq::ctrl>(control, qubit, target);
  }
};
