/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s
// clang-format off

#include <cudaq.h>

// A function of any kind can only take a quantum type if it is a kernel
// (`__qpu__`) or an intrinsic (`__qpu_intrinsic__`).

// expected-error@+1{{may not use quantum types in non-kernel functions}}
void byReference(cudaq::qubit &q) {}

// expected-error@+1{{may not use quantum types in non-kernel functions}}
void byValue(cudaq::qubit q);

// expected-error@+1{{may not use quantum types in non-kernel functions}}
void aContainer(const cudaq::qvector<> &v);

// expected-error@+1{{may not use quantum types in non-kernel functions}}
void aPointer(cudaq::qubit *q);

using Alias = cudaq::qubit;
// expected-error@+1{{may not use quantum types in non-kernel functions}}
void anAlias(Alias &q);

// expected-error@+1{{may not use quantum types in non-kernel functions}}
void ofQubits(std::vector<cudaq::qubit> &qs);

struct Methods {
  // expected-error@+1{{may not use quantum types in non-kernel functions}}
  Methods(cudaq::qubit &q) {}
  // expected-error@+1{{may not use quantum types in non-kernel functions}}
  void method(cudaq::qvector<> &v) {}
  // expected-error@+1{{may not use quantum types in non-kernel functions}}
  static void staticMethod(cudaq::qubit &q);
};

// A lambda that is not in a kernel is a function like any other.
// expected-error@+1{{may not use quantum types in non-kernel functions}}
auto classicalLambda = [](cudaq::qubit &q) {};

// These are fine.
__qpu__ void aKernel(cudaq::qubit &q) { h(q); }

__qpu_intrinsic__ void anIntrinsic(cudaq::qubit &q);

// A kernel that is declared before it is defined.
void declaredFirst(cudaq::qubit &q);
__qpu__ void declaredFirst(cudaq::qubit &q) { x(q); }

// Not taking quantum types.
void classical(double d, std::vector<double> &v) {}
// clang-format on
