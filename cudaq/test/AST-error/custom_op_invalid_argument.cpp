/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

CUDAQ_REGISTER_OPERATION(custom_cnot, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0})

__qpu__ void too_many() {
  cudaq::qubit a, b, c;
  // expected-error@+1{{requires 2 individual qubit target(s)}}
  custom_cnot(a, b, c);
}

__qpu__ void too_few() {
  cudaq::qubit q;
  // expected-error@+1{{requires 2 individual qubit target(s)}}
  custom_cnot(q);
}

__qpu__ void whole_qvector() {
  cudaq::qvector q(2);
  // expected-error@+1{{requires 2 individual qubit target(s)}}
  custom_cnot(q);
}

__qpu__ void qubit_then_qvector() {
  cudaq::qvector q(2);
  cudaq::qubit r;
  // expected-error@+1{{requires 2 individual qubit target(s)}}
  custom_cnot(r, q);
}

__qpu__ void ctrl_with_qvector_target() {
  cudaq::qubit c;
  cudaq::qvector q(2);
  // expected-error@+1{{requires 2 individual qubit target(s)}}
  custom_cnot<cudaq::ctrl>(c, q);
}
