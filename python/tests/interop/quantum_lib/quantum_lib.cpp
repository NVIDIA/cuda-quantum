/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "quantum_lib.h"

namespace py = pybind11;

__qpu__ void
cudaq::entryPoint(const std::function<void(cudaq::qvector<> &)> &statePrep) {
  cudaq::qvector q(2);
  statePrep(q);
}

__qpu__ void cudaq::qft(cudaq::qview<> qubits) {
  // not really qft, just for testing
  h(qubits);
}

__qpu__ void cudaq::qft(cudaq::qview<> qubits, const std::vector<double> &x,
                        std::size_t k) {
  h(qubits[k]);
  ry(x[0], qubits[k]);
}

__qpu__ void cudaq::another(cudaq::qview<> qubits, std::size_t i) {
  x(qubits[i]);
}

__qpu__ void cudaq::uccsd(cudaq::qview<> qubits, std::size_t) { h(qubits[0]); }

//===----------------------------------------------------------------------===//
// Callback tests.
//
// These are C++ kernels that were called from the Python interpreter that call
// back to Python kernel decorators.
//===----------------------------------------------------------------------===//

__qpu__ void cudaq::sit_and_spin_test(cudaq::qkernel<void()> &&qern) {
  // Arguments to qern were all synthesized.
  qern();
}

__qpu__ void cudaq::marshal_test(cudaq::qkernel<void(std::size_t)> &&qern,
                                 std::size_t qnum) {
  qern(qnum);
  qern(qnum + 4);
}

__qpu__ void cudaq::plug_and_chug_test(cudaq::qkernel<void()> &&qern) {
  // qern had no arguments; base case.
  qern();
}

__qpu__ void
cudaq::brain_bend_test(cudaq::qkernel<void(cudaq::qvector<> &)> &&qern) {
  // qern takes a quantum argument.
  cudaq::qvector qs(5);
  qern(qs);
}

__qpu__ void cudaq::most_curious_test(
    cudaq::qkernel<void(cudaq::qvector<> &, std::size_t)> &&qern) {
  // qern takes a quantum argument and a classical argument.
  cudaq::qvector qs(5);
  qern(qs, 4);
}
