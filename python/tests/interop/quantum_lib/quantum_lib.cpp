/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "quantum_lib.h"

namespace cudaq {
__qpu__ void
entryPoint(const std::function<void(cudaq::qvector<> &)> &statePrep) {
  cudaq::qvector q(2);
  statePrep(q);
}

__qpu__ void qft(cudaq::qview<> qubits) {
  // not really qft, just for testing
  h(qubits);
}

__qpu__ void qft(cudaq::qview<> qubits, const std::vector<double> &x,
                 std::size_t k) {
  h(qubits[k]);
  ry(x[0], qubits[k]);
}

__qpu__ void another(cudaq::qview<> qubits, std::size_t i) { x(qubits[i]); }

__qpu__ void uccsd(cudaq::qview<> qubits, std::size_t) { h(qubits[0]); }
} // namespace cudaq