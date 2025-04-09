/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Sample]
#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithms/draw.h>

// Define a quantum kernel function.
__qpu__ void kernel(int qubit_count) {
  cudaq::qvector qvector(qubit_count);
  // 2-qubit GHZ state.
  h(qvector[0]);
  for (auto qubit : cudaq::range(qubit_count - 1)) {
    x<cudaq::ctrl>(qvector[0], qvector[qubit + 1]);
  }
  // If we do not specify measurements, all qubits are measured in
  // the Z-basis by default or we can manually specify it also
  mz(qvector);
}

int main() {
  int qubit_count = 2;
  auto produced_str = cudaq::draw(kernel, qubit_count);
  std::cout << produced_str << std::endl;
  auto result = cudaq::sample(kernel, qubit_count);
  result.dump();
  return 0;
}
// [End Sample]
/* [Begin `SampleOutput`]
     ╭───╮
q0 : ┤ h ├──●──
     ╰───╯╭─┴─╮
q1 : ─────┤ x ├
          ╰───╯

{ 11:506 00:494 }
 [End `SampleOutput`] */

// [Begin Observe]
// Define a Hamiltonian in terms of Pauli Spin operators.
auto hamiltonian = cudaq::spin::z(0) + cudaq::spin::y(1) +
                   cudaq::spin::x(0) * cudaq::spin::z(0);

int qubit_count = 2;
// Compute the expectation value given the state prepared by the kernel.
auto result = cudaq::observe(kernel, hamiltonian, qubit_count).expectation();

std::cout << "<H> =" << result.dump() << std::endl;
// [End Observe]
/* [Begin `ObserveOutput`]
<H> = 0.0
 [End `ObserveOutput`] */

// [Begin `GetState`]
// Compute the statevector of the kernel
cudaq::state result = cudaq::get_state(kernel, qubit_count);

result.dump();
// [End `GetState`]
/* [Begin `GetStateOutput`]
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
 [End `GetStateOutput`] */

// [Begin `ObserveAsync`]
// Measuring the expectation value of 2 different Hamiltonians in parallel
auto hamiltonian_1 = cudaq::spin::x(0) + cudaq::spin::y(1) +
                     cudaq::spin::z(0) * cudaq::spin::y(1);

// Asynchronous execution on multiple `qpus` via nvidia gpus.
auto future = cudaq::observe_async(0, kernel, hamiltonian_1, qubit_count);

auto result_1 = future.get();

// Retrieve results
printf(result_1.expectation());
// [End `ObserveAsync`]
/* [Begin `ObserveAsyncOutput`]
2.220446049250313e-16
[End `ObserveAsyncOutput`] */
