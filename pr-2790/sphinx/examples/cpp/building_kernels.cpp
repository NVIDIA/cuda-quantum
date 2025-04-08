/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Definition]
#include <cudaq.h>

__qpu__ void kernel() {
  cudaq::qubit A;
  cudaq::qvector B(3);
  cudaq::qvector C(5);
}
// [End Definition]

// [Begin `InputDefinition`]
int N = 2;

__qpu__ void kernel(int N) { cudaq::qvector r(N); }
// [End `InputDefinition`]

// [Begin `PassingComplexVector`]
// Passing complex vectors as parameters
auto c = {.707 + 0j, 0 - .707j};

__qpu__ void kernel(std::vector<std::complex<double>> vec) {
  cudaq::qubit q(vec);
}
// [End `PassingComplexVector`]

// [Begin `CapturingComplexVector`]
// Capturing complex vectors
std::vector<std::complex<double>> c = {0.70710678 + 0j, 0., 0., 0.70710678};

__qpu__ void kernel() { cudaq::qvector q(c); }
// [End `CapturingComplexVector`]

// [Begin `PrecisionAgnosticAPI`]
// Precision-Agnostic API
auto c = cudaq::complex{0.70710678 + 0j, 0., 0., 0.70710678};

__qpu__ void kernel() { cudaq::qvector q(c); }
// [End `PrecisionAgnosticAPI`]

// [Begin `AllQubits`]
__qpu__ void kernel() {
  cudaq::qvector r(10);
  cudaq::h(r);
}
// [End `AllQubits`]

// [Begin `IndividualQubits`]

__qpu__ void kernel() {
  cudaq::qvector r(10);
  cudaq::h(r[0]); // first qubit
  cudaq::h(r[9]); // last qubit
}
// [End `IndividualQubits`]

// [Begin `ControlledOperations`]
__qpu__ void kernel() {
  cudaq::qvector r(10);
  x<cudaq::ctrl>(r[0], r[1]); // CNOT gate applied with qubit 0 as control
}
// [End `ControlledOperations`]

// [Begin `MultiControlledOperations`]
__qpu__ void kernel() {
  cudaq::qvector r(10);
  x<cudaq::ctrl>({r[0], r[1]},
                 r[2]); // CNOT gate applied with qubit 0 and 1 as control
}
// [End `MultiControlledOperations`]

// [Begin `ControlledKernel`]
__qpu__ void x_kernel(cudaq::qubit q) { x(q); }

// A kernel that will call `x_kernel` as a controlled operation.
__qpu__ void kernel() {
  cudaq::qvector control_vector(2);
  cudaq::qubit target;

  x(control_vector);
  x(target);
  x(control_vector[1]);
  cudaq::control(x_kernel, control_vector, target);
}

// The above is equivalent to:
__qpu__ void kernel() {
  cudaq::qvector qvector(3);

  x(qvector);
  x(qvector[1]);
  x<cudaq::ctrl>({qvector[0], qvector[1]}, qvector[2]);
  mz(qvector);
}

int main() {
  auto results = cudaq::sample(kernel);
  results.dump();
}
// [End `ControlledKernel`]

// [Begin `AdjointOperations`]
__qpu__ void kernel() {
  cudaq::qvector r(10);
  cudaq::adjoint(t, r[0]);
}
// [End `AdjointOperations`]

// [Begin `BuildingKernelsWithKernels`]
__qpu__ void kernel_A(cudaq::qubit q0, cudaq::qubit q1) {
  x<cudaq::ctrl>(q0, q1);
}

__qpu__ void kernel_B() {
  cudaq::qvector reg(10);
  for (int i = 0; i < 5; i++) {
    kernel_A(reg[i], reg[i + 1]);
  }
}
// [End `BuildingKernelsWithKernels`]

// [Begin `ParameterizedKernels`]
__qpu__ void kernel(std::vector<double> thetas) {
  cudaq::qvector qubits(2);
  rx(thetas[0], qubits[0]);
  ry(thetas[1], qubits[1]);
}

std::vector<double> thetas = {.024, .543};
kernel(thetas);
// [End `ParameterizedKernels`]
