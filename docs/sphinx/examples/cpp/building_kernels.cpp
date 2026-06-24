/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Definition]
#include <cudaq.h>

using namespace std::complex_literals;
using complex = std::complex<cudaq::real>;

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
__qpu__ void kernel(const std::vector<complex> &vec) { cudaq::qubit q; }
// [End `PassingComplexVector`]

// [Begin `CapturingComplexVector`]
// Capturing complex vectors
__qpu__ void kernel0(const std::vector<complex> &vec) { cudaq::qvector q(vec); }

void function0() {
  std::vector<complex> d = {
      {0.70710678, 0}, {0.0, 0.0}, {0.0, 0.0}, {0.70710678, 0.0}};
  kernel0(d);
}
// [End `CapturingComplexVector`]

// [Begin `PrecisionAgnosticAPI`]
// Precision-Agnostic API
__qpu__ void kernel1(const std::vector<cudaq::complex> &e) {
  cudaq::qvector q(e);
}

void function1() {
  auto e = {
      cudaq::complex{0.70710678, 0}, {0.0, 0.0}, {0.0, 0.0}, {0, 0.70710678}};
  kernel1(e);
}
// [End `PrecisionAgnosticAPI`]

// [Begin `AllQubits`]
__qpu__ void kernel2() {
  cudaq::qvector r(10);
  cudaq::h(r);
}
// [End `AllQubits`]

// [Begin `IndividualQubits`]

__qpu__ void kernel3() {
  cudaq::qvector r(10);
  cudaq::h(r[0]); // first qubit
  cudaq::h(r[9]); // last qubit
}
// [End `IndividualQubits`]

// [Begin `ControlledOperations`]
__qpu__ void kernel4() {
  cudaq::qvector r(10);
  x<cudaq::ctrl>(r[0], r[1]); // CNOT gate applied with qubit 0 as control
}
// [End `ControlledOperations`]

// [Begin `MultiControlledOperations`]
__qpu__ void kernel5() {
  cudaq::qvector r(10);
  x<cudaq::ctrl>(r[0], r[1]); // CNOT gate applied with qubit 0 and 1 as control
}
// [End `MultiControlledOperations`]

// [Begin `ControlledKernel`]
__qpu__ void x_kernel(cudaq::qubit &q) { x(q); }

// A kernel that will call `x_kernel` as a controlled operation.
__qpu__ void kernel6() {
  cudaq::qvector control_vector(2);
  cudaq::qubit target;

  x(control_vector);
  x(target);
  x(control_vector[1]);
  cudaq::control(x_kernel, control_vector, target);
}

// The above is equivalent to:
__qpu__ void kernel7() {
  cudaq::qvector qvector(3);

  x(qvector);
  x(qvector[1]);
  x<cudaq::ctrl>(qvector[0], qvector[1]);
  mz(qvector);
}

int main() {
  auto results = cudaq::sample(kernel7);
  results.dump();
}
// [End `ControlledKernel`]

// [Begin `AdjointOperations`]
__qpu__ void kernel_t(cudaq::qvector<> &qubits, double theta) {
  ry(theta, qubits[0]);
  h<cudaq::ctrl>(qubits[0], qubits[1]);
  x(qubits[1]);
}

__qpu__ void kernel8() {
  cudaq::qvector<> r(10);
  cudaq::adjoint(kernel_t, r, 0.0);
}
// [End `AdjointOperations`]

// [Begin `BuildingKernelsWithKernels`]
__qpu__ void kernel_A(cudaq::qubit &q0, cudaq::qubit &q1) {
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
__qpu__ void kernel9(std::vector<double> thetas) {
  cudaq::qvector qubits(2);
  rx(thetas[0], qubits[0]);
  ry(thetas[1], qubits[1]);
}

std::vector<double> thetas = {.024, .543};
// kernel9(thetas);
// [End `ParameterizedKernels`]
