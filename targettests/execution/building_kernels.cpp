/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Simulators
// RUN: nvq++ %cpp_std --enable-mlir  %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --library-mode %s -o %t && %t | FileCheck %s

// Quantum emulators
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target anyon                    --emulate %s -o %t && %t | FileCheck %s
// 2 different IQM machines for 2 different topologies
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Apollo --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ %cpp_std --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std::complex_literals;
using complex = std::complex<cudaq::real>;

void kernel() __qpu__ {
  cudaq::qubit A;
  cudaq::qvector B(2);
}

void kernel0(int N) __qpu__ { cudaq::qvector r(N); }

// Passing complex vectors
void kernel1(const std::vector<complex> &vec) __qpu__ { cudaq::qvector q(vec); }

// Precision-Agnostic API
void kernel2(const std::vector<cudaq::complex> &e) __qpu__ {
  cudaq::qvector q(e);
}

void kernel3() __qpu__ {
  cudaq::qvector r(3);
  cudaq::h(r);
}

void kernel4() __qpu__ {
  cudaq::qvector r(3);
  cudaq::h(r[0]); // first qubit
  cudaq::h(r[2]); // last qubit
}

void kernel5() __qpu__ {
  cudaq::qvector r(2);
  cudaq::h(r[0]);
  x<cudaq::ctrl>(r[0], r[1]); // CNOT gate applied with qubit 0 as control
}

void kernel6() __qpu__ {
  cudaq::qvector r(3);
  cudaq::h(r[0]);
  cudaq::h(r[1]);
  x<cudaq::ctrl>(r[0], r[1], r[2]); // CNOT gate applied with qubit 0 and 1 as control (Toffoli gate)
}

void x_kernel(cudaq::qubit &q) __qpu__ { x(q); }

// A kernel that will call `x_kernel` as a controlled operation.
void kernel7() __qpu__ {
  cudaq::qvector control_vector(2);
  cudaq::qubit target;

  x(control_vector);
  x(target);
  x(control_vector[1]);
  cudaq::control(x_kernel, control_vector, target);
}

// The above is equivalent to:
void kernel8() __qpu__ {
  cudaq::qvector qvector(3);

  x(qvector);
  x(qvector[1]);
  x<cudaq::ctrl>(qvector[0], qvector[1]);
  mz(qvector);
}

__qpu__ void kernel_t(cudaq::qvector<> &qubits, double theta) {
  ry(theta, qubits[0]);
  h<cudaq::ctrl>(qubits[0], qubits[1]);
  x(qubits[1]);
}

__qpu__ void kernel9() {
  cudaq::qvector<> r(3);
  cudaq::adjoint(kernel_t, r, 0.0);
}

__qpu__ void kernel_A(cudaq::qubit &q0, cudaq::qubit &q1) {
  x<cudaq::ctrl>(q0, q1);
}

__qpu__ void kernel_B() {
  cudaq::qvector reg(10);
  for (int i = 0; i < 5; i++) {
    kernel_A(reg[i], reg[i + 1]);
  }
}

__qpu__ void kernel10(std::vector<double> thetas) {
  cudaq::qvector qubits(2);
  rx(thetas[0], qubits[0]);
  ry(thetas[1], qubits[1]);
}

std::vector<double> thetas = {.024, .543};

void printCounts(cudaq::sample_result &result) {
std::vector<std::string> values{};
for (auto &&[bits, counts] : result) {
    values.push_back(bits);
}

std::sort(values.begin(), values.end());
for (auto &&bits : values) {
    std::cout << bits << std::endl;
}
}

int main() {
  std::vector<complex> d = {
    {0.70710678, 0}, {0.0, 0.0}, {0.0, 0.0}, {0.70710678, 0.0}};
  auto e = {
    cudaq::complex{0.70710678, 0}, {0.0, 0.0}, {0.0, 0.0}, {0, 0.70710678}};
  {
    std::cout << "Building kernel (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel (kernel mode)
  // CHECK: 000000000
  // clang-format on

  {
    std::cout << "Building kernel with an argument (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel0, 10);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with an argument (kernel mode)
  // CHECK: 0000000000
  // clang-format on

  {
    std::cout << "Building kernel with passing complex vector (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel1, d);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with passing complex vector (kernel mode)
  // CHECK: 00
  // CHECK: 01
  // clang-format on

  {
    std::cout << "Building kernel with a precision agnostic API (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel2, e);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with a precision agnostic API (kernel mode)
  // CHECK: 00
  // CHECK: 01
  // clang-format on

  {
    std::cout << "Building kernel with a Hadamard gate (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel3);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with a Hadamard gate (kernel mode)
  // CHECK: 000
  // CHECK: 001
  // CHECK: 010
  // CHECK: 011
  // CHECK: 100
  // CHECK: 101
  // CHECK: 110
  // CHECK: 111
  // clang-format on

  {
    std::cout << "Building kernel with a Hadamard gate on the first and the last qubit (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel4);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with a Hadamard gate on the first and the last qubit (kernel mode)
  // CHECK: 000
  // CHECK: 001
  // CHECK: 100
  // CHECK: 101
  // clang-format on

  {
    std::cout << "Building kernel with a CNOT gate (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel5);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with a CNOT gate (kernel mode)
  // CHECK: 00
  // CHECK: 11
  // clang-format on

  {
    std::cout << "Building kernel with a CNOT gate on the first and the last qubit (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel6);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with a CNOT gate on the first and the last qubit (kernel mode)
  // CHECK: 000
  // CHECK: 010
  // CHECK: 100
  // CHECK: 111
  // clang-format on

  {
    std::cout << "Building kernel with a nested kernel (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel8);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with a nested kernel (kernel mode)
  // CHECK: 111
  // clang-format on

  {
    std::cout << "Building kernel with a nested kernel with theta (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel9);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with a nested kernel (kernel mode)
  // CHECK: 010
  // clang-format on

  {
    std::cout << "Building kernel with a double vector (kernel mode)" << std::endl;
    auto counts = cudaq::sample(kernel10, thetas);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Building kernel with a double vector (kernel mode)
  // CHECK: 00
  // CHECK: 01
  // clang-format on
}
