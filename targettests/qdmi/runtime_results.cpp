/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: qdmi
// clang-format off
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default --qdmi-program-format qasm2 %s -o %t
// RUN: %t | FileCheck %s
// clang-format on

#include <cudaq.h>

#include <array>
#include <iostream>
#include <string>

struct wide_sample {
  void operator()() __qpu__ {
    cudaq::qvector qubits(128);
    x(qubits[0]);
    x(qubits[2]);
    x(qubits[64]);
    x(qubits[127]);
    mz(qubits);
  }
};

struct one_state {
  void operator()() __qpu__ {
    cudaq::qubit qubit;
    x(qubit);
  }
};

struct subset_sample {
  void operator()() __qpu__ {
    cudaq::qvector qubits(3);
    x(qubits[0]);
    mz(qubits[2]);
    mz(qubits[0]);
  }
};

struct x_state {
  void operator()() __qpu__ {
    cudaq::qubit qubit;
    h(qubit);
  }
};

struct adjoint_rotations {
  void operator()() __qpu__ {
    cudaq::qvector qubits(4);
    constexpr double angle = 1.5707963267948966;
    h(qubits[0]);
    r1(angle, qubits[0]);
    r1<cudaq::adj>(angle, qubits[0]);
    h(qubits[0]);
    rx(angle, qubits[1]);
    rx<cudaq::adj>(angle, qubits[1]);
    ry(angle, qubits[2]);
    ry<cudaq::adj>(angle, qubits[2]);
    h(qubits[3]);
    rz(angle, qubits[3]);
    rz<cudaq::adj>(angle, qubits[3]);
    h(qubits[3]);
    mz(qubits);
  }
};

struct y_state {
  void operator()() __qpu__ {
    cudaq::qubit qubit;
    h(qubit);
    s(qubit);
  }
};

struct measurement_bases {
  void operator()() __qpu__ {
    cudaq::qvector qubits(2);
    h(qubits);
    s(qubits[1]);
    mx(qubits[0]);
    my(qubits[1]);
  }
};

int main() {
  std::string expected(128, '0');
  constexpr std::array setBits{0U, 2U, 64U, 127U};
  for (const auto index : setBits)
    expected[index] = '1';

  auto samples = cudaq::sample_async(32, 0, wide_sample{});
  auto observation =
      cudaq::observe_async(32, 0, one_state{}, cudaq::spin_op::z(0));
  std::cout << "pattern=" << samples.get().count(expected) << '\n';
  std::cout << "expectation=" << observation.get().expectation() << '\n';
  std::cout << "implicit=" << cudaq::sample(32, one_state{}).count("1") << '\n';
  std::cout << "subset=" << cudaq::sample(32, subset_sample{}).count("10")
            << '\n';
  std::cout << "x="
            << cudaq::observe(32, x_state{}, cudaq::spin_op::x(0)).expectation()
            << '\n';
  std::cout << "y="
            << cudaq::observe(32, y_state{}, cudaq::spin_op::y(0)).expectation()
            << '\n';
  std::cout << "adjoints="
            << cudaq::sample(32, adjoint_rotations{}).count("0000") << '\n';
  std::cout << "measurements="
            << cudaq::sample(32, measurement_bases{}).count("00") << '\n';
}

// CHECK: pattern=32
// CHECK: expectation=-1
// CHECK: implicit=32
// CHECK: subset=32
// CHECK: x=1
// CHECK: y=1
// CHECK: adjoints=32
// CHECK: measurements=32
