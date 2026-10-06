/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target density-matrix-cpu %s -o %t && %t | FileCheck %s
// clang-format on

// `cudaq::apply_noise` with a `cudaq::qvector` argument applies the channel to
// every qubit of the register. Each channel here flips its targets with
// probability 1, so every shot has the same outcome.

#include <cudaq.h>
#include <iostream>

// `x_error` takes exactly one target, so the two-qubit register needs a
// two-target channel. Its only Kraus operator is the Pauli XX.
struct xx_flip : public cudaq::kraus_channel {
  static constexpr std::size_t num_parameters = 0;
  static constexpr std::size_t num_targets = 2;
  xx_flip(const std::vector<cudaq::real> &) {
    std::vector<cudaq::complex> xx{0, 0, 0, 1, 0, 0, 1, 0,
                                   0, 1, 0, 0, 1, 0, 0, 0};
    push_back(cudaq::kraus_op(xx));
    validateCompleteness();
    generateUnitaryParameters();
  }
  REGISTER_KRAUS_CHANNEL("xx_flip");
};

struct x_error_on_qubit {
  void operator()() __qpu__ {
    cudaq::qvector q(1);
    cudaq::apply_noise<cudaq::x_error>(1.0, q[0]);
  }
};

struct x_error_on_qvector {
  void operator()() __qpu__ {
    cudaq::qvector q(1);
    cudaq::apply_noise<cudaq::x_error>(1.0, q);
  }
};

struct xx_flip_on_qvector {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    cudaq::apply_noise<xx_flip>(q);
  }
};

int main() {
  cudaq::noise_model noise;
  noise.register_channel<xx_flip>();
  const cudaq::sample_options options{.shots = 100, .noise = noise};

  std::cout << "x_error on qubit: " << std::flush;
  cudaq::sample(options, x_error_on_qubit{}).dump();
  std::cout << "x_error on qvector(1): " << std::flush;
  cudaq::sample(options, x_error_on_qvector{}).dump();
  std::cout << "xx_flip on qvector(2): " << std::flush;
  cudaq::sample(options, xx_flip_on_qvector{}).dump();
  return 0;
}

// CHECK: x_error on qubit: { 1:100 }
// CHECK: x_error on qvector(1): { 1:100 }
// CHECK: xx_flip on qvector(2): { 11:100 }
