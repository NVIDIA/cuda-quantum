/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake -verify %s

#include <bitset>
#include <cudaq.h>
#include <iostream>
#include <random>

template <int nrOfBits>
std::bitset<nrOfBits> random_bits() {

  std::bitset<nrOfBits> randomBits;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  float randNum;
  for (size_t i = 0; i < nrOfBits; i++) {
    randNum = distribution(generator);
    if (randNum < 0.5) {
      randNum = 0;
    } else {
      randNum = 1;
    }
    randomBits.set(i, randNum);
  }
  return randomBits;
}

template <int nrOfBits>
struct oracle {
  // expected-error@+1{{kernel argument type not supported}}
  auto operator()(std::bitset<nrOfBits> bitvector, cudaq::qview<> qs,
                  cudaq::qubit &aux) __qpu__ {

    for (size_t i = 0; i < nrOfBits; i++) {
      if (bitvector[i] & 1) {
        x<cudaq::ctrl>(qs[nrOfBits - i - 1], aux);
      }
    }
  }
};

template <int nrOfBits>
struct bernstein_vazirani {
  // expected-error@+1{{kernel argument type not supported}}
  auto operator()(std::bitset<nrOfBits> bitvector) __qpu__ {

    cudaq::qarray<nrOfBits> qs;
    cudaq::qubit aux;
    h(aux);
    z(aux);
    h(qs);

    oracle<nrOfBits>{}(bitvector, qs, aux);
    h(qs);
    mz(qs);
  }
};

int main() {
  // The number of qubits can be increased when targeting the `nvidia-mgpu`
  // backend.
  const int nr_qubits = 5;
  auto bitvector = random_bits<nr_qubits>();
  auto kernel = bernstein_vazirani<nr_qubits>{};
  auto counts = cudaq::sample(kernel, bitvector);

  if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
    printf("Encoded bitstring:  %s\n", bitvector.to_string().c_str());
    printf("Measured bitstring: %s\n\n", counts.most_probable().c_str());

    for (auto &[bits, count] : counts) {
      printf("observed %s (probability %u%%)\n", bits.data(),
             100 * (uint)((double)count / 1000.));
    }
  }

  return 0;
}
