/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// nvq++ dem_from_kernel.cpp -o dem && ./dem

// [Begin Docs]
#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithms/dem.h>
// [End Docs]

// [Begin Kernel]
// A 3-qubit bit-flip memory experiment. Each round measures the data qubits;
// cross-round detectors pair each measurement with its value in the previous
// round, and a final logical observable reads out the register. In-kernel
// `apply_noise` seeds the error mechanisms the detector error model reports.
__qpu__ void memory_experiment(int rounds) {
  cudaq::qvector data(3);
  auto prev = mz(data);

  for (int r = 0; r < rounds; ++r) {
    cudaq::apply_noise<cudaq::x_error>(0.01, data[0]);
    cudaq::apply_noise<cudaq::x_error>(0.01, data[1]);
    cudaq::apply_noise<cudaq::x_error>(0.01, data[2]);

    auto curr = mz(data);
    // One detector per qubit, pairing this round with the previous one.
    cudaq::detectors(prev, curr);
    prev = curr;
  }
  cudaq::logical_observable(prev[0], prev[1], prev[2]);
}
// [End Kernel]

int main() {
  // [Begin Generate]
  // Generate the detector error model as Stim `.dem` text. A noise model must
  // be supplied for the in-kernel `apply_noise` mechanisms to take effect.
  // Parse the text with Stim (`stim::DetectorErrorModel{dem}`) to drive a
  // decoder.
  cudaq::noise_model noise;
  std::string dem =
      cudaq::dem_from_kernel(memory_experiment, &noise, /*rounds=*/2);
  std::printf("%s\n", dem.c_str());
  // [End Generate]
  return 0;
}
