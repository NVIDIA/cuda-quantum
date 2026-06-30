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

// [Begin Options Kernel]
// A hyperedge arises naturally when one fault trips both an
// X-type and a Z-type parity check. This circuit prepares a Bell pair |Phi+>,
// the +1 eigenstate of both XX and ZZ, and measures each stabilizer with its
// own ancilla. A Y error anticommutes with both checks and flips the data
// readout, lighting up three detectors at once. Because Y = X * Z, that
// hyperedge decomposes into the separate X and Z edges seeded by the
// accompanying single-qubit errors.
__qpu__ void correlated_checks() {
  cudaq::qvector data(2);
  cudaq::qubit z_anc;
  cudaq::qubit x_anc;

  // Prepare |00> + |11>
  h(data[0]);
  x<cudaq::ctrl>(data[0], data[1]);

  cudaq::apply_noise<cudaq::x_error>(0.01, data[0]);
  cudaq::apply_noise<cudaq::z_error>(0.01, data[0]);
  cudaq::apply_noise<cudaq::y_error>(0.02, data[0]);

  // ZZ parity check: the data qubits control the ancilla.
  x<cudaq::ctrl>(data[0], z_anc);
  x<cudaq::ctrl>(data[1], z_anc);
  auto z_syndrome = mz(z_anc);

  // XX parity check: the ancilla controls the data, read out in the X basis.
  h(x_anc);
  x<cudaq::ctrl>(x_anc, data[0]);
  x<cudaq::ctrl>(x_anc, data[1]);
  h(x_anc);
  auto x_syndrome = mz(x_anc);

  auto final = mz(data);
  cudaq::detector(z_syndrome);
  cudaq::detector(x_syndrome);
  cudaq::detector(final[0], final[1]);
}
// [End Options Kernel]

int main() {
  // [Begin Generate]
  // Generate the detector error model as Stim `.dem` text. A noise model must
  // be supplied for the in-kernel `apply_noise` mechanisms to take effect.
  // Parse the text with Stim (`stim::DetectorErrorModel{dem}`) to drive a
  // decoder.
  cudaq::noise_model noise;
  std::string dem =
      cudaq::dem_from_kernel(memory_experiment, &noise, /*rounds=*/2);
  std::printf("Memory experiment DEM:\n%s\n", dem.c_str());
  // [End Generate]

  // [Begin Options]
  // Pass a cudaq::dem_options struct to control the Stim error analyzer.
  // decompose_errors=true splits hyperedge mechanisms (three or more detectors)
  // into pairs of graphlike edges, which is required by most MWPM decoders.
  std::string dem_raw = cudaq::dem_from_kernel(correlated_checks, &noise);
  cudaq::dem_options opts;
  opts.decompose_errors = true;
  std::string dem_decomposed =
      cudaq::dem_from_kernel(correlated_checks, &noise, opts);
  std::printf("Raw DEM:\n%s\n", dem_raw.c_str());
  std::printf("Decomposed DEM:\n%s\n", dem_decomposed.c_str());
  // [End Options]

  return 0;
}
