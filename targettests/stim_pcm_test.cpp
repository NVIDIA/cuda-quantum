/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with this command:
// nvq++ --target stim stim_pcm_test.cpp
// CUDAQ_TIMING_TAGS=0-7 ./a.out
// BMH_NUM_SHOTS=100 CUDAQ_LOG_LEVEL=info BF_NOISE=0.1 BMH_NUM_QUBITS=5 BMH_NUM_ROUNDS=3 ./a.out

#include "cudaq.h"
#include <cudaq/algorithms/draw.h>

struct stress_test {
  void operator()(int num_qubits, int num_rounds) __qpu__ {
    cudaq::qvector q(num_qubits);
    for (int round = 0; round < num_rounds; round++) {
      h(q[0]);
      for (int qi = 1; qi < num_qubits; qi++)
        x<cudaq::ctrl>(q[qi - 1], q[qi]);
      mz(q);
      for (int qi = 0; qi < num_qubits; qi++)
        reset(q[qi]);
    }
  }
};

int main(int argc, char *argv[]) {
  const double noise_bf_prob = []() {
    if (auto *ch = getenv("BF_NOISE"))
      return atof(ch);
    return 0.0;
  }();
  cudaq::noise_model noise;
  cudaq::bit_flip_channel bf(noise_bf_prob);
  for (std::size_t i = 0; i < 100; i++)
    noise.add_channel("mz", {i}, bf);
  noise.add_all_qubit_channel("x", cudaq::depolarization2(noise_bf_prob),
                              /*num_controls=*/1);
  cudaq::set_noise(noise);
  int num_qubits = 10;
  int num_rounds = 10;
  int num_shots = 10;
  if (auto *ch = getenv("BMH_NUM_QUBITS"))
    num_qubits = atoi(ch);
  if (auto *ch = getenv("BMH_NUM_ROUNDS"))
    num_rounds = atoi(ch);
  if (auto *ch = getenv("BMH_NUM_SHOTS"))
    num_shots = atoi(ch);
  bool doTracer = false;
  if (auto *ch = getenv("BMH_DO_TRACER"))
    doTracer = atoi(ch) > 0;

  // Stage 1 - get the PCM size by running with "experimental_pcm_size". The
  // result will be returned in ctx_pcm_size.shots.
  cudaq::ExecutionContext ctx_pcm_size("pcm_size");
  ctx_pcm_size.noiseModel = &noise;
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_pcm_size);
  stress_test{}(num_qubits, num_rounds);
  platform.reset_exec_ctx();

  // Stage 2 - get the PCM using the ctx_pcm_size.shots value.
  cudaq::ExecutionContext ctx_pcm("pcm");
  ctx_pcm.noiseModel = &noise;
  ctx_pcm.pcm_dimensions = ctx_pcm_size.pcm_dimensions;
  platform.set_exec_ctx(&ctx_pcm);
  stress_test{}(num_qubits, num_rounds);
  platform.reset_exec_ctx();

  // The PCM is now stored in ctx_pcm.result. More precisely, the unfiltered
  // PCM is stored there, but some post-processing may be required to
  // eliminate duplicate columns.
  auto pcm_as_strings = ctx_pcm.result.sequential_data();
  printf("Columns of PCM:\n");
  for (int col = 0; auto x : pcm_as_strings) {
    printf("Column %02d (Prob %.6f): %s\n", col,
           ctx_pcm.pcm_probabilities.value()[col], x.c_str());
    col++;
  }
  for (auto &[k, v] : ctx_pcm.result.to_map()) {
    if (v > 1) {
      printf("Key %s found with >1 entries (%lu)\n", k.c_str(), v);
    }
  }

  // Stage 3 - now sample the same circuit using the same noise model.
  auto counts = cudaq::sample({.shots = static_cast<std::size_t>(num_shots),
                               .noise = noise,
                               .explicit_measurements = true},
                              stress_test{}, num_qubits, num_rounds);
//   counts.dump();
  auto seq_data = counts.sequential_data();
  printf("seq_data.size() = %lu and seq_data[0].size() = %lu\n",
         seq_data.size(), seq_data[0].size());
  for (std::size_t shot_ix = 0; auto &shot_meas : seq_data)
    printf("Shot %02lu : %s\n", shot_ix++, shot_meas.c_str());
  printf("Done\n");
  return 0;
}
