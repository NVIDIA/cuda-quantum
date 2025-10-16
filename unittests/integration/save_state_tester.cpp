/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/builder/kernels.h"
#include <cudaq.h>
#include <iostream>

#if defined(CUDAQ_BACKEND_STIM)
CUDAQ_TEST(SaveStateTester, checkStimState) {

  struct multi_round_kernel {
    void operator()(int num_qubits, int num_rounds,
                    double noise_probability) __qpu__ {
      cudaq::qvector q(num_qubits);
      for (int round = 0; round < num_rounds; round++) {
        h(q[0]);
        for (int qi = 0; qi < num_qubits; qi++)
          cudaq::apply_noise<cudaq::depolarization_channel>(noise_probability,
                                                            q[qi]);

        cudaq::save_state();
      }
      mz(q);
      for (int qi = 0; qi < num_qubits; qi++)
        reset(q[qi]);
    }
  };

  int num_qubits = 5;
  int num_rounds = 3;
  double noise_bf_prob = 0.0625;

  cudaq::noise_model noise;
  cudaq::bit_flip_channel bf(noise_bf_prob);
  for (std::size_t i = 0; i < num_qubits; i++)
    noise.add_channel("mz", {i}, bf);
  cudaq::set_noise(noise);

  cudaq::ExecutionContext ctx_msm_size("msm_size");
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_msm_size);
  multi_round_kernel{}(num_qubits, num_rounds, noise_bf_prob);
  platform.reset_exec_ctx();

  cudaq::ExecutionContext ctx_msm("msm");
  ctx_msm.noiseModel = &noise;
  ctx_msm.msm_dimensions = ctx_msm_size.msm_dimensions;
  platform.set_exec_ctx(&ctx_msm);
  multi_round_kernel{}(num_qubits, num_rounds, noise_bf_prob);
  platform.reset_exec_ctx();

  // accessing the execution context to get the recorded states
  assert(ctx_msm.get_recorded_states().size() == num_rounds &&
         "Expected 3 state snapshots");

  for (std::size_t round = 0; round < ctx_msm.get_recorded_states().size();
       ++round) {
    const auto &state = ctx_msm.get_recorded_states()[round];

    assert(state.getNumQubits() == num_qubits &&
           "Number of qubits must equal num_qubits");
    /*
    for (std::size_t qi = 0; qi < state.getNumQubits(); ++qi) {
        const auto &x_row = state.getTableau().x_output[qi];
        const auto &z_row = state.getTableau().z_output[qi];

        // Each row should have the same number of qubits.
        assert(x_row.num_qubits == state.getNumQubits());
        assert(z_row.num_qubits == state.getNumQubits());

        // Check that each character in the row is a valid Pauli symbol.
        for (char c : x_row.str()) {
            assert((c == 'I' || c == 'X' || c == 'Y' || c == 'Z') &&
                  "Invalid symbol in tableau X-output row");
        }
        for (char c : z_row.str()) {
            assert((c == 'I' || c == 'X' || c == 'Y' || c == 'Z') &&
                  "Invalid symbol in tableau Z-output row");
        }
    }
    const auto &frame = state.getPauliFrame();

    // Frame size should match number of qubits.
    assert(frame.size() == state.getNumQubits());

    // Each frame entry must be a valid Pauli operator.
    for (char c : frame) {
        assert((c == 'I' || c == 'X' || c == 'Y' || c == 'Z') &&
              "Invalid symbol in Pauli frame");
    }
  }
  */
  }
}
#endif
