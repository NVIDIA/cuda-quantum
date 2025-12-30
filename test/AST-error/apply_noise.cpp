/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify -D CUDAQ_REMOTE_SIM=1 %s

#include <cudaq.h>

struct SantaKraus : public cudaq::kraus_channel {
  constexpr static std::size_t num_parameters = 0;
  constexpr static std::size_t num_targets = 2;
  static std::size_t get_key() { return (std::size_t)&get_key; }
  SantaKraus() {}
};

struct testApplyNoise {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    // expected-error@+1{{no matching function for call to 'apply_noise'}}
    cudaq::apply_noise<SantaKraus>(q0, q1);
    // expected-note@* 2-3 {{}}
  }
};
