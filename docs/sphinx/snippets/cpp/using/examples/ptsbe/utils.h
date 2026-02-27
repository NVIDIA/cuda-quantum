/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cudaq.h>

auto bell = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  mz(q);
};

inline cudaq::noise_model bell_noise_model() {
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::h>({0}, cudaq::depolarization_channel(0.01));
  noise.add_channel<cudaq::types::x>({0, 1}, cudaq::depolarization2(0.005));
  return noise;
}