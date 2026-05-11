/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin PTSBE_Bell]
#include "cudaq/ptsbe/PTSBESample.h"
#include "utils.h"

int main() {
  cudaq::ptsbe::sample_options opts;
  opts.shots = 10000;
  opts.noise = bell_noise_model();

  auto result = cudaq::ptsbe::sample(opts, bell);
  result.dump();
}
// [End PTSBE_Bell]
