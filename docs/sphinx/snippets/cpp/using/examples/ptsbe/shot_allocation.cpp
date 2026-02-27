/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin PTSBE_Shot_Allocation]
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/ShotAllocationStrategy.h"
#include "utils.h"

int main() {
  cudaq::ptsbe::sample_options opts;
  opts.ptsbe.shot_allocation = cudaq::ptsbe::ShotAllocationStrategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
      /*bias=*/2.0);

  auto result = cudaq::ptsbe::sample(opts, bell);
  result.dump();
}
// [End PTSBE_Shot_Allocation]
