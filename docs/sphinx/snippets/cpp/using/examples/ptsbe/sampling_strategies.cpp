/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin PTSBE_Sampling]
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/strategies/ProbabilisticSamplingStrategy.h"
#include "cudaq/ptsbe/strategies/OrderedSamplingStrategy.h"

int main() {
    // Reproducible probabilistic sampling
    cudaq::ptsbe::sample_options opts;
    opts.ptsbe.strategy =
        std::make_shared<cudaq::ptsbe::ProbabilisticSamplingStrategy>(/*seed=*/42);

    // Top-100 trajectories
    opts.ptsbe.max_trajectories = 100;
    opts.ptsbe.strategy =
        std::make_shared<cudaq::ptsbe::OrderedSamplingStrategy>();
}
// [End PTSBE_Sampling]