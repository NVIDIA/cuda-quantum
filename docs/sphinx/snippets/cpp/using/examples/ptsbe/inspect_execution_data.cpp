/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin PTSBE_Execution_Data]
#include <utils.h>
#include <cstdio>
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/ShotAllocationStrategy.h"

int main() {
    cudaq::ptsbe::sample_options opts;
    opts.shots  = 10000;
    opts.noise  = bell_noise_model();
    opts.ptsbe.return_execution_data = true;

    auto result = cudaq::ptsbe::sample(opts, bell);

    if (result.has_execution_data()) {
        const auto &data = result.execution_data();
        for (const auto &trajectory : data.trajectories)
            printf("id=%zu  p=%.4f  shots=%zu\n",
                    trajectory.trajectory_id, trajectory.probability, trajectory.num_shots);
    }   
}
// [End PTSBE_Execution_Data]