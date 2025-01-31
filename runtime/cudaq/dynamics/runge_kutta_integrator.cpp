/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/runge_kutta_integrator.h"
#include <iostream>

using namespace cudaq;

namespace cudaq {
void runge_kutta_integrator::integrate(double target_time) {
    if (!stepper) {
        throw std::runtime_error("Time stepper is not initialized.");
    }

    double dt = integrator_options["dt"];
    if (dt <= 0) {
        throw std::invalid_argument("Invalid time step size for integration.");
    }

    auto handle = state.get_handle();
    auto hilbertSpaceDims = state.get_hilbert_space_dims();

    while (t < target_time) {
        double step_size = std::min(dt, target_time - 1);

        std::cout << "Runge-Kutta step at time " << t << " with step size: " << step_size << std::endl;

        // Empty vectors of same size as state.get_raw_data()
        std::vector<std::complex<double>> zero_state(state.get_raw_data().size(), {0.0, 0.0});

        cudm_state k1(handle, zero_state, hilbertSpaceDims);
        cudm_state k2(handle, zero_state, hilbertSpaceDims);
        cudm_state k3(handle, zero_state, hilbertSpaceDims);
        cudm_state k4(handle, zero_state, hilbertSpaceDims);

        if (substeps_ == 1) {
            // Euler method (1st order)
            k1 = stepper->compute(state, t, step_size);
            state = k1;
        } else if (substeps_ == 2) {
            // Midpoint method (2nd order)
            k1 = stepper->compute(state, t, step_size / 2.0);
            k2 = stepper->compute(k1, t + step_size / 2.0, step_size);
            state = (k1 + k2) * 0.5;
        } else if (substeps_ == 4) {
            // Runge-Kutta method (4th order)
            k1 = stepper->compute(state, t, step_size / 2.0);
            k2 = stepper->compute(k1, t + step_size / 2.0, step_size / 2.0);
            k3 = stepper->compute(k2, t + step_size / 2.0, step_size);
            k4 = stepper->compute(k3, t + step_size, step_size);
            state = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (1.0 / 6.0);
        }

        // Update time
        t += step_size;
    }

    std::cout << "Integration complete. Final time: " << t << std::endl;
}
}
