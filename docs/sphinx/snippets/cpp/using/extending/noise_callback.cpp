// ============================================================================ #
// Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// Dynamic Noise Channel Callback (C++)
// This example shows how to create a noise channel that depends on
// the gate operands and parameters.

#include "cudaq.h"

int main() {
    cudaq::NoiseModel noise;

    // Add a dynamic noise channel to the 'rx' gate using a lambda
    noise.add_channel("rx",
        [](const std::vector<std::size_t> &qubits,
           const std::vector<double> &params) -> cudaq::kraus_channel {
            // Example: Stronger noise for larger rotation angles
            double angle = params.empty() ? 0.0 : params[0];
            double noise_strength = std::min(std::abs(angle) / 3.14159 * 0.1, 0.1);

            // Return a depolarization channel with angle-dependent strength
            return cudaq::DepolarizationChannel(noise_strength);
        });

    cudaq::set_noise(noise);

    auto kernel = [](double theta) {
        auto q = cudaq::qvector(1);
        cudaq::rx(theta, q[0]);
        cudaq::mz(q);
    };

    // The noise strength will vary based on the rotation angle
    for (double angle : {0.1, 0.5, 1.0, 2.0, 3.14}) {
        cudaq::set_noise(noise);
        auto counts = cudaq::sample(kernel, angle, 1000);
        std::cout << "Angle: " << angle << ", Counts: " << counts << std::endl;
    }

    cudaq::unset_noise();
    return 0;
}