/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/AnalogHamiltonian.h"

const std::string sampleTimeEvolutionPayload = R"(
    {
        "type": "time_evolution",
        "payload": {
            "hamiltonian": {
                "Hinit": {
                    "X0": 0.8,
                    "X1": 0.6,
                    "X0X1": 0.4
                },
                "Hinter": {
                    "Z0": 1.0,
                    "Z1": 0.9,
                    "Z0Z1": 0.5
                },
                "Hfinal": {
                    "Z0": 1.2,
                    "Z1": 1.0,
                    "Z0Z1": 0.7
                }
            },
            "schedules": [
                {
                    "Hinit": [1.0, 0.75, 0.5, 0.25, 0.0],
                    "Hinter": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "Hfinal": [0.0, 0.0, 0.0, 0.0, 0.0]
                },
                {
                    "Hinit": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "Hinter": [1.0, 0.75, 0.5, 0.25, 0.0],
                    "Hfinal": [0.0, 0.25, 0.5, 0.75, 1.0]
                }
            ],
            "observables": [
                {
                    "Z0": 1.0,
                    "Z1": 0.8,
                    "Z0Z1": 0.6
                },
                {
                    "X0": 0.9,
                    "X1": 0.7,
                    "X0X1": 0.5
                }
            ],
            "initial_state": [
                {"real": 0.7071, "imag": 0.0},
                {"real": 0.0, "imag": 0.0},
                {"real": 0.0, "imag": 0.0},
                {"real": 0.7071, "imag": 0.0}
            ]
        }
    }
)";

CUDAQ_TEST(QilimanjaroTester, checkTimeEvolutionJson) {
    cudaq::spin_op Hinit = 0.8 * cudaq::spin::x(0) + 0.6 * cudaq::spin::x(1) + 0.4 * cudaq::spin::x(0) * cudaq::spin::x(1);
    cudaq::spin_op Hinter = 1.0 * cudaq::spin::z(0) + 0.9 * cudaq::spin::z(1) + 0.5 * cudaq::spin::z(0) * cudaq::spin::z(1);
    cudaq::spin_op Hfinal = 1.2 * cudaq::spin::z(0) + 1.0 * cudaq::spin::z(1) + 0.7 * cudaq::spin::z(0) * cudaq::spin::z(1);
    
    std::vector<std::complex<double>> psi_data = {
        {0.7071, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.7071, 0.0}
    };
    cudaq::state psi = cudaq::state::from_data(psi_data);
}