/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/AnalogHamiltonian.h"


const std::string sampleSequence = R"(
{
  "setup": {
    "ahs_register": {
      "sites": [["0.0", "0.0"], ["0.0", "0.000006"], ["0.000006", "0.0"]],
      "filling": [1, 1, 1]
    }
  },
  "hamiltonian": {
    "drivingFields": [
      {
        "amplitude": {
          "time_series": {
            "values": ["0.0", "10700000.0", "10700000.0", "0.0"],
            "times": ["0.0", "0.000001", "0.000002", "0.000003"]
          },
            "pattern": "uniform"
        },
        "phase": {
          "time_series": {"values": ["0.0"], "times": ["0.0"]},
          "pattern": "uniform"
        },
        "detuning": {
          "time_series": {
            "values": ["-5400000.0", "5400000.0"],
            "times": ["0.0", "0.000003"]
          },
          "pattern": "uniform"
        }
      }
    ],
    "localDetuning": []
  }
})";

const std::string resnposeSample = R"(
{
  "code": 200,
  "data": {
    "open": false,
    "created_at": "2021-11-10T15:24:38.155824",
    "device_type": "MOCK_DEVICE",
    "project_id": "00000000-0000-0000-0000-000000000001",
    "id": "00000000-0000-0000-0000-000000000001",
    "priority": 10,
    "sequence_builder": "pulser_test_sequence",
    "status": "DONE",
    "updated_at": "2021-11-10T15:27:44.110274",
    "user_id": "EQZj1ZQE",
    "webhook": "10.0.1.5",
    "jobs": [
      {
        "batch_id": "00000000-0000-0000-0000-000000000001",
        "id": "00000000-0000-0000-0000-000000022010",
        "project_id": "00000000-0000-0000-0000-000000022111",
        "runs": 50,
        "status": "DONE",
        "created_at": "2021-11-10T15:27:06.698066",
        "errors": [],
        "result": { "1001": 12, "0110": 35, "1111": 1 },
        "full_result": {
          "counter": { "1001": 12, "0110": 35, "1111": 1 },
          "raw": ["1001", "1001", "0110", "1001", "0110"]
        },
        "updated_at": "2021-11-10T15:27:06.698066",
        "variables": {
          "Omega_max": 14.4,
          "last_target": "q1",
          "ts": [200, 500]
        }
      }
    ]
  },
  "message": "OK.",
  "status": "success"
})";

CUDAQ_TEST(PasqalTester, checkHamiltonianJson) {
  cudaq::ahs::AtomArrangement layout;
  layout.sites = {{0.0, 0.0}, {0.0, 6.0e-6}, {6.0e-6, 0.0}};
  layout.filling = {1, 1, 1};

  cudaq::ahs::PhysicalField amplitude;
  amplitude.time_series = std::vector<std::pair<double, double>>{
    {0.0, 0.0}, {1.07e7, 1.0e-6}, {1.07e7, 2.0e-6}, {0.0, 3.0e-6}
  };

  cudaq::ahs::PhysicalField detuning;
  detuning.time_series =
  std::vector<std::pair<double, double>>{{-5.4e6, 0.0}, {5.4e6, 3.0e-6}};

  cudaq::ahs::PhysicalField phase;
  phase.time_series = std::vector<std::pair<double, double>>{{0.0, 0.0}};

  cudaq::ahs::DrivingField drive;
  drive.amplitude = amplitude;
  drive.detuning = detuning;
  drive.phase = phase;

  cudaq::ahs::Program sequence;
  sequence.setup.ahs_register = layout;
  sequence.hamiltonian.drivingFields = {drive};

  nlohmann::json serializedSequence = sequence;
  cudaq::ahs::Program refSequence = nlohmann::json::parse(sampleSequence);
  EXPECT_EQ(serializedSequence, refSequence);
}
