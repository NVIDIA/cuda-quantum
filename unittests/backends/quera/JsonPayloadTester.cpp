/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/AnalogHamiltonian.h"
#include <gtest/gtest.h>

namespace {
// Sample payload
const std::string referencePayload = R"(
  {
  "setup": {
    "ahs_register": {
      "sites": [
        [
          "0.0",
          "0.0"
        ],
        [
          "0.0",
          "0.000003"
        ],
        [
          "0.0",
          "0.000006"
        ],
        [
          "0.000003",
          "0.0"
        ],
        [
          "0.000003",
          "0.000003"
        ],
        [
          "0.000003",
          "0.000003"
        ],
        [
          "0.000003",
          "0.000006"
        ]
      ],
      "filling": [
        1,
        1,
        1,
        1,
        1,
        0,
        0
      ]
    }
  },
  "hamiltonian": {
    "drivingFields": [
      {
        "amplitude": {
          "time_series": {
            "values": [
              "0.0",
              "25132700.0",
              "25132700.0",
              "0.0"
            ],
            "times": [
              "0.0",
              "3E-7",
              "0.0000027",
              "0.000003"
            ]
          },
          "pattern": "uniform"
        },
        "phase": {
          "time_series": {
            "values": [
              "0",
              "0"
            ],
            "times": [
              "0.0",
              "0.000003"
            ]
          },
          "pattern": "uniform"
        },
        "detuning": {
          "time_series": {
            "values": [
              "-125664000.0",
              "-125664000.0",
              "125664000.0",
              "125664000.0"
            ],
            "times": [
              "0.0",
              "3E-7",
              "0.0000027",
              "0.000003"
            ]
          },
          "pattern": "uniform"
        }
      }
    ],
    "localDetuning": [
      {
        "magnitude": {
          "time_series": {
            "values": [
              "-125664000.0",
              "125664000.0"
            ],
            "times": [
              "0.0",
              "0.000003"
            ]
          },
          "pattern": [
            "0.5",
            "1.0",
            "0.5",
            "0.5",
            "0.5",
            "0.5"
          ]
        }
      }
    ]
  }
}
  )";

// Sample result
const std::string referenceResult = R"(
  {
    "taskMetadata": {
        "id": "foo",
        "shots": 2,
        "deviceId": "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
        "createdAt": "2022-10-25T20:59:10.788Z",
        "endedAt": "2022-10-25T21:00:58.218Z",
        "status": "COMPLETED"
    },
    "measurements": [
        {
            "shotMetadata": {"shotStatus": "Success"},
            "shotResult": {
                "preSequence": [1, 1, 1, 1],
                "postSequence": [0, 1, 1, 1]
            }
        },
        {
            "shotMetadata": {"shotStatus": "Success"},
            "shotResult": {
                "preSequence": [1, 1, 0, 1],
                "postSequence": [1, 0, 0, 0]
            }
        }
    ],
    "additionalMetadata": {
        "action": {
            "setup": {
                "ahs_register": {
                    "sites": [
                        ["0", "0"],
                        ["0", "0.000004"],
                        ["0.000004", "0"]
                    ],
                    "filling": [1, 1, 1]
                }
            },
            "hamiltonian": {
                "drivingFields": [
                    {
                        "amplitude": {
                            "time_series": {
                                "values": ["0", "15700000", "15700000", "0"],
                                "times": ["0", "0.000001", "0.000002", "0.000003"]
                            },
                            "pattern": "uniform"
                        },
                        "phase": {
                            "time_series": {
                                "values": ["0", "0"],
                                "times": ["0", "0.000003"]
                            },
                            "pattern": "uniform"
                        },
                        "detuning": {
                            "time_series": {
                                "values": ["-54000000", "54000000"],
                                "times": ["0", "0.000003"]
                            },
                            "pattern": "uniform"
                        }
                    }
                ],
                "localDetuning": [
                    {
                        "magnitude": {
                            "time_series": {
                                "values": ["0", "25000000", "25000000", "0"],
                                "times": ["0", "0.000001", "0.000002", "0.000003"]
                            },
                            "pattern": ["0.8", "1", "0.9"]
                        }
                    }
                ]
            }
        },
        "queraMetadata": {
            "numSuccessfulShots": 100
        }
    }
}
)";

const std::string measurements_only = R"(
{
  "measurements": [
    {
      "shotMetadata": {
        "shotStatus": "Success"
      },
      "shotResult": {
        "postSequence": [
          0,
          1,
          1,
          0,
          0,
          0,
          1,
          1
        ],
        "preSequence": [
          1,
          1,
          1,
          1,
          0,
          1,
          1,
          1
        ]
      }
    },
    {
      "shotMetadata": {
        "shotStatus": "Success"
      },
      "shotResult": {
        "postSequence": [
          1,
          0,
          1,
          1,
          1,
          1,
          1,
          1
        ],
        "preSequence": [
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1
        ]
      }
    }
  ]
})";
} // namespace

CUDAQ_TEST(QueraTester, checkProgramJsonify) {
  cudaq::ahs::AtomArrangement atoms;
  atoms.sites = {{0.0, 0.0},      {0.0, 3.0e-6},    {0.0, 6.0e-6},
                 {3.0e-6, 0.0},   {3.0e-6, 3.0e-6}, {3.0e-6, 3.0e-6},
                 {3.0e-6, 6.0e-6}};
  atoms.filling = {1, 1, 1, 1, 1, 0, 0};
  cudaq::ahs::Program program;
  program.setup.ahs_register = atoms;

  cudaq::ahs::PhysicalField Omega;
  Omega.time_series = std::vector<std::pair<double, double>>{
      {0.0, 0.0}, {2.51327e7, 3.0e-7}, {2.51327e7, 2.7e-6}, {0.0, 3.0e-6}};

  cudaq::ahs::PhysicalField Phi;
  Phi.time_series =
      std::vector<std::pair<double, double>>{{0.0, 0.0}, {0.0, 3.0e-6}};

  cudaq::ahs::PhysicalField Delta;
  Delta.time_series =
      std::vector<std::pair<double, double>>{{-1.25664e8, 0.0},
                                             {-1.25664e8, 3.0e-7},
                                             {1.25664e8, 2.7e-6},
                                             {1.25664e8, 3.0e-6}};
  cudaq::ahs::DrivingField drive;
  drive.amplitude = Omega;
  drive.phase = Phi;
  drive.detuning = Delta;
  program.hamiltonian.drivingFields = {drive};

  cudaq::ahs::PhysicalField localDetuning;
  localDetuning.time_series = std::vector<std::pair<double, double>>{
      {-1.25664e8, 0.0}, {1.25664e8, 3.0e-6}};
  localDetuning.pattern = std::vector<double>{0.5, 1.0, 0.5, 0.5, 0.5, 0.5};
  cudaq::ahs::LocalDetuning detuning;
  detuning.magnitude = localDetuning;

  program.hamiltonian.localDetuning = {detuning};
  nlohmann::json j = program;
  std::cout << j.dump(4) << std::endl;

  cudaq::ahs::Program refProgram = nlohmann::json::parse(referencePayload);
  EXPECT_EQ(refProgram.setup.ahs_register.sites,
            program.setup.ahs_register.sites);
  EXPECT_EQ(refProgram.setup.ahs_register.filling,
            program.setup.ahs_register.filling);
  EXPECT_EQ(refProgram.hamiltonian.drivingFields.size(),
            program.hamiltonian.drivingFields.size());
  EXPECT_EQ(refProgram.hamiltonian.localDetuning.size(),
            program.hamiltonian.localDetuning.size());

  const auto checkField = [](const cudaq::ahs::PhysicalField &field1,
                             const cudaq::ahs::PhysicalField &field2) {
    EXPECT_EQ(field1.pattern, field2.pattern);
    EXPECT_TRUE(field1.time_series.almostEqual(field2.time_series));
  };
  for (size_t i = 0; i < program.hamiltonian.drivingFields.size(); ++i) {
    auto refField = refProgram.hamiltonian.drivingFields[i];
    auto field = program.hamiltonian.drivingFields[i];
    checkField(refField.amplitude, field.amplitude);
    checkField(refField.phase, field.phase);
    checkField(refField.detuning, field.detuning);
  }
  for (size_t i = 0; i < program.hamiltonian.localDetuning.size(); ++i) {
    auto refField = refProgram.hamiltonian.localDetuning[i];
    auto field = program.hamiltonian.localDetuning[i];
    checkField(refField.magnitude, field.magnitude);
  }
}

CUDAQ_TEST(QueraTester, checkResultJsonify) {
  cudaq::ahs::TaskMetadata tm;
  tm.id = "foo";
  tm.shots = 2;
  tm.deviceId = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila";
  tm.createdAt = "2022-10-25T20:59:10.788Z";
  tm.endedAt = "2022-10-25T21:00:58.218Z";
  tm.status = "COMPLETED";

  cudaq::ahs::ShotMetadata md0;
  md0.shotStatus = "Success";
  cudaq::ahs::ShotResult r0;
  r0.preSequence = {1, 1, 1, 1};
  r0.postSequence = {0, 1, 1, 1};

  cudaq::ahs::ShotMeasurement sm0;
  sm0.shotMetadata = md0;
  sm0.shotResult = r0;

  cudaq::ahs::ShotMetadata md1;
  md1.shotStatus = "Success";
  cudaq::ahs::ShotResult r1;
  r1.preSequence = {1, 1, 0, 1};
  r1.postSequence = {1, 0, 0, 0};

  cudaq::ahs::ShotMeasurement sm1;
  sm1.shotMetadata = md1;
  sm1.shotResult = r1;

  std::vector<cudaq::ahs::ShotMeasurement> measurements;
  measurements.push_back(sm0);
  measurements.push_back(sm1);

  cudaq::ahs::QueraMetadata qm;
  qm.numSuccessfulShots = 100;

  cudaq::ahs::AdditionalMetadata am;
  am.queraMetadata = qm;
  am.action = cudaq::ahs::Program();

  cudaq::ahs::TaskResult tr;
  tr.taskMetadata = tm;
  tr.measurements = measurements;
  tr.additionalMetadata = am;

  cudaq::ahs::TaskResult refResult = nlohmann::json::parse(referenceResult);

  EXPECT_EQ(refResult.taskMetadata.id, tr.taskMetadata.id);
  EXPECT_EQ(refResult.taskMetadata.shots, tr.taskMetadata.shots);
  EXPECT_EQ(refResult.taskMetadata.deviceId, tr.taskMetadata.deviceId);
  EXPECT_EQ(refResult.taskMetadata.deviceParameters,
            tr.taskMetadata.deviceParameters);
  EXPECT_EQ(refResult.taskMetadata.createdAt, tr.taskMetadata.createdAt);
  EXPECT_EQ(refResult.taskMetadata.endedAt, tr.taskMetadata.endedAt);
  EXPECT_EQ(refResult.taskMetadata.status, tr.taskMetadata.status);
  EXPECT_EQ(refResult.taskMetadata.failureReason,
            tr.taskMetadata.failureReason);

  EXPECT_EQ(refResult.measurements.value().size(),
            tr.measurements.value().size());

  EXPECT_EQ(
      refResult.additionalMetadata.value().queraMetadata.numSuccessfulShots,
      tr.additionalMetadata.value().queraMetadata.numSuccessfulShots);
}

CUDAQ_TEST(QueraTester, checkProcessResult) {
  nlohmann::json resultsJson = nlohmann::json::parse(measurements_only);
  std::unordered_map<std::string, std::size_t> globalReg;
  std::unordered_map<std::string, std::size_t> preSeqReg;
  std::unordered_map<std::string, std::size_t> postSeqReg;
  if (resultsJson.contains("measurements")) {
    auto const &measurements = resultsJson.at("measurements");
    for (auto const &m : measurements) {
      cudaq::ahs::ShotMeasurement sm = m;
      if (sm.shotMetadata.shotStatus == "Success") {
        auto pre = sm.shotResult.preSequence.value();
        std::string preString = "";
        for (int bit : pre)
          preString += std::to_string(bit);
        preSeqReg[preString]++;
        auto post = sm.shotResult.postSequence.value();
        std::string postString = "";
        for (int bit : post)
          postString += std::to_string(bit);
        postSeqReg[postString]++;
        std::vector<int> state_idx(pre.size());
        for (size_t i = 0; i < pre.size(); ++i)
          state_idx[i] = pre[i] * (1 + post[i]);
        std::string bitString = "";
        for (int bit : state_idx)
          bitString += std::to_string(bit);
        globalReg[bitString]++;
      }
    }
  }
  for (const auto &pair : preSeqReg) {
    std::cout << pair.first << ": " << pair.second << std::endl;
    EXPECT_EQ(1, pair.second);
  }
  for (const auto &pair : postSeqReg) {
    std::cout << pair.first << ": " << pair.second << std::endl;
    EXPECT_EQ(1, pair.second);
  }
  for (const auto &pair : globalReg) {
    std::cout << pair.first << ": " << pair.second << std::endl;
    EXPECT_EQ(1, pair.second);
  }
}
