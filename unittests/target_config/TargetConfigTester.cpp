/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/TargetConfigYaml.h"
#include <gtest/gtest.h>

TEST(TargetConfigTester, checkMachineList) {
  const std::string configYmlContents = R"(
name: test
description: "CUDA-Q test target."
config:
  # Tell DefaultQuantumPlatform what QPU subtype to use
  platform-qpu: remote_rest
  codegen-emission: qir-base
  # Library mode is only for simulators, physical backends must turn this off
  library-mode: false

target-arguments:
  - key: machine
    required: false
    type: machine-config
    platform-arg: machine 
    help-string: "Specify QPU."
    machine-config:
      - name: gen1
        machine-names: 
          - device1-1
          - device1-2 
        config: 
          codegen-spec: qir-adaptive:0.1:int_computations
      - name: gen2
        machine-names: 
          - device2-1
          - device2-2
        config: 
          codegen-spec: qir-adaptive:0.2:int_computations,float_computations
)";

  cudaq::config::TargetConfig config;
  llvm::yaml::Input Input(configYmlContents.c_str());
  Input >> config;
  // No machine, use default
  EXPECT_EQ(config.getCodeGenSpec({}), "qir-base");
  // Unspecified machine, use default
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "unknown"}}), "qir-base");
  // Gen 1
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "device1-1"}}),
            "qir-adaptive:0.1:int_computations");
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "device1-2"}}),
            "qir-adaptive:0.1:int_computations");
  // Gen 2
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "device2-1"}}),
            "qir-adaptive:0.2:int_computations,float_computations");
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "device2-2"}}),
            "qir-adaptive:0.2:int_computations,float_computations");
}
