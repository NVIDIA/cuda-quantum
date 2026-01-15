/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
  platform-qpu: remote_rest
  codegen-emission: qir-base
  library-mode: false

target-arguments:
  - key: machine
    required: false
    type: machine-config
    platform-arg: machine 
    help-string: "Specify QPU."
    machine-config:
      - arch-name: gen1
        machine-names: 
          - device1-1
          - device1-2 
        config: 
          codegen-emission: qir-adaptive:0.1:int_computations
      - arch-name: gen2
        machine-names: 
          - device2-1
          - device2-2
        config: 
          codegen-emission: qir-adaptive:1.0:int_computations,float_computations
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
            "qir-adaptive:1.0:int_computations,float_computations");
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "device2-2"}}),
            "qir-adaptive:1.0:int_computations,float_computations");
}

TEST(TargetConfigTester, checkRegex) {
  const std::string configYmlContents = R"(
name: test
description: "CUDA-Q test target."
config:
  platform-qpu: remote_rest
  codegen-emission: qir-base
  library-mode: false

target-arguments:
  - key: machine
    required: false
    type: machine-config
    platform-arg: machine
    help-string: "Specify QPU."
    machine-config:
      - arch-name: gen1
        pattern: H[0-9.-]+-[A-Z0-9.-]+
        config:
          codegen-emission: qir-adaptive:0.1:int_computations
      - arch-name: gen2
        pattern: Helios.*
        config:
          codegen-emission: qir-adaptive:1.0:int_computations,float_computations
)";

  cudaq::config::TargetConfig config;
  llvm::yaml::Input Input(configYmlContents.c_str());
  Input >> config;
  // No machine, use default
  EXPECT_EQ(config.getCodeGenSpec({}), "qir-base");
  // Unmatched machine, use default
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "unknown"}}), "qir-base");
  // Gen 1
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "H1-1"}}),
            "qir-adaptive:0.1:int_computations");
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "H2-1SC"}}),
            "qir-adaptive:0.1:int_computations");
  // Gen 2
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "Helios-1SC"}}),
            "qir-adaptive:1.0:int_computations,float_computations");
  EXPECT_EQ(config.getCodeGenSpec({{"machine", "Helios-1E"}}),
            "qir-adaptive:1.0:int_computations,float_computations");
}

TEST(TargetConfigTester, checkBoolArguments) {
  const std::string configYmlContents = R"(
name: test-bool
description: "CUDA-Q test target with Bool arguments."
config:
  platform-qpu: remote_rest
  codegen-emission: qir-base
  library-mode: false

target-arguments:
  - key: enable-opt
    required: false
    type: bool
    platform-arg: enable_opt
    help-string: "Enable optimization (Bool flag, no value needed)."
  - key: debug
    required: false
    type: bool
    help-string: "Enable debug mode (Bool flag)."
  - key: machine
    required: false
    type: string
    platform-arg: qpu
    help-string: "Specify QPU (string argument)."
  - key: shots
    required: false
    type: integer
    platform-arg: shots
    help-string: "Number of shots (integer argument)."
)";

  cudaq::config::TargetConfig config;
  llvm::yaml::Input Input(configYmlContents.c_str());
  Input >> config;

  // Verify target name and description
  EXPECT_EQ(config.Name, "test-bool");
  EXPECT_EQ(config.Description, "CUDA-Q test target with Bool arguments.");

  // Verify we have 4 arguments
  EXPECT_EQ(config.TargetArguments.size(), 4);

  // Verify Bool arguments are correctly parsed
  EXPECT_EQ(config.TargetArguments[0].KeyName, "enable-opt");
  EXPECT_EQ(config.TargetArguments[0].Type, cudaq::config::ArgumentType::Bool);
  EXPECT_EQ(config.TargetArguments[0].PlatformArgKey, "enable_opt");
  EXPECT_FALSE(config.TargetArguments[0].IsRequired);

  EXPECT_EQ(config.TargetArguments[1].KeyName, "debug");
  EXPECT_EQ(config.TargetArguments[1].Type, cudaq::config::ArgumentType::Bool);
  EXPECT_FALSE(config.TargetArguments[1].IsRequired);

  // Verify non-Bool arguments still work
  EXPECT_EQ(config.TargetArguments[2].KeyName, "machine");
  EXPECT_EQ(config.TargetArguments[2].Type,
            cudaq::config::ArgumentType::String);

  EXPECT_EQ(config.TargetArguments[3].KeyName, "shots");
  EXPECT_EQ(config.TargetArguments[3].Type, cudaq::config::ArgumentType::Int);
}
