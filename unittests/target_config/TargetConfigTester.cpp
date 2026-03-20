/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LinkedLibraryHolder.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include <filesystem>
#include <fstream>
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

TEST(TargetConfigTester, setsServerHelperLibDir) {
  auto tmpDir = std::filesystem::temp_directory_path() /
                "cudaq_test_find_available_targets";
  auto targetsDir = tmpDir / "targets";
  auto libDir = tmpDir / "lib";
  std::filesystem::create_directories(targetsDir);
  std::filesystem::create_directories(libDir);

  const std::string ymlContent = R"(
name: my-backend
description: "Test backend."
config:
  platform-qpu: remote_rest
  library-mode: false
)";
  std::ofstream(targetsDir / "my-backend.yml") << ymlContent;

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(targetsDir, targets, simTargets, libDir);

  ASSERT_EQ(targets.count("my-backend"), 1);
  EXPECT_EQ(targets.at("my-backend").serverHelperLibDir, libDir.string());
  EXPECT_EQ(targets.at("my-backend").name, "my-backend");

  std::filesystem::remove_all(tmpDir);
}

TEST(TargetConfigTester, backendPathMultipleEntries) {
  auto tmpDir =
      std::filesystem::temp_directory_path() / "cudaq_test_backend_path";

  auto createBackend = [&](const std::string &name) {
    auto root = tmpDir / name;
    std::filesystem::create_directories(root / "targets");
    std::filesystem::create_directories(root / "lib");
    std::ofstream(root / "targets" / (name + ".yml"))
        << "name: " << name << "\ndescription: \"Test.\"\nconfig:\n"
        << "  platform-qpu: remote_rest\n  library-mode: false\n";
    return root;
  };

  auto rootA = createBackend("backend-a");
  auto rootB = createBackend("backend-b");

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  for (auto &root : {rootA, rootB})
    cudaq::findAvailableTargets(root / "targets", targets, simTargets,
                                root / "lib");

  ASSERT_EQ(targets.count("backend-a"), 1);
  ASSERT_EQ(targets.count("backend-b"), 1);
  EXPECT_EQ(targets.at("backend-a").serverHelperLibDir,
            (rootA / "lib").string());
  EXPECT_EQ(targets.at("backend-b").serverHelperLibDir,
            (rootB / "lib").string());

  std::filesystem::remove_all(tmpDir);
}
