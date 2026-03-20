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

class ExternalBackendTester : public ::testing::Test {
protected:
  std::filesystem::path tmpRoot;

  void SetUp() override {
    tmpRoot = std::filesystem::temp_directory_path() /
              ("cudaq_test_" + std::string(
                                   ::testing::UnitTest::GetInstance()
                                       ->current_test_info()
                                       ->name()));
    std::filesystem::create_directories(tmpRoot);
  }

  void TearDown() override { std::filesystem::remove_all(tmpRoot); }

  std::filesystem::path createBackendPackage(const std::string &name,
                                             bool createSo = false) {
    auto root = tmpRoot / name;
    auto targetsDir = root / "targets";
    auto libDir = root / "lib";
    std::filesystem::create_directories(targetsDir);
    std::filesystem::create_directories(libDir);

    std::ofstream(targetsDir / (name + ".yml"))
        << "name: " << name << "\ndescription: \"Test backend.\"\nconfig:\n"
        << "  platform-qpu: remote_rest\n  library-mode: false\n";

    if (createSo)
      std::ofstream(libDir / ("libcudaq-serverhelper-" + name + ".so")).close();

    return root;
  }
};

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

TEST_F(ExternalBackendTester, setsServerHelperLibDir) {
  auto root = createBackendPackage("my-backend");

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(root / "targets", targets, simTargets,
                              root / "lib");

  ASSERT_EQ(targets.count("my-backend"), 1);
  EXPECT_EQ(targets.at("my-backend").serverHelperLibDir,
            (root / "lib").string());
  EXPECT_EQ(targets.at("my-backend").name, "my-backend");
}

TEST_F(ExternalBackendTester, backendPathMultipleEntries) {
  auto rootA = createBackendPackage("backend-a");
  auto rootB = createBackendPackage("backend-b");

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
}

TEST_F(ExternalBackendTester, serverHelperPathResolvesToLibDir) {
  auto root = createBackendPackage("my-backend", /*createSo=*/true);

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(root / "targets", targets, simTargets,
                              root / "lib");

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &target = targets.at("my-backend");
  auto resolvedPath = std::filesystem::path(target.serverHelperLibDir) /
                      ("libcudaq-serverhelper-" + target.name + ".so");
  EXPECT_TRUE(std::filesystem::exists(resolvedPath));
}

TEST_F(ExternalBackendTester, reconstructYmlPathFromServerHelperLibDir) {
  auto root = createBackendPackage("my-backend");

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(root / "targets", targets, simTargets,
                              root / "lib");

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &target = targets.at("my-backend");
  ASSERT_FALSE(target.serverHelperLibDir.empty());

  auto ymlPath =
      std::filesystem::path(target.serverHelperLibDir).parent_path() /
      "targets" / (target.name + ".yml");
  EXPECT_TRUE(std::filesystem::exists(ymlPath));
}
