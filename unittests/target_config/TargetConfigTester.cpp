/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LinkedLibraryHolder.h"
#include "common/RuntimeTarget.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unordered_map>

// All ExternalBackendTester tests require CUDAQ_ENABLE_PYTHON
#ifdef CUDAQ_ENABLE_PYTHON
class ExternalBackendTester : public ::testing::Test {
protected:
  std::filesystem::path tmpRoot;

  void SetUp() override {
    tmpRoot =
        std::filesystem::temp_directory_path() /
        ("cudaq_test_" +
         std::string(
             ::testing::UnitTest::GetInstance()->current_test_info()->name()));
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

TEST_F(ExternalBackendTester, setsPluginLibDir) {
  auto root = createBackendPackage("my-backend");

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(root / "targets", targets, simTargets,
                              root / "lib");

  ASSERT_EQ(targets.count("my-backend"), 1);
  EXPECT_EQ(targets.at("my-backend").pluginLibDir, (root / "lib").string());
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
  EXPECT_EQ(targets.at("backend-a").pluginLibDir, (rootA / "lib").string());
  EXPECT_EQ(targets.at("backend-b").pluginLibDir, (rootB / "lib").string());
}

TEST_F(ExternalBackendTester, serverHelperPathResolvesToLibDir) {
  auto root = createBackendPackage("my-backend", /*createSo=*/true);

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(root / "targets", targets, simTargets,
                              root / "lib");

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &target = targets.at("my-backend");
  auto resolvedPath = std::filesystem::path(target.pluginLibDir) /
                      ("libcudaq-serverhelper-" + target.name + ".so");
  EXPECT_TRUE(std::filesystem::exists(resolvedPath));
}

TEST_F(ExternalBackendTester, pluginYamlPath_resolvesToTargetsDir) {
  auto root = createBackendPackage("my-backend");

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(root / "targets", targets, simTargets,
                              root / "lib");

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &target = targets.at("my-backend");
  ASSERT_FALSE(target.pluginLibDir.empty());

  auto ymlPath = target.pluginYamlPath();
  EXPECT_EQ(ymlPath, root / "targets" / "my-backend.yml");
  EXPECT_TRUE(std::filesystem::exists(ymlPath));
}

// -- B1: registerBackendPath -------------------------------------------------

TEST_F(ExternalBackendTester, registerBackendPath_addsTargets) {
  auto root = createBackendPackage("my-backend");

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::registerBackendPath(root, targets, simTargets);

  ASSERT_EQ(targets.count("my-backend"), 1);
  EXPECT_EQ(targets.at("my-backend").name, "my-backend");
  EXPECT_EQ(targets.at("my-backend").pluginLibDir, (root / "lib").string());
}

TEST_F(ExternalBackendTester, registerBackendPath_rejectsMissingPath) {
  auto bogus = tmpRoot / "does-not-exist";
  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  try {
    cudaq::registerBackendPath(bogus, targets, simTargets);
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(bogus.string()), std::string::npos)
        << "error message should mention the bad path: " << e.what();
  }
}

TEST_F(ExternalBackendTester, registerBackendPath_rejectsMissingTargetsDir) {
  // Create a root that exists but has no targets/ subdir.
  auto root = tmpRoot / "no-targets";
  std::filesystem::create_directories(root);

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  try {
    cudaq::registerBackendPath(root, targets, simTargets);
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(root.string()), std::string::npos)
        << "error message should mention the offending path: " << e.what();
  }
}

TEST_F(ExternalBackendTester, pluginLibrariesFieldIsParsed) {
  auto root = tmpRoot / "pluginlibtest";
  auto targetsDir = root / "targets";
  auto libDir = root / "lib";
  std::filesystem::create_directories(targetsDir);
  std::filesystem::create_directories(libDir);

  // Write a YAML with plugin-libraries
  std::ofstream(targetsDir / "my-backend.yml") << R"(
name: my-backend
description: Plugin-libraries test
target-arguments: []
config:
  platform-qpu: remote_rest
  library-mode: false
  plugin-libraries:
    - libdummy1.so
    - libdummy2.so
)";

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(targetsDir, targets, simTargets, libDir);

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &target = targets.at("my-backend");
  const auto &libs = target.config.PluginLibraries;
  ASSERT_EQ(libs.size(), 2);
  EXPECT_EQ(libs[0], "libdummy1.so");
  EXPECT_EQ(libs[1], "libdummy2.so");
}

TEST_F(ExternalBackendTester, pluginLibrariesAreDlopenedOnSetTarget) {
  auto root = tmpRoot / "pluginlibdltest";
  auto targetsDir = root / "targets";
  auto libDir = root / "lib";
  std::filesystem::create_directories(targetsDir);
  std::filesystem::create_directories(libDir);

  auto pluginPath = std::filesystem::path(CUDAQ_DLOPEN_SENTINEL_PLUGIN_PATH);
  auto pluginFileName = std::string(CUDAQ_DLOPEN_SENTINEL_PLUGIN_FILENAME);
  std::filesystem::copy_file(pluginPath, libDir / pluginFileName,
                             std::filesystem::copy_options::overwrite_existing);

  auto sentinelPath = tmpRoot / "plugin-dlopen.sentinel";
  std::filesystem::remove(sentinelPath);
  setenv("CUDAQ_DLOPEN_SENTINEL_PATH", sentinelPath.c_str(), 1);

  // Write a YAML with plugin-libraries
  std::ofstream(targetsDir / "my-backend.yml") << R"(
name: my-backend
description: Plugin-libraries dlopen test
target-arguments: []
config:
  nvqir-simulation-backend: qpp
  library-mode: false
  plugin-libraries:
    - )" << pluginFileName << R"(
)";

  // Register the backend
  cudaq::LinkedLibraryHolder holder;
  holder.registerBackendPath(root);

  EXPECT_FALSE(std::filesystem::exists(sentinelPath));
  EXPECT_NO_THROW(holder.setTarget("my-backend"));
  EXPECT_TRUE(std::filesystem::exists(sentinelPath));
  unsetenv("CUDAQ_DLOPEN_SENTINEL_PATH");
}

TEST_F(ExternalBackendTester,
       pluginRootTokenIsSubstitutedWhenTargetsAreScanned) {
  auto root = tmpRoot / "pluginroottest";
  auto targetsDir = root / "targets";
  auto libDir = root / "lib";
  auto dataDir = root / "data";
  std::filesystem::create_directories(targetsDir);
  std::filesystem::create_directories(libDir);
  std::filesystem::create_directories(dataDir);

  const auto expectedTopology = (root / "data" / "topology.txt").string();
  std::ofstream(dataDir / "topology.txt") << "topology\n";
  std::ofstream(targetsDir / "my-backend.yml") << R"(
name: my-backend
description: Plugin-root substitution test
target-arguments: []
config:
  nvqir-simulation-backend: qpp
  jit-mid-level-pipeline: "map{device=file(%PLUGIN_ROOT%/data/topology.txt)}"
  preprocessor-defines:
    - "-DTOPOLOGY=%PLUGIN_ROOT%/data/topology.txt"
)";

  std::unordered_map<std::string, cudaq::RuntimeTarget> targets, simTargets;
  cudaq::findAvailableTargets(targetsDir, targets, simTargets, libDir);

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &config = targets.at("my-backend").config;
  ASSERT_TRUE(config.BackendConfig.has_value());
  EXPECT_EQ(config.BackendConfig->JITMidLevelPipeline,
            "map{device=file(" + expectedTopology + ")}");
  ASSERT_EQ(config.BackendConfig->PreprocessorDefines.size(), 1);
  EXPECT_EQ(config.BackendConfig->PreprocessorDefines.front(),
            "-DTOPOLOGY=" + expectedTopology);
}

TEST(TargetConfigTester, pluginRootTokenSubstitutionReplacesAllOccurrences) {
  const std::string yaml = R"(
name: token-test
description: Token substitution test
config:
  jit-mid-level-pipeline: "%PLUGIN_ROOT%/a:%PLUGIN_ROOT%/b"
)";

  const auto substituted = cudaq::config::substitutePluginRoot(
      yaml, std::filesystem::path("/opt/cudaq/plugins/token-test"));

  EXPECT_NE(substituted.find("/opt/cudaq/plugins/token-test/a"),
            std::string::npos);
  EXPECT_NE(substituted.find("/opt/cudaq/plugins/token-test/b"),
            std::string::npos);
  EXPECT_EQ(substituted.find("%PLUGIN_ROOT%"), std::string::npos);
}

#endif // CUDAQ_ENABLE_PYTHON
