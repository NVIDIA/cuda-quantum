/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetConfigYaml.h"
#ifdef CUDAQ_ENABLE_PYTHON
#include "LinkedLibraryHolder.h"
#include "common/RuntimeTarget.h"
#include "cudaq/platform/qpu_utils.h"
#endif
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unordered_map>

// ExternalBackendTester is not inherently Python-specific, but this test group
// currently uses backend discovery helpers and LinkedLibraryHolder from
// cudaq-py-utils, which is built only when the Python project is enabled. Keep
// it gated until those general-purpose helpers are moved into the runtime.
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

  std::filesystem::path
  createBackendPackage(const std::string &name, bool createSo = false,
                       std::string version = CUDAQ_TEST_VERSION) {
    auto root = tmpRoot / name;
    auto targetsDir = root / "targets";
    auto libDir = root / "lib";
    std::filesystem::create_directories(targetsDir);
    std::filesystem::create_directories(libDir);

    std::ofstream configFile(targetsDir / (name + ".yml"));
    configFile << "name: " << name << "\ndescription: \"Test backend.\"\n";
    if (!version.empty())
      configFile << "cudaq-version: \"" << version << "\"\n";
    configFile << "config:\n"
               << "  platform-qpu: remote_rest\n  library-mode: false\n";

    if (createSo)
      std::ofstream(libDir / ("libcudaq-serverhelper-" + name + ".so")).close();

    return root;
  }
};
#endif

TEST(TargetConfigTester, parsesCudaqVersion) {
  const auto config = cudaq::config::parseTargetConfig(R"(
name: version-test
description: Version parsing test
cudaq-version: "0.9.0-rc2+build.1"
config:
  library-mode: true
)");
  EXPECT_EQ(config.CudaqVersion, "0.9.0-rc2+build.1");
}

TEST(TargetConfigTester, missingTargetConfigThrows) {
  const auto missingPath = std::filesystem::temp_directory_path() /
                           "cudaq-missing-target-config.yml";
  try {
    (void)cudaq::config::loadTargetConfig(missingPath);
    FAIL() << "Expected loadTargetConfig to throw";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(missingPath.string()),
              std::string::npos);
  }
}

TEST(TargetConfigTester, checksExternalTargetVersionCompatibility) {
  using Compatibility = cudaq::config::TargetVersionCompatibility;
  struct TestCase {
    const char *Plugin;
    const char *Current;
    Compatibility Expected;
    // Substring the diagnostic must contain (empty for the Compatible cases).
    const char *DiagContains;
  };
  const TestCase cases[] = {
      {"0.9.0", "0.8.1", Compatibility::Error, "was built for CUDA-Q 0.9.0"},
      {"0.9.2", "0.9.1", Compatibility::Error, "was built for CUDA-Q 0.9.2"},
      {"0.10.0", "0.9.9", Compatibility::Error, "was built for CUDA-Q 0.10.0"},
      {"0.0.0", "0.0.0", Compatibility::Compatible, ""},
      {"0.9.0", "0.9.0", Compatibility::Compatible, ""},
      {"0.9.0", "0.9.3", Compatibility::Compatible, ""},
      {"0.9.0", "0.10.0", Compatibility::Warning,
       "compatibility is not guaranteed"},
      {"0.9.0", "1.0.0", Compatibility::Warning,
       "compatibility is not guaranteed"},
      {"0.9.0", "0.9.0-rc2-developer", Compatibility::Compatible, ""},
      // Non-numeric current version: string-compare, warn if different.
      {"0.9.0", "developer", Compatibility::Warning,
       "versions are non-numeric so compatibility cannot be verified"},
      {"amd64-pr-1234", "amd64-pr-1234", Compatibility::Compatible, ""},
      {"amd64-pr-1234", "amd64-pr-5678", Compatibility::Warning,
       "versions are non-numeric so compatibility cannot be verified"},
      // Both empty (CI dev builds with no version set): compatible.
      {"", "", Compatibility::Compatible, ""},
      // Numeric current + empty plugin: plugin metadata is required → Error.
      {"", "0.9.0", Compatibility::Error, "missing or malformed"},
  };

  cudaq::config::TargetConfig config;
  config.Name = "version-test";
  for (const auto &test : cases) {
    config.CudaqVersion = test.Plugin;
    const auto result = cudaq::config::checkExternalTargetVersion(
        config, test.Current, "/tmp/version-test.yml");
    EXPECT_EQ(result.Status, test.Expected)
        << "plugin=" << test.Plugin << " current=" << test.Current;
    if (test.Expected != Compatibility::Compatible) {
      EXPECT_NE(result.Diagnostic.find("version-test"), std::string::npos);
      EXPECT_NE(result.Diagnostic.find("/tmp/version-test.yml"),
                std::string::npos);
      EXPECT_NE(result.Diagnostic.find(test.DiagContains), std::string::npos)
          << "plugin=" << test.Plugin << " current=" << test.Current
          << " diagnostic=" << result.Diagnostic;
    }
  }
}

TEST(TargetConfigTester, nonNumericVersionsFallBackToStringComparison) {
  using Compatibility = cudaq::config::TargetVersionCompatibility;
  cudaq::config::TargetConfig config;
  config.Name = "version-test";

  // When the current version is non-numeric, any plugin version produces a
  // Warning (differing) or Compatible (equal) — never an Error.
  for (const auto *pluginVer : {"", "0.9", "v0.9.0", "not-a-version"}) {
    config.CudaqVersion = pluginVer;
    // Different non-numeric strings → Warning.
    const auto diffResult = cudaq::config::checkExternalTargetVersion(
        config, "developer", "/tmp/version-test.yml");
    const bool isEqual = std::string(pluginVer) == std::string("developer");
    EXPECT_EQ(diffResult.Status,
              isEqual ? Compatibility::Compatible : Compatibility::Warning)
        << "plugin=" << pluginVer;
    if (!isEqual)
      EXPECT_NE(diffResult.Diagnostic.find("cannot be verified"),
                std::string::npos)
          << "plugin=" << pluginVer;
  }

  // Equal non-numeric strings (both empty, both same tag) → Compatible.
  config.CudaqVersion = "amd64-pr-1234";
  const auto equalResult = cudaq::config::checkExternalTargetVersion(
      config, "amd64-pr-1234", "/tmp/version-test.yml");
  EXPECT_EQ(equalResult.Status, Compatibility::Compatible);

  // When the current version IS numeric, non-numeric plugin versions remain
  // an Error (missing or malformed metadata).
  for (const auto *badPlugin : {"", "0.9", "v0.9.0", "0.-1.0"}) {
    config.CudaqVersion = badPlugin;
    const auto result = cudaq::config::checkExternalTargetVersion(
        config, "0.9.0", "/tmp/version-test.yml");
    EXPECT_EQ(result.Status, Compatibility::Error) << "plugin=" << badPlugin;
    EXPECT_NE(result.Diagnostic.find("missing or malformed"), std::string::npos)
        << "plugin=" << badPlugin;
  }
}

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

#ifdef CUDAQ_ENABLE_PYTHON
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

TEST_F(ExternalBackendTester, setTargetRequiresValidPluginVersionMetadata) {
  const auto missingRoot = createBackendPackage("missing-version", false, "");
  const auto malformedRoot =
      createBackendPackage("malformed-version", false, "not-a-version");

  cudaq::LinkedLibraryHolder holder;
  holder.registerBackendPath(missingRoot);
  holder.registerBackendPath(malformedRoot);

  EXPECT_THROW(holder.setTarget("missing-version"), std::runtime_error);
  EXPECT_THROW(holder.setTarget("malformed-version"), std::runtime_error);
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

TEST_F(ExternalBackendTester, versionFailurePreventsPluginLibraryLoad) {
  // This test requires a numeric current CUDA-Q version to perform semver
  // comparison. When the version is non-numeric (e.g. empty dev builds), the
  // validator falls back to string comparison and only warns, so no throw
  // occurs.
  const std::string testVersion(CUDAQ_TEST_VERSION);
  if (testVersion.empty() ||
      testVersion.find_first_of("0123456789") == std::string::npos ||
      testVersion.find('.') == std::string::npos) {
    GTEST_SKIP() << "Skipping: current CUDA-Q version '" << testVersion
                 << "' is non-numeric; semver rejection is not tested in dev "
                    "builds";
  }

  auto root = tmpRoot / "pluginversiontest";
  auto targetsDir = root / "targets";
  auto libDir = root / "lib";
  std::filesystem::create_directories(targetsDir);
  std::filesystem::create_directories(libDir);

  const auto pluginPath =
      std::filesystem::path(CUDAQ_DLOPEN_SENTINEL_PLUGIN_PATH);
  const auto pluginFileName =
      std::string(CUDAQ_DLOPEN_SENTINEL_PLUGIN_FILENAME);
  std::filesystem::copy_file(pluginPath, libDir / pluginFileName,
                             std::filesystem::copy_options::overwrite_existing);

  const auto sentinelPath = tmpRoot / "version-failure-dlopen.sentinel";
  std::filesystem::remove(sentinelPath);
  setenv("CUDAQ_DLOPEN_SENTINEL_PATH", sentinelPath.c_str(), 1);

  std::ofstream(targetsDir / "future-backend.yml") << R"(
name: future-backend
description: Future-version plugin test
cudaq-version: 999999.0.0
target-arguments: []
config:
  nvqir-simulation-backend: qpp
  library-mode: false
  plugin-libraries:
    - )" << pluginFileName << R"(
)";

  cudaq::LinkedLibraryHolder holder;
  holder.registerBackendPath(root);

  EXPECT_FALSE(std::filesystem::exists(sentinelPath));
  EXPECT_THROW(holder.setTarget("future-backend"), std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(sentinelPath));
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

TEST_F(ExternalBackendTester, nativeTargetLoadsPluginLibraries) {
  auto root = tmpRoot / "native-plugin-load";
  auto targetsDir = root / "targets";
  auto libDir = root / "lib";
  std::filesystem::create_directories(targetsDir);
  std::filesystem::create_directories(libDir);

  const auto pluginPath =
      std::filesystem::path(CUDAQ_DLOPEN_SENTINEL_PLUGIN_PATH);
  const auto pluginFileName =
      std::string(CUDAQ_DLOPEN_SENTINEL_PLUGIN_FILENAME);
  std::filesystem::copy_file(pluginPath, libDir / pluginFileName,
                             std::filesystem::copy_options::overwrite_existing);

  const auto sentinelPath = tmpRoot / "native-plugin-load.sentinel";
  std::filesystem::remove(sentinelPath);
  setenv("CUDAQ_DLOPEN_SENTINEL_PATH", sentinelPath.c_str(), 1);

  cudaq::config::TargetConfig config;
  config.PluginLibraries.push_back(pluginFileName);
  cudaq::detail::loadTargetPluginLibraries(
      "native-plugin-load", targetsDir / "native-plugin-load.yml", config);

  EXPECT_TRUE(std::filesystem::exists(sentinelPath));
  unsetenv("CUDAQ_DLOPEN_SENTINEL_PATH");
}

TEST_F(ExternalBackendTester, autoLoadsPlatformQpuLibrary) {
  auto root = tmpRoot / "qpu-auto-load";
  auto targetsDir = root / "targets";
  auto libDir = root / "lib";
  std::filesystem::create_directories(targetsDir);
  std::filesystem::create_directories(libDir);

  const auto pluginPath =
      std::filesystem::path(CUDAQ_DLOPEN_SENTINEL_PLUGIN_PATH);
  // Convention: libcudaq-<platform-qpu>-qpu.<ext>
  const auto qpuLibName =
      std::string("libcudaq-mock_observe_qpu-qpu") +
      std::filesystem::path(CUDAQ_DLOPEN_SENTINEL_PLUGIN_FILENAME).extension().string();
  std::filesystem::copy_file(pluginPath, libDir / qpuLibName,
                             std::filesystem::copy_options::overwrite_existing);

  const auto sentinelPath = tmpRoot / "qpu-auto-load.sentinel";
  std::filesystem::remove(sentinelPath);
  setenv("CUDAQ_DLOPEN_SENTINEL_PATH", sentinelPath.c_str(), 1);

  cudaq::config::TargetConfig config;
  cudaq::config::BackendEndConfigEntry backendConfig;
  backendConfig.PlatformQpu = "mock_observe_qpu";
  config.BackendConfig = backendConfig;
  cudaq::detail::loadTargetPluginLibraries(
      "mock_observe_qpu", targetsDir / "mock_observe_qpu.yml", config);

  EXPECT_TRUE(std::filesystem::exists(sentinelPath));
  unsetenv("CUDAQ_DLOPEN_SENTINEL_PATH");
}

#endif // CUDAQ_ENABLE_PYTHON

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
