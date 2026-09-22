/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/TargetConfigSchema.h"
#ifdef CUDAQ_ENABLE_PYTHON
#include "LinkedLibraryHolder.h"
#include "common/RuntimeTarget.h"
#include "cudaq/platform/qpu_utils.h"
#endif
#include "cudaq/Target/TargetCatalog.h"
#include "cudaq/Target/TargetPluginLibrary.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <unordered_map>

// ExternalBackendTester is not inherently Python-specific, but this test group
// currently uses backend discovery helpers and LinkedLibraryHolder from
// python/utils, which is only available when the Python project is enabled.
#ifdef CUDAQ_ENABLE_PYTHON
namespace {
// Filename a compiled target plugin library gets on this platform.
std::string pluginLibraryName(const std::string &stem) {
  return stem + std::string(cudaq::config::kSharedLibraryExtension);
}

// Compiles `yamlContent` into a target plugin library at
// `targetsDir/<name>` (with the platform's shared library extension) via
// `cudaq-target-db-gen --plugin`, exactly as an external plugin author would
// (see packaging.rst) - external targets are no longer resolved from raw YAML
// text. The YAML is staged at its final `targetsDir/<name>.yml` location just
// long enough to generate/compile (so `%PLUGIN_ROOT%` resolves against the
// real target root), then removed; only the compiled artifact remains,
// matching production behavior.
void compileTargetPluginLibrary(const std::string &name,
                                const std::string &yamlContent,
                                const std::filesystem::path &targetsDir) {
  const auto stagedYml = targetsDir / (name + ".yml");
  {
    std::ofstream out(stagedYml);
    out << yamlContent;
  }
  const auto genCpp = targetsDir / (name + "_target.gen.cpp");
  const auto libPath = targetsDir / pluginLibraryName(name);

  std::string genCmd = std::string(CUDAQ_TARGET_DB_GEN_PATH) + " --plugin -o " +
                       genCpp.string() + " " + name + "=" + stagedYml.string();
  ASSERT_EQ(std::system(genCmd.c_str()), 0) << genCmd;

  std::string compileCmd = std::string(CUDAQ_TEST_CXX_COMPILER) + " " +
                           CUDAQ_TEST_CXX_FLAGS + " -I " +
                           CUDAQ_TEST_INCLUDE_DIR + " " + genCpp.string() +
                           " -o " + libPath.string();
  ASSERT_EQ(std::system(compileCmd.c_str()), 0) << compileCmd;

  std::filesystem::remove(stagedYml);
  std::filesystem::remove(genCpp);
}

std::unordered_map<std::string, cudaq::RuntimeTarget>
loadFromPluginRoot(const std::filesystem::path &pkgRoot) {
  cudaq::config::TargetCatalog registry;
  registry.addPluginRoot(pkgRoot);
  std::unordered_map<std::string, cudaq::RuntimeTarget> targets;
  for (const auto *entry : registry.list()) {
    if (entry->origin == cudaq::config::detail::TargetOrigin::Builtin)
      continue;
    cudaq::RuntimeTarget target;
    target.name = entry->name;
    target.config = *entry->config;
    target.pluginLibDir = entry->pluginLibDir.string();
    target.configPath = entry->configPath;
    target.description = entry->config->Description;
    targets.emplace(target.name, std::move(target));
  }
  return targets;
}
} // namespace

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

    std::ostringstream yaml;
    yaml << "name: " << name << "\ndescription: \"Test backend.\"\n";
    if (!version.empty())
      yaml << "cudaq-version: \"" << version << "\"\n";
    yaml << "config:\n"
         << "  platform-qpu: remote_rest\n  library-mode: false\n";
    compileTargetPluginLibrary(name, yaml.str(), targetsDir);

    if (createSo)
      std::ofstream(libDir / pluginLibraryName("libcudaq-serverhelper-" + name))
          .close();

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
)",
                                                       {});
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
      {"0.9.0", "0.8.1", Compatibility::Warning, "was built for CUDA-Q 0.9.0"},
      {"0.9.2", "0.9.1", Compatibility::Warning, "was built for CUDA-Q 0.9.2"},
      {"0.10.0", "0.9.9", Compatibility::Warning,
       "was built for CUDA-Q 0.10.0"},
      {"0.0.0", "0.0.0", Compatibility::Compatible, ""},
      {"0.9.0", "0.9.0", Compatibility::Compatible, ""},
      {"0.9.0", "0.9.3", Compatibility::Warning,
       "compatibility is not guaranteed"},
      {"0.9.0", "0.10.0", Compatibility::Warning,
       "compatibility is not guaranteed"},
      {"0.9.0", "1.0.0", Compatibility::Warning,
       "compatibility is not guaranteed"},
      {"0.9.0", "0.9.0-rc2-developer", Compatibility::Warning,
       "compatibility is not guaranteed"},
      {"0.9.0", "developer", Compatibility::Warning,
       "compatibility is not guaranteed"},
      {"amd64-pr-1234", "amd64-pr-1234", Compatibility::Compatible, ""},
      {"amd64-pr-1234", "amd64-pr-5678", Compatibility::Warning,
       "compatibility is not guaranteed"},
      {"", "", Compatibility::Compatible, ""},
      {"", "0.9.0", Compatibility::Warning, "compatibility is not guaranteed"},
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

TEST(TargetConfigTester, allVersionDifferencesProduceWarnings) {
  using Compatibility = cudaq::config::TargetVersionCompatibility;
  cudaq::config::TargetConfig config;
  config.Name = "version-test";

  for (const auto *pluginVer : {"", "0.9", "v0.9.0", "not-a-version"}) {
    config.CudaqVersion = pluginVer;
    const auto diffResult = cudaq::config::checkExternalTargetVersion(
        config, "developer", "/tmp/version-test.yml");
    const bool isEqual = std::string(pluginVer) == std::string("developer");
    EXPECT_EQ(diffResult.Status,
              isEqual ? Compatibility::Compatible : Compatibility::Warning)
        << "plugin=" << pluginVer;
    if (!isEqual)
      EXPECT_NE(diffResult.Diagnostic.find("compatibility is not guaranteed"),
                std::string::npos)
          << "plugin=" << pluginVer;
  }

  // Equal non-numeric strings (both empty, both same tag) → Compatible.
  config.CudaqVersion = "amd64-pr-1234";
  const auto equalResult = cudaq::config::checkExternalTargetVersion(
      config, "amd64-pr-1234", "/tmp/version-test.yml");
  EXPECT_EQ(equalResult.Status, Compatibility::Compatible);

  for (const auto *badPlugin : {"", "0.9", "v0.9.0", "0.-1.0"}) {
    config.CudaqVersion = badPlugin;
    const auto result = cudaq::config::checkExternalTargetVersion(
        config, "0.9.0", "/tmp/version-test.yml");
    EXPECT_EQ(result.Status, Compatibility::Warning) << "plugin=" << badPlugin;
    EXPECT_NE(result.Diagnostic.find("compatibility is not guaranteed"),
              std::string::npos)
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

  auto config = cudaq::config::parseTargetConfig(configYmlContents, {});
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

  auto config = cudaq::config::parseTargetConfig(configYmlContents, {});
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

  auto targets = loadFromPluginRoot(root);

  ASSERT_EQ(targets.count("my-backend"), 1);
  EXPECT_EQ(targets.at("my-backend").pluginLibDir, (root / "lib").string());
  EXPECT_EQ(targets.at("my-backend").name, "my-backend");
}

TEST_F(ExternalBackendTester, backendPathMultipleEntries) {
  auto rootA = createBackendPackage("backend-a");
  auto rootB = createBackendPackage("backend-b");

  auto targets = loadFromPluginRoot(rootA);
  auto targetsB = loadFromPluginRoot(rootB);
  targets.insert(targetsB.begin(), targetsB.end());

  ASSERT_EQ(targets.count("backend-a"), 1);
  ASSERT_EQ(targets.count("backend-b"), 1);
  EXPECT_EQ(targets.at("backend-a").pluginLibDir, (rootA / "lib").string());
  EXPECT_EQ(targets.at("backend-b").pluginLibDir, (rootB / "lib").string());
}

TEST_F(ExternalBackendTester, serverHelperPathResolvesToLibDir) {
  auto root = createBackendPackage("my-backend", /*createSo=*/true);

  auto targets = loadFromPluginRoot(root);

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &target = targets.at("my-backend");
  auto resolvedPath = std::filesystem::path(target.pluginLibDir) /
                      pluginLibraryName("libcudaq-serverhelper-" + target.name);
  EXPECT_TRUE(std::filesystem::exists(resolvedPath));
}

TEST_F(ExternalBackendTester, configPath_resolvesToTargetsDir) {
  auto root = createBackendPackage("my-backend");

  auto targets = loadFromPluginRoot(root);

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &target = targets.at("my-backend");
  ASSERT_FALSE(target.pluginLibDir.empty());

  EXPECT_EQ(target.configPath,
            root / "targets" / pluginLibraryName("my-backend"));
  EXPECT_TRUE(std::filesystem::exists(target.configPath));
}

// -- B1: registerBackendPath -------------------------------------------------

TEST_F(ExternalBackendTester, registerBackendPath_addsTargets) {
  auto root = createBackendPackage("my-backend");

  auto targets = loadFromPluginRoot(root);

  ASSERT_EQ(targets.count("my-backend"), 1);
  EXPECT_EQ(targets.at("my-backend").name, "my-backend");
  EXPECT_EQ(targets.at("my-backend").pluginLibDir, (root / "lib").string());
}

TEST_F(ExternalBackendTester, registerBackendPath_rejectsMissingPath) {
  auto bogus = tmpRoot / "does-not-exist";
  cudaq::LinkedLibraryHolder holder;
  try {
    holder.registerBackendPath(bogus);
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

  cudaq::LinkedLibraryHolder holder;
  try {
    holder.registerBackendPath(root);
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find(root.string()), std::string::npos)
        << "error message should mention the offending path: " << e.what();
  }
}

TEST_F(ExternalBackendTester,
       setTargetAllowsMissingOrMalformedPluginVersionMetadata) {
  const auto createVersionBackend = [&](const std::string &name,
                                        const std::string &version) {
    const auto root = tmpRoot / name;
    const auto targetsDir = root / "targets";
    std::filesystem::create_directories(targetsDir);
    std::filesystem::create_directories(root / "lib");

    std::ostringstream yaml;
    yaml << "name: " << name << "\ndescription: \"Test backend.\"\n";
    if (!version.empty())
      yaml << "cudaq-version: \"" << version << "\"\n";
    yaml << "config:\n"
         << "  nvqir-simulation-backend: qpp\n"
         << "  library-mode: false\n";
    compileTargetPluginLibrary(name, yaml.str(), targetsDir);
    return root;
  };

  const auto missingRoot = createVersionBackend("missing-version", "");
  const auto malformedRoot =
      createVersionBackend("malformed-version", "not-a-version");

  cudaq::LinkedLibraryHolder holder;
  holder.registerBackendPath(missingRoot);
  holder.registerBackendPath(malformedRoot);

  EXPECT_NO_THROW(holder.setTarget("missing-version"));
  EXPECT_NO_THROW(holder.setTarget("malformed-version"));
}

TEST_F(ExternalBackendTester, pluginLibrariesFieldIsParsed) {
  auto root = tmpRoot / "pluginlibtest";
  auto targetsDir = root / "targets";
  auto libDir = root / "lib";
  std::filesystem::create_directories(targetsDir);
  std::filesystem::create_directories(libDir);

  // Compile a target plugin library with plugin-libraries set.
  compileTargetPluginLibrary("my-backend", R"(
name: my-backend
description: Plugin-libraries test
target-arguments: []
config:
  platform-qpu: remote_rest
  library-mode: false
  plugin-libraries:
    - libdummy1.so
    - libdummy2.so
)",
                             targetsDir);

  auto targets = loadFromPluginRoot(root);

  ASSERT_EQ(targets.count("my-backend"), 1);
  const auto &target = targets.at("my-backend");
  const auto &libs = target.config.PluginLibraries;
  ASSERT_EQ(libs.size(), 2);
  EXPECT_EQ(libs[0], "libdummy1.so");
  EXPECT_EQ(libs[1], "libdummy2.so");
}

TEST_F(ExternalBackendTester, versionWarningAllowsPluginLibraryLoad) {
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

  std::ostringstream yaml;
  yaml << R"(
name: future-backend
description: Future-version plugin test
cudaq-version: 999999.0.0
target-arguments: []
config:
  nvqir-simulation-backend: qpp
  library-mode: false
  plugin-libraries:
    - )"
       << pluginFileName << "\n";
  compileTargetPluginLibrary("future-backend", yaml.str(), targetsDir);

  cudaq::LinkedLibraryHolder holder;
  holder.registerBackendPath(root);

  EXPECT_FALSE(std::filesystem::exists(sentinelPath));
  EXPECT_NO_THROW(holder.setTarget("future-backend"));
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
  compileTargetPluginLibrary("my-backend", R"(
name: my-backend
description: Plugin-root substitution test
target-arguments: []
config:
  nvqir-simulation-backend: qpp
  jit-mid-level-pipeline: "map{device=file(%PLUGIN_ROOT%/data/topology.txt)}"
  preprocessor-defines:
    - "-DTOPOLOGY=%PLUGIN_ROOT%/data/topology.txt"
)",
                             targetsDir);

  auto targets = loadFromPluginRoot(root);

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
