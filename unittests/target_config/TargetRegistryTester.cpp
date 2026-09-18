/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "TargetConfigHelper.h"
#include "cudaq/Target/TargetPluginLibrary.h"
#include "cudaq/Target/TargetRegistry.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path makeTempRoot() {
  auto root =
      std::filesystem::temp_directory_path() /
      ("cudaq_registry_" +
       std::string(
           ::testing::UnitTest::GetInstance()->current_test_info()->name()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  return root;
}

void writeFile(const std::filesystem::path &path, const std::string &contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << contents;
}

cudaq::config::HostEnvironment hostWithLibs(const std::filesystem::path &libDir,
                                            unsigned gpuCount = 0) {
  cudaq::config::HostEnvironment env;
  env.gpuCount = gpuCount;
  env.cudaqVersion = "0.0.0";
  env.libraryPaths.push_back(libDir);
  return env;
}

} // namespace

TEST(TargetRegistryTester, schemaVersionAbsentIsOne) {
  auto config = cudaq::config::parseTargetConfig(R"(
name: version-absent
description: absent version means 1
config:
  library-mode: true
)");
  EXPECT_EQ(config.Name, "version-absent");
}

TEST(TargetRegistryTester, schemaVersionOneAccepted) {
  auto config = cudaq::config::parseTargetConfig(R"(
version: 1
name: version-one
description: explicit version 1
config:
  library-mode: true
)");
  EXPECT_EQ(config.Name, "version-one");
}

TEST(TargetRegistryTester, schemaVersionOtherRejected) {
  EXPECT_THROW(cudaq::config::parseTargetConfig(R"(
version: 2
name: version-two
description: unsupported
config:
  library-mode: true
)"),
               std::runtime_error);
}

TEST(TargetRegistryTester, prefersPluginLibraryOverYamlInSameRoot) {
  auto root = makeTempRoot();
  const auto ymlPath = root / "targets" / "pref.yml";
  writeFile(ymlPath, R"(
version: 1
name: pref
description: compiled description
config:
  library-mode: true
  preprocessor-defines: ["-DFROM_SO"]
)");

  const auto genCpp = std::filesystem::temp_directory_path() / "pref.gen.cpp";
  const auto libPath =
      root / "targets" /
      ("pref" + std::string(cudaq::config::kSharedLibraryExtension));
  std::string genCmd = std::string(CUDAQ_TARGET_DB_GEN_PATH) + " --plugin -o " +
                       genCpp.string() + " pref=" + ymlPath.string();
  ASSERT_EQ(std::system(genCmd.c_str()), 0) << genCmd;
  std::string compileCmd = std::string(CUDAQ_TEST_CXX_COMPILER) + " " +
                           CUDAQ_TEST_CXX_FLAGS + " -I " +
                           CUDAQ_TEST_INCLUDE_DIR + " " + genCpp.string() +
                           " -o " + libPath.string();
  ASSERT_EQ(std::system(compileCmd.c_str()), 0) << compileCmd;
  writeFile(ymlPath, R"(
version: 1
name: pref
description: yaml description
config:
  library-mode: true
  preprocessor-defines: ["-DFROM_YAML"]
)");

  cudaq::config::TargetRegistry registry;
  auto added = registry.addPluginRoot(root);
  ASSERT_EQ(added.size(), 1u);
  const auto *entry = registry.lookup("pref");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->origin, cudaq::config::detail::TargetOrigin::PluginLibrary);
  EXPECT_EQ(entry->configPath, libPath);
  ASSERT_TRUE(entry->config->BackendConfig.has_value());
  ASSERT_FALSE(entry->config->BackendConfig->PreprocessorDefines.empty());
  EXPECT_EQ(entry->config->BackendConfig->PreprocessorDefines.front(),
            "-DFROM_SO");
  std::filesystem::remove_all(root);
  std::filesystem::remove(genCpp);
}

TEST(TargetRegistryTester, yamlOnlyPluginRootIsLoaded) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "yamlonly.yml", R"(
version: 1
name: yamlonly
description: yaml-only plugin
config:
  library-mode: true
  preprocessor-defines: ["-DYAML_ONLY"]
)");

  cudaq::config::TargetRegistry registry;
  auto added = registry.addPluginRoot(root);
  ASSERT_EQ(added.size(), 1u);
  const auto *entry = registry.lookup("yamlonly");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->origin, cudaq::config::detail::TargetOrigin::YamlFile);
  ASSERT_TRUE(entry->config->BackendConfig.has_value());
  EXPECT_EQ(entry->config->BackendConfig->PreprocessorDefines.front(),
            "-DYAML_ONLY");
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, addsStandaloneTargetConfigFile) {
  auto root = makeTempRoot();
  // Note: directly under `root`, with no `targets/` or `lib/` layout.
  const auto configPath = root / "standalone.yml";
  writeFile(configPath, R"(
version: 1
name: standalone
description: standalone target YAML
config:
  library-mode: true
  preprocessor-defines: ["-DSTANDALONE"]
)");

  cudaq::config::TargetRegistry registry;
  ASSERT_TRUE(registry.addTargetConfigFile(configPath));
  const auto *entry = registry.lookup("standalone");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->origin, cudaq::config::detail::TargetOrigin::YamlFile);
  EXPECT_EQ(entry->configPath, configPath);
  EXPECT_TRUE(entry->pluginLibDir.empty());
  ASSERT_TRUE(entry->config->BackendConfig.has_value());
  EXPECT_EQ(entry->config->BackendConfig->PreprocessorDefines.front(),
            "-DSTANDALONE");

  // Registering the same name twice reports failure rather than shadowing.
  EXPECT_FALSE(registry.addTargetConfigFile(configPath));
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, standaloneTargetConfigCannotShadowBuiltin) {
  auto root = makeTempRoot();
  const auto configPath = root / "qpp-cpu.yml";
  writeFile(configPath, R"(
version: 1
name: qpp-cpu
description: should not shadow
config:
  library-mode: true
)");

  cudaq::config::TargetRegistry registry;
  EXPECT_FALSE(registry.addTargetConfigFile(configPath));
  EXPECT_EQ(registry.lookup("qpp-cpu")->origin,
            cudaq::config::detail::TargetOrigin::Builtin);
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, refusesToShadowBuiltin) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "qpp-cpu.yml", R"(
version: 1
name: qpp-cpu
description: should not shadow
config:
  library-mode: true
)");
  cudaq::config::TargetRegistry registry;
  auto added = registry.addPluginRoot(root);
  EXPECT_TRUE(added.empty());
  const auto *entry = registry.lookup("qpp-cpu");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->origin, cudaq::config::detail::TargetOrigin::Builtin);
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, requiresGpuAvailability) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "needs-gpu.yml", R"(
version: 1
name: needs-gpu
description: gpu required
gpu-requirements: true
config:
  library-mode: true
)");
  cudaq::config::TargetRegistry registry;
  registry.addPluginRoot(root);
  auto env = hostWithLibs(root / "lib", /*gpuCount=*/0);
  auto resolved = registry.resolve("needs-gpu", env);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->status.availability,
            cudaq::config::detail::Availability::RequiresGpu);
  EXPECT_FALSE(resolved->status.diagnostic.empty());

  env.gpuCount = 1;
  resolved = registry.resolve("needs-gpu", env);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_TRUE(resolved->status.isAvailable());
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, missingSimulatorAvailability) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "needs-sim.yml", R"(
version: 1
name: needs-sim
description: missing simulator
config:
  nvqir-simulation-backend: does-not-exist
)");
  cudaq::config::TargetRegistry registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("needs-sim", hostWithLibs(root / "lib"));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->status.availability,
            cudaq::config::detail::Availability::MissingSimulator);
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, multiSimulatorFallback) {
  auto root = makeTempRoot();
  auto libDir = root / "lib";
  std::filesystem::create_directories(libDir);
  writeFile(libDir / ("libnvqir-second" +
                      std::string(cudaq::config::kSharedLibraryExtension)),
            "stub");
  writeFile(root / "targets" / "multi-sim.yml", R"(
version: 1
name: multi-sim
description: fallback simulators
config:
  nvqir-simulation-backend: first, second
)");
  cudaq::config::TargetRegistry registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("multi-sim", hostWithLibs(libDir));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_TRUE(resolved->status.isAvailable());
  EXPECT_EQ(resolved->resolved.simulatorName, "second");
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, missingPlatformLibraryAvailability) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "needs-plat.yml", R"(
version: 1
name: needs-plat
description: missing platform
config:
  platform-library: does-not-exist
)");
  cudaq::config::TargetRegistry registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("needs-plat", hostWithLibs(root / "lib"));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->status.availability,
            cudaq::config::detail::Availability::MissingPlatformLibrary);
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, missingPluginLibraryAvailability) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "needs-plugin.yml", R"(
version: 1
name: needs-plugin
description: missing plugin lib
config:
  library-mode: true
  plugin-libraries:
    - libmissing-plugin.so
)");
  cudaq::config::TargetRegistry registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("needs-plugin", hostWithLibs(root / "lib"));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->status.availability,
            cudaq::config::detail::Availability::MissingPluginLibrary);
  std::filesystem::remove_all(root);
}

// A target configuration is platform independent, so a plugin library named
// with another platform's extension must still resolve against the file this
// platform actually ships.
TEST(TargetRegistryTester, pluginLibraryExtensionIsPlatformIndependent) {
  auto root = makeTempRoot();
  auto libDir = root / "lib";
  writeFile(libDir / ("libforeign-plugin" +
                      std::string(cudaq::config::kSharedLibraryExtension)),
            "stub");
  writeFile(root / "targets" / "foreign-ext.yml", R"(
version: 1
name: foreign-ext
description: plugin library named with a foreign extension
config:
  library-mode: true
  plugin-libraries:
    - libforeign-plugin.dylib
    - libforeign-plugin.so
    - libforeign-plugin
)");
  cudaq::config::TargetRegistry registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("foreign-ext", hostWithLibs(libDir));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_TRUE(resolved->status.isAvailable()) << resolved->status.diagnostic;
  std::filesystem::remove_all(root);
}

TEST(TargetRegistryTester, userScopeBeforeSystemScope) {
  auto user = makeTempRoot() / "user";
  auto system = makeTempRoot() / "system";
  writeFile(user / "targets" / "shared.yml", R"(
version: 1
name: shared
description: from user
config:
  library-mode: true
  preprocessor-defines: ["-DFROM_USER"]
)");
  writeFile(system / "targets" / "shared.yml", R"(
version: 1
name: shared
description: from system
config:
  library-mode: true
  preprocessor-defines: ["-DFROM_SYSTEM"]
)");
  cudaq::config::TargetRegistry registry;
  registry.addPluginRoot(user);
  registry.addPluginRoot(system);
  const auto *entry = registry.lookup("shared");
  ASSERT_NE(entry, nullptr);
  ASSERT_TRUE(entry->config->BackendConfig.has_value());
  EXPECT_EQ(entry->config->BackendConfig->PreprocessorDefines.front(),
            "-DFROM_USER");
  std::filesystem::remove_all(user.parent_path());
  std::filesystem::remove_all(system.parent_path());
}
