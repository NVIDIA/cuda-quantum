/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetCatalog.h"
#include "cudaq/Target/TargetConfig.h"
#include "cudaq/Target/TargetPluginLibrary.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <variant>
#include <vector>

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

TEST(TargetCatalogTester, prefersPluginLibraryOverYamlInSameRoot) {
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

  const auto genCpp = root / "pref.gen.cpp";
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

  cudaq::config::TargetCatalog registry;
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

TEST(TargetCatalogTester, v2MultiStagePluginLibraryRoundTrip) {
  auto root = makeTempRoot();
  const auto ymlPath = root / "targets" / "multistage.yml";
  writeFile(ymlPath, R"(
version: 2
name: multistage
description: multi-stage v2 plugin
config:
  library-mode: true

compile-target:
  stages:
    - ir-output-profile:
        name: unchecked
      lowering:
        custom-pipeline: "canonicalize"
    - ir-output-profile:
        name: quake-reference-semantics
        options:
          basis-gates: ["h", "rz"]
          measurement-type: single-qubit
  codegen: qir-base
)");

  const auto genCpp = root / "multistage.gen.cpp";
  const auto libPath =
      root / "targets" /
      ("multistage" + std::string(cudaq::config::kSharedLibraryExtension));
  std::string genCmd = std::string(CUDAQ_TARGET_DB_GEN_PATH) + " --plugin -o " +
                       genCpp.string() + " multistage=" + ymlPath.string();
  ASSERT_EQ(std::system(genCmd.c_str()), 0) << genCmd;
  std::string compileCmd = std::string(CUDAQ_TEST_CXX_COMPILER) + " " +
                           CUDAQ_TEST_CXX_FLAGS + " -I " +
                           CUDAQ_TEST_INCLUDE_DIR + " " + genCpp.string() +
                           " -o " + libPath.string();
  ASSERT_EQ(std::system(compileCmd.c_str()), 0) << compileCmd;

  cudaq::config::TargetCatalog registry;
  auto added = registry.addPluginRoot(root);
  ASSERT_EQ(added.size(), 1u);
  const auto *entry = registry.lookup("multistage");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->origin, cudaq::config::detail::TargetOrigin::PluginLibrary);
  ASSERT_TRUE(entry->config->BackendConfig.has_value());
  const auto &stages = entry->config->BackendConfig->CompileTarget.stages;
  ASSERT_EQ(stages.size(), 2u);
  // Stage 1: custom pipeline lowering to the unchecked profile.
  const auto *custom =
      std::get_if<cudaq::config::CustomPipelineLowering>(&stages[0].lowering);
  ASSERT_NE(custom, nullptr);
  EXPECT_EQ(custom->overridePassPipeline, "canonicalize");
  EXPECT_TRUE(std::holds_alternative<cudaq::config::UncheckedProfile>(
      stages[0].irOutputProfile));
  // Stage 2: inferred lowering to the physical profile, options preserved.
  const auto *physical = std::get_if<cudaq::config::QuakeReferenceSemanticsProfile>(
      &stages[1].irOutputProfile);
  ASSERT_NE(physical, nullptr);
  EXPECT_EQ(physical->basisGates, (std::vector<std::string>{"h", "rz"}));
  EXPECT_EQ(physical->measurementType,
            cudaq::config::MeasurementForm::single_qubit);
  EXPECT_FALSE(physical->resetAfterMeasure);
  EXPECT_TRUE(std::holds_alternative<cudaq::config::InferredLowering>(
      stages[1].lowering));
  EXPECT_EQ(entry->config->BackendConfig->CompileTarget.codegen.emissionFormat,
            "qir-base");
  std::filesystem::remove_all(root);
  std::filesystem::remove(genCpp);
}

TEST(TargetCatalogTester, v2EmptyStagesPluginLibraryRoundTrip) {
  auto root = makeTempRoot();
  const std::string emptyStagesYaml = R"(
version: 2
name: emptystages
description: empty-stages v2 plugin
config:
  library-mode: true

compile-target:
  stages: []
  codegen: qir-base
)";
  const auto yamlRoot = root / "yaml";
  const auto ymlPath = yamlRoot / "targets" / "emptystages.yml";
  writeFile(ymlPath, emptyStagesYaml);

  // YAML catalog loading accepts the explicit empty list.
  cudaq::config::TargetCatalog yamlRegistry;
  ASSERT_EQ(yamlRegistry.addPluginRoot(yamlRoot).size(), 1u);
  const auto *yamlEntry = yamlRegistry.lookup("emptystages");
  ASSERT_NE(yamlEntry, nullptr);
  ASSERT_TRUE(yamlEntry->config->BackendConfig.has_value());
  const auto &yamlTarget = yamlEntry->config->BackendConfig->CompileTarget;
  EXPECT_TRUE(yamlTarget.stages.empty());
  EXPECT_EQ(yamlTarget.codegen.emissionFormat, "qir-base");
  EXPECT_EQ(yamlTarget.getOverridePassPipeline(), nullptr);
  EXPECT_FALSE(yamlTarget.hasPassPipeline());

  const auto genCpp = root / "emptystages.gen.cpp";
  const auto libPath =
      root / "plugin" / "targets" /
      ("emptystages" + std::string(cudaq::config::kSharedLibraryExtension));
  std::filesystem::create_directories(libPath.parent_path());
  std::string genCmd = std::string(CUDAQ_TARGET_DB_GEN_PATH) + " --plugin -o " +
                       genCpp.string() + " emptystages=" + ymlPath.string();
  ASSERT_EQ(std::system(genCmd.c_str()), 0) << genCmd;
  std::string compileCmd = std::string(CUDAQ_TEST_CXX_COMPILER) + " " +
                           CUDAQ_TEST_CXX_FLAGS + " -I " +
                           CUDAQ_TEST_INCLUDE_DIR + " " + genCpp.string() +
                           " -o " + libPath.string();
  ASSERT_EQ(std::system(compileCmd.c_str()), 0) << compileCmd;

  cudaq::config::TargetCatalog registry;
  ASSERT_EQ(registry.addPluginRoot(root / "plugin").size(), 1u);
  const auto *entry = registry.lookup("emptystages");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->origin, cudaq::config::detail::TargetOrigin::PluginLibrary);
  ASSERT_TRUE(entry->config->BackendConfig.has_value());
  const auto &pluginTarget = entry->config->BackendConfig->CompileTarget;
  // The generated plugin round trips the same empty-stages target.
  EXPECT_EQ(pluginTarget, yamlTarget);
  EXPECT_TRUE(pluginTarget.stages.empty());
  EXPECT_EQ(pluginTarget.codegen.emissionFormat, "qir-base");
  EXPECT_EQ(pluginTarget.getOverridePassPipeline(), nullptr);
  EXPECT_FALSE(pluginTarget.hasPassPipeline());

  // Native construction preserves zero stages and the codegen selection.
  const auto native =
      cudaq::CompileTarget::createFromConfig(*entry->config, {});
  EXPECT_TRUE(native.stages.empty());
  EXPECT_EQ(native.codegen.emissionFormat, "qir-base");
  EXPECT_EQ(native.getOverridePassPipeline(), nullptr);
  EXPECT_FALSE(native.hasPassPipeline());
  std::filesystem::remove_all(root);
  std::filesystem::remove(genCpp);
}

TEST(TargetCatalogTester, yamlOnlyPluginRootIsLoaded) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "yamlonly.yml", R"(
version: 1
name: yamlonly
description: yaml-only plugin
config:
  library-mode: true
  preprocessor-defines: ["-DYAML_ONLY"]
)");

  cudaq::config::TargetCatalog registry;
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

TEST(TargetCatalogTester, addsStandaloneTargetConfigFile) {
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

  cudaq::config::TargetCatalog registry;
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

TEST(TargetCatalogTester, standaloneTargetConfigCannotShadowBuiltin) {
  auto root = makeTempRoot();
  const auto configPath = root / "qpp-cpu.yml";
  writeFile(configPath, R"(
version: 1
name: qpp-cpu
description: should not shadow
config:
  library-mode: true
)");

  cudaq::config::TargetCatalog registry;
  EXPECT_FALSE(registry.addTargetConfigFile(configPath));
  EXPECT_EQ(registry.lookup("qpp-cpu")->origin,
            cudaq::config::detail::TargetOrigin::Builtin);
  std::filesystem::remove_all(root);
}

TEST(TargetCatalogTester, refusesToShadowBuiltin) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "qpp-cpu.yml", R"(
version: 1
name: qpp-cpu
description: should not shadow
config:
  library-mode: true
)");
  cudaq::config::TargetCatalog registry;
  auto added = registry.addPluginRoot(root);
  EXPECT_TRUE(added.empty());
  const auto *entry = registry.lookup("qpp-cpu");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->origin, cudaq::config::detail::TargetOrigin::Builtin);
  std::filesystem::remove_all(root);
}

TEST(TargetCatalogTester, requiresGpuAvailability) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "needs-gpu.yml", R"(
version: 1
name: needs-gpu
description: gpu required
gpu-requirements: true
config:
  library-mode: true
)");
  cudaq::config::TargetCatalog registry;
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

TEST(TargetCatalogTester, missingSimulatorAvailability) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "needs-sim.yml", R"(
version: 1
name: needs-sim
description: missing simulator
config:
  nvqir-simulation-backend: does-not-exist
)");
  cudaq::config::TargetCatalog registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("needs-sim", hostWithLibs(root / "lib"));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->status.availability,
            cudaq::config::detail::Availability::MissingSimulator);
  std::filesystem::remove_all(root);
}

TEST(TargetCatalogTester, multiSimulatorFallback) {
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
  cudaq::config::TargetCatalog registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("multi-sim", hostWithLibs(libDir));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_TRUE(resolved->status.isAvailable());
  EXPECT_EQ(resolved->resolved.simulatorName, "second");
  std::filesystem::remove_all(root);
}

TEST(TargetCatalogTester, missingPlatformLibraryAvailability) {
  auto root = makeTempRoot();
  writeFile(root / "targets" / "needs-plat.yml", R"(
version: 1
name: needs-plat
description: missing platform
config:
  platform-library: does-not-exist
)");
  cudaq::config::TargetCatalog registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("needs-plat", hostWithLibs(root / "lib"));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->status.availability,
            cudaq::config::detail::Availability::MissingPlatformLibrary);
  std::filesystem::remove_all(root);
}

TEST(TargetCatalogTester, missingPluginLibraryAvailability) {
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
  cudaq::config::TargetCatalog registry;
  registry.addPluginRoot(root);
  auto resolved = registry.resolve("needs-plugin", hostWithLibs(root / "lib"));
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->status.availability,
            cudaq::config::detail::Availability::MissingPluginLibrary);
  std::filesystem::remove_all(root);
}

TEST(TargetCatalogTester, userScopeBeforeSystemScope) {
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
  cudaq::config::TargetCatalog registry;
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
