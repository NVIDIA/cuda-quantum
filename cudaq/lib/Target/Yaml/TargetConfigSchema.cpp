/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/TargetConfigSchema.h"
#include "TargetConfigHelper.h"
#include "cudaq/Target/TargetCatalog.h"
#include <atomic>
#include <fstream>
#include <rfl/yaml.hpp>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

cudaq::config::TargetConfig
cudaq::config::createTargetConfigFromSchema(const TargetConfigSchema &schema) {
  return schema.visit(
      [](const v1::TargetConfigV1 &cfg) { return v1::toCanonical(cfg); });
}

cudaq::config::TargetConfig
cudaq::config::parseTargetConfig(const std::string &substitutedYamlContent) {
  YAML::Node node;
  try {
    node = YAML::Load(substitutedYamlContent);
  } catch (const YAML::Exception &e) {
    throw std::runtime_error("Failed to parse target configuration YAML: " +
                             std::string(e.what()));
  }
  // Unversioned configurations are schema v1, permanently.
  if (node.IsMap() && !node["version"])
    node["version"] = "1";
  const auto parsed = rfl::yaml::read<TargetConfigSchema, YamlProcessors>(
      rfl::yaml::InputVarType(node), substitutedYamlContent);
  if (!parsed)
    throw std::runtime_error("Failed to parse target configuration YAML: " +
                             parsed.error().what());
  try {
    return createTargetConfigFromSchema(parsed.value());
  } catch (const std::runtime_error &e) {
    throw std::runtime_error("Failed to parse target configuration YAML: " +
                             std::string(e.what()));
  }
}

cudaq::config::TargetConfig
cudaq::config::parseTargetConfig(std::string yamlContent,
                                 const std::filesystem::path &pluginRoot) {
  return parseTargetConfig(
      substitutePluginRoot(std::move(yamlContent), pluginRoot));
}

std::string
cudaq::config::substitutePluginRoot(std::string yamlContent,
                                    const std::filesystem::path &pluginRoot) {
  static constexpr std::string_view token = "%PLUGIN_ROOT%";
  const auto rootPath =
      pluginRoot.empty() ? std::filesystem::current_path() : pluginRoot;
  const auto root =
      std::filesystem::absolute(rootPath).lexically_normal().string();

  std::size_t pos = 0;
  while ((pos = yamlContent.find(token, pos)) != std::string::npos) {
    yamlContent.replace(pos, token.size(), root);
    pos += root.size();
  }

  return yamlContent;
}

namespace {
std::atomic<bool> yamlParsingDisabled{false};
}

void cudaq::config::disableYAMLTargetConfigParsing() {
  yamlParsingDisabled.store(true, std::memory_order_relaxed);
}

bool cudaq::config::isDisabledYAMLParsing() {
  return yamlParsingDisabled.load(std::memory_order_relaxed);
}

cudaq::config::TargetConfig
cudaq::config::loadTargetConfig(const std::filesystem::path &configPath,
                                const std::filesystem::path &pluginRoot) {
  if (isDisabledYAMLParsing())
    throw std::runtime_error(
        "Loading target configurations from YAML is disabled; only "
        "pre-compiled target plugin libraries are accepted");
  std::ifstream configFile(configPath.string());
  if (!configFile.is_open())
    throw std::runtime_error("Unable to open target configuration file: " +
                             configPath.string());
  std::string yamlContent((std::istreambuf_iterator<char>(configFile)),
                          std::istreambuf_iterator<char>());
  if (configFile.bad())
    throw std::runtime_error("Unable to read target configuration file: " +
                             configPath.string());
  const auto root =
      pluginRoot.empty() ? configPath.parent_path().parent_path() : pluginRoot;
  return cudaq::config::parseTargetConfig(std::move(yamlContent), root);
}
