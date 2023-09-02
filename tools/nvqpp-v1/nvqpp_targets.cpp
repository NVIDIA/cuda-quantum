/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "nvqpp_targets.h"
#include "nvqpp_options.h"
#include <fstream>
#include <iostream>
#include <regex>

namespace cudaq {
TargetPlatformConfig
Target::generateConfig(const llvm::opt::InputArgList &clOptions,
                       const std::filesystem::path &targetConfigSearchPath) {
  TargetPlatformConfig config;
  // Load from file:
  const std::string fileName = name + std::string(".config");
  auto configFilePath = targetConfigSearchPath / fileName;
  std::ifstream configFile(configFilePath.string());
  const std::string configContents((std::istreambuf_iterator<char>(configFile)),
                                   std::istreambuf_iterator<char>());
  auto split = [](const std::string &s, char delim) {
    std::vector<std::string> elems;
    auto internal_split = [](const std::string &s, char delim, auto op) {
      std::stringstream ss(s);
      for (std::string item; std::getline(ss, item, delim);) {
        *op++ = item;
      }
    };
    internal_split(s, delim, std::back_inserter(elems));
    return elems;
  };

  const auto trimQuotes = [](const std::string &str) {
    std::string copy = str;
    if (copy.front() == '"')
      copy.erase(0, 1);
    if (copy.back() == '"')
      copy.pop_back();
    return copy;
  };

  const auto toBool = [](std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    std::istringstream is(str);
    bool b;
    is >> std::boolalpha >> b;
    return b;
  };

  auto lines = split(configContents, '\n');

  std::regex simulationBackend("^NVQIR_SIMULATION_BACKEND\\s*=\\s*(\\S+)");
  std::regex platformConfig("^PLATFORM_LIBRARY\\s*=\\s*(\\S+)");
  std::regex libModeConfig("^LIBRARY_MODE\\s*=\\s*(\\S+)");
  std::regex genTargetBackendConfig("^GEN_TARGET_BACKEND\\s*=\\s*(\\S+)");
  std::regex linkLibConfig("^LINKLIBS\\s*=\\s*(.+)");

  std::smatch match;
  std::string linkLibConfigStr;
  for (const std::string &line : lines) {
    if (std::regex_search(line, match, simulationBackend))
      config.nvqirSimulationBackend = trimQuotes(match[1].str());

    if (std::regex_search(line, match, platformConfig))
      config.nvqirPlatform = trimQuotes(match[1].str());

    if (std::regex_search(line, match, libModeConfig))
      config.libraryMode = toBool(trimQuotes(match[1].str()));

    if (std::regex_search(line, match, genTargetBackendConfig))
      config.genTargetBackend = toBool(trimQuotes(match[1].str()));

    if (std::regex_search(line, match, linkLibConfig))
      linkLibConfigStr = trimQuotes(match[1].str());
  }

  const char *linkPrefix = "${LINKLIBS} ";
  if (linkLibConfigStr.rfind(linkPrefix, 0) == 0) {
    // Start with the prefix
    linkLibConfigStr = linkLibConfigStr.substr(strlen(linkPrefix));
  }

  config.linkFlags = split(linkLibConfigStr, ' ');
  if (platformArgsCtor)
    config.platformExtraArgs = platformArgsCtor(clOptions);
  return config;
}
std::optional<Target>
TargetRegistry::lookupTarget(const std::string &targetName) {
  const auto iter = registry.find(targetName);
  return iter != registry.end() ? iter->second : std::optional<Target>();
}
void TargetRegistry::registerTarget(
    const char *name, const char *desc,
    Target::PlatformExtraArgsCtorFnTy argsCtorFn) {
  // Allow overriding
  registry[name] =
      Target{.name = name, .description = desc, .platformArgsCtor = argsCtorFn};
}
void registerAllTargets() {
  TargetRegistry::registerTarget(
      "ionq", "IonQ hardware and simulator backend",
      [](const llvm::opt::InputArgList &cudaqOptions) -> std::string {
        std::string platformExtraArgs;
        if (cudaqOptions.hasArg(cudaq::nvqpp::options::OPT_ionq_machine)) {
          const std::string ionqMachine =
              cudaqOptions
                  .getLastArgValue(cudaq::nvqpp::options::OPT_ionq_machine, "")
                  .str();
          platformExtraArgs += std::string(";qpu;");
          platformExtraArgs += ionqMachine;
        }
        if (cudaqOptions.hasArg(cudaq::nvqpp::options::OPT_ionq_noise_model)) {
          const std::string noiseModel =
              cudaqOptions
                  .getLastArgValue(cudaq::nvqpp::options::OPT_ionq_noise_model,
                                   "")
                  .str();
          platformExtraArgs += std::string(";noise;");
          platformExtraArgs += noiseModel;
        }

        return platformExtraArgs;
      });

  TargetRegistry::registerTarget(
      "quantinuum", "Quantinuum hardware and simulator backend",
      [](const llvm::opt::InputArgList &cudaqOptions) -> std::string {
        std::string platformExtraArgs;
        if (cudaqOptions.hasArg(cudaq::nvqpp::options::OPT_quantinuum_url)) {
          const std::string quantinuumUrl =
              cudaqOptions
                  .getLastArgValue(cudaq::nvqpp::options::OPT_quantinuum_url,
                                   "")
                  .str();
          platformExtraArgs += std::string(";url;");
          platformExtraArgs += quantinuumUrl;
        }
        if (cudaqOptions.hasArg(
                cudaq::nvqpp::options::OPT_quantinuum_machine)) {
          const std::string quantinuumMachine =
              cudaqOptions
                  .getLastArgValue(
                      cudaq::nvqpp::options::OPT_quantinuum_machine, "")
                  .str();
          platformExtraArgs += std::string(";machine;");
          platformExtraArgs += quantinuumMachine;
        }

        return platformExtraArgs;
      });
}
} // namespace cudaq