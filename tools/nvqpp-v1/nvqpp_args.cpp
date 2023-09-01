/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "nvqpp_args.h"
#include <fstream>
#include <regex>
namespace {
// FIXME: these implementations should be located elsewhere (e.g., near the
// ServerHelper impl), not here.
struct IonQTargetPlatformArgs : public cudaq::TargetPlatformArgs {
  virtual cudaq::TargetPlatformArgs::Data
  parsePlatformArgs(ArgvStorageBase &args) override {
    std::string platformExtraArgs;
    // Note: erase args within the loop
    for (auto it = args.begin(); it != args.end(); ++it) {
      auto arg = llvm::StringRef(*it);
      if (arg.equals("--ionq-machine")) {
        platformExtraArgs += std::string(";qpu;");
        platformExtraArgs += std::string(*std::next(it));
        it = args.erase(it, std::next(it, 2));
      }

      if (arg.equals("--ionq-noise-model")) {
        platformExtraArgs += std::string(";noise;");
        platformExtraArgs += std::string(*std::next(it));
        it = args.erase(it, std::next(it, 2));
      }
    }

    return cudaq::TargetPlatformArgs::Data{.genTargetBackend = true,
                                           .linkFlags = {"-lcudaq-rest-qpu"},
                                           .libraryMode = false,
                                           .platformExtraArgs =
                                               platformExtraArgs};
  }
};
struct ConfigFileArgs : public cudaq::TargetPlatformArgs {
  std::string configContents;
  ConfigFileArgs(const std::string &configContents)
      : configContents(configContents){};
  virtual cudaq::TargetPlatformArgs::Data
  parsePlatformArgs(ArgvStorageBase &args) override {
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

    auto lines = split(configContents, '\n');
    std::regex simulationBackend("^NVQIR_SIMULATION_BACKEND\\s*=\\s*(\\S+)");
    std::regex platformConfig("^PLATFORM_LIBRARY\\s*=\\s*(\\S+)");
    std::smatch match;
    std::string backend;
    std::string platform;
    for (const std::string &line : lines) {
      if (std::regex_search(line, match, simulationBackend))
        backend = match[1].str();

      if (std::regex_search(line, match, platformConfig))
        platform = match[1].str();
    }
    if (backend.front() == '"')
      backend.erase(0, 1);
    if (backend.back() == '"')
      backend.pop_back();
    return cudaq::TargetPlatformArgs::Data{.nvqirSimulationBackend = backend,
                                           .nvqirPlatform = platform};
  }
};
} // namespace
namespace cudaq {
std::shared_ptr<TargetPlatformArgs>
getTargetPlatformArgs(const std::string &targetName,
                      const std::filesystem::path &platformPath) {
  std::string fileName = targetName + std::string(".config");
  auto configFilePath = platformPath / fileName;
  std::ifstream configFile(configFilePath.string());
  const std::string configContents((std::istreambuf_iterator<char>(configFile)),
                                   std::istreambuf_iterator<char>());
  static std::unordered_map<std::string, std::shared_ptr<TargetPlatformArgs>>
      TARGET_ARGS_HANDLERS = {
          {"ionq", std::make_shared<IonQTargetPlatformArgs>()}};
  // FIXME: new structure of config file to assist args parsing
  // convert bash script logics into some parsing logic for this code to
  // consume.
  auto iter = TARGET_ARGS_HANDLERS.find(targetName);
  if (iter != TARGET_ARGS_HANDLERS.end())
    return iter->second;

  // NVQIR only
  return std::make_shared<ConfigFileArgs>(configContents);
}
} // namespace cudaq