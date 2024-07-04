/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "llvm/Support/Allocator.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "target-config"

//===----------------------------------------------------------------------===//
// Command line options.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
    inputConfigFile(llvm::cl::Positional,
                    llvm::cl::desc("<input target config YAML file>"),
                    llvm::cl::init("-"), llvm::cl::value_desc("filename"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Specify output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    targetArgs("arg", llvm::cl::desc("Specify target CLI arguments"),
               llvm::cl::value_desc("string"), llvm::cl::init("-"));

static constexpr const char BOLD[] = "\033[1m";
static constexpr const char RED[] = "\033[91m";
static constexpr const char CLEAR[] = "\033[0m";

using namespace llvm;
using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
using llvm::yaml::ScalarEnumerationTraits;

static void checkErrorCode(const std::error_code &ec) {
  if (ec) {
    llvm::errs() << "could not open output file";
    std::exit(ec.value());
  }
}

enum ArgumentType { String, Int, UUID, Flag };
namespace llvm {
namespace yaml {
template <>
struct ScalarEnumerationTraits<ArgumentType> {
  static void enumeration(IO &io, ArgumentType &value) {
    io.enumCase(value, "string", String);
    io.enumCase(value, "integer", Int);
    io.enumCase(value, "uuid", UUID);
    io.enumCase(value, "flag", Flag);
  }
};
} // namespace yaml
} // namespace llvm

struct TargetArgument {
  std::string KeyName;
  bool IsRequired = false;
  std::string PlatformArgKey;
  std::string HelpString;
  ArgumentType Type = ArgumentType::String;
  std::string DefaultValue;
  std::vector<std::string> ValidValues;
};

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<TargetArgument> {
  static void mapping(IO &io, TargetArgument &info) {
    io.mapRequired("key", info.KeyName);
    io.mapOptional("required", info.IsRequired);
    io.mapOptional("platform-arg", info.PlatformArgKey);
    io.mapOptional("help-string", info.HelpString);
    io.mapOptional("type", info.Type);
    io.mapOptional("default", info.DefaultValue);
    io.mapOptional("valid-values", info.ValidValues);
  }
};
} // namespace yaml
} // namespace llvm
LLVM_YAML_IS_SEQUENCE_VECTOR(TargetArgument)

enum TargetFeatureFlag {
  flagsFP32 = 1,
  flagsFP64 = 2,
  flagsMgpu = 4,
  flagsMqpu = 8,
};
namespace llvm {
namespace yaml {
template <>
struct ScalarEnumerationTraits<TargetFeatureFlag> {
  static void enumeration(IO &io, TargetFeatureFlag &value) {
    io.enumCase(value, "fp32", flagsFP32);
    io.enumCase(value, "fp64", flagsFP64);
    io.enumCase(value, "mgpu", flagsMgpu);
    io.enumCase(value, "mqpu", flagsMqpu);
  }
};
} // namespace yaml
} // namespace llvm

struct BackendEndConfigEntry {
  bool GenTargetBackend = false;
  bool LibraryMode = true;
  std::string PlatformLoweringConfig;
  std::string CodegenEmission;
  std::string PostCodeGenPasses;
  std::string PlatformLibrary;
  std::string PlatformQpu;
  std::vector<std::string> PreprocessorDefines;
  std::vector<std::string> CompilerFlags;
  std::vector<std::string> LinkLibs;
  std::vector<std::string> LinkerFlags;
  std::string SimulationBackend;
  std::string ConfigBashCommands;
};

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<BackendEndConfigEntry> {
  static void mapping(IO &io, BackendEndConfigEntry &info) {
    io.mapRequired("gen-target-backend", info.GenTargetBackend);
    io.mapOptional("library-mode", info.LibraryMode);
    io.mapOptional("platform-lowering-config", info.PlatformLoweringConfig);
    io.mapOptional("codegen-emission", info.CodegenEmission);
    io.mapOptional("post-codegen-passes", info.PostCodeGenPasses);
    io.mapOptional("platform-library", info.PlatformLibrary);
    io.mapOptional("platform-qpu", info.PlatformQpu);
    io.mapOptional("preprocessor-defines", info.PreprocessorDefines);
    io.mapOptional("compiler-flags", info.CompilerFlags);
    io.mapOptional("link-libs", info.LinkLibs);
    io.mapOptional("linker-flags", info.LinkerFlags);
    io.mapOptional("simulation-backend", info.SimulationBackend);
    io.mapOptional("bash", info.ConfigBashCommands);
  }
};
} // namespace yaml
} // namespace llvm

struct BackendFeatureMap {
  std::string Name;
  std::vector<TargetFeatureFlag> Flags;
  BackendEndConfigEntry Config;
};

struct TargetConfig {
  std::string Name;
  std::string Description;
  std::vector<TargetArgument> TargetArguments;
  bool GpuRequired = false;

  std::optional<BackendEndConfigEntry> BackendConfig;
  std::vector<BackendFeatureMap> ConfigMap;
};

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<TargetConfig> {
  static void mapping(IO &io, TargetConfig &info) {
    io.mapRequired("name", info.Name);
    io.mapRequired("description", info.Description);
    io.mapOptional("target-arguments", info.TargetArguments);
    io.mapOptional("gpu-requirements", info.GpuRequired);
    io.mapOptional("config", info.BackendConfig);
  }
};
} // namespace yaml
} // namespace llvm

std::string processRuntimeArgs(const TargetConfig &config,
                               const std::vector<std::string> &targetArgv) {
  std::stringstream output;
  if (config.BackendConfig.has_value()) {
    auto configValue = config.BackendConfig.value();
    output << "GEN_TARGET_BACKEND="
           << (configValue.GenTargetBackend ? "true" : "false") << "\n";
    output << "LIBRARY_MODE=" << (configValue.LibraryMode ? "true" : "false")
           << "\n";

    if (!configValue.PlatformLoweringConfig.empty()) {
      output << "PLATFORM_LOWERING_CONFIG="
             << configValue.PlatformLoweringConfig << "\n";
    }
    if (!configValue.CodegenEmission.empty()) {
      output << "CODEGEN_EMISSION=" << configValue.CodegenEmission << "\n";
    }
    if (!configValue.PostCodeGenPasses.empty()) {
      output << "POST_CODEGEN_PASSES=" << configValue.PostCodeGenPasses << "\n";
    }
    if (!configValue.PlatformLibrary.empty()) {
      output << "PLATFORM_LIBRARY=" << configValue.PlatformLibrary << "\n";
    }
    if (!configValue.PlatformQpu.empty()) {
      output << "PLATFORM_QPU=" << configValue.PlatformQpu << "\n";
    }

    if (!configValue.PreprocessorDefines.empty()) {
      output << "PREPROCESSOR_DEFINES=\"${PREPROCESSOR_DEFINES}";
      for (const auto &def : configValue.PreprocessorDefines) {
        output << " " << def;
      }
      output << "\"\n";
    }

    if (!configValue.CompilerFlags.empty()) {
      output << "COMPILER_FLAGS=\"${COMPILER_FLAGS}";
      for (const auto &def : configValue.CompilerFlags) {
        output << " " << def;
      }
      output << "\"\n";
    }

    if (!configValue.LinkLibs.empty()) {
      output << "LINKLIBS=\"${LINKLIBS}";
      for (const auto &lib : configValue.LinkLibs) {
        output << " " << lib;
      }
      output << "\"\n";
    }

    if (!configValue.LinkerFlags.empty()) {
      output << "LINKER_FLAGS=\"${LINKER_FLAGS}";
      for (const auto &def : configValue.LinkerFlags) {
        output << " " << def;
      }
      output << "\"\n";
    }

    if (!configValue.SimulationBackend.empty()) {
      output << "NVQIR_SIMULATION_BACKEND=" << configValue.SimulationBackend
             << "\n";
    }

    if (!configValue.ConfigBashCommands.empty()) {
      output << configValue.ConfigBashCommands << "\n";
    }
  }

  std::stringstream platformExtraArgs;
  for (std::size_t idx = 0; idx < targetArgv.size();) {
    const auto argsStr = targetArgv[idx];
    const auto iter = std::find_if(
        config.TargetArguments.begin(), config.TargetArguments.end(),
        [&](const TargetArgument &argConfig) {
          const std::string argKey =
              "--" + config.Name + "-" + argConfig.KeyName;
          return (argKey == argsStr);
        });
    if (iter == config.TargetArguments.end()) {
      llvm::errs() << "Unknown target argument '" << argsStr << "'\n";
      llvm::errs() << "Supported arguments for target '" << config.Name
                   << "' are: " << "\n";
      for (const auto &argConfig : config.TargetArguments) {
        llvm::errs() << "  " << "--" + config.Name + "-" + argConfig.KeyName;
        if (!argConfig.HelpString.empty()) {
          llvm::errs() << " (" << argConfig.HelpString << ")";
        }

        llvm::errs() << "\n";
      }
      abort();
    } else {
      if (iter->Type != ArgumentType::Flag) {
        if (!iter->PlatformArgKey.empty()) {
          platformExtraArgs << ";" << iter->PlatformArgKey << ";"
                            << targetArgv[idx + 1];
        }

        idx += 2;

      } else {
        idx += 1;
      }
    }
  }
  const auto platformExtraArgsStr = platformExtraArgs.str();
  if (!platformExtraArgsStr.empty()) {
    output << "PLATFORM_EXTRA_ARGS=\"${PLATFORM_EXTRA_ARGS}"
           << platformExtraArgsStr << "\"\n";
  }
  return output.str();
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "CUDA-Q Target Build Configuration Generator\n");

  LLVM_DEBUG(llvm::dbgs() << "Using configuration file " << inputConfigFile
                          << "\n");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputConfigFile);
  if (std::error_code ec = fileOrErr.getError())
    checkErrorCode(ec);
  TargetConfig config;
  llvm::yaml::Input Input(*(fileOrErr.get()));
  Input >> config;
  llvm::SmallVector<llvm::StringRef> args;
  std::string targetArgsString = targetArgs;
  if (targetArgsString.starts_with("base64_")) {
    if (targetArgsString.size() > 7) {
      auto targetArgsStr = targetArgsString.substr(7);
      std::vector<char> decodedStr;
      if (auto err = llvm::decodeBase64(targetArgsStr, decodedStr)) {
        llvm::errs() << "DecodeBase64 error for '" << targetArgsStr
                     << "' string.";
        abort();
      }
      std::string decoded(decodedStr.data(), decodedStr.size());
      targetArgsString = decoded;
    } else {
      targetArgsString = "";
    }
  }
  llvm::StringRef(targetArgsString).split(args, ' ', -1, false);
  std::vector<std::string> targetArgv;
  for (const auto &arg : args) {
    std::string targetArgsStr = arg.str();
    if (targetArgsStr.starts_with("base64_")) {
      targetArgsStr.erase(0, 7); // erase "base64_"
      std::vector<char> decodedStr;
      if (auto err = llvm::decodeBase64(targetArgsStr, decodedStr)) {
        llvm::errs() << "DecodeBase64 error for '" << targetArgsStr
                     << "' string.";
        abort();
      }
      std::string decoded(decodedStr.data(), decodedStr.size());
      LLVM_DEBUG(llvm::dbgs() << "Decoded '" << decoded << "' from '"
                              << targetArgsStr << "\n");
      targetArgsStr = decoded;
    }
    targetArgv.emplace_back(targetArgsStr);
  }

  const auto nvqppConfigs = processRuntimeArgs(config, targetArgv);
  // Success! Dump the config (bash variable setters)
  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);
  if (ec) {
    errs() << "Failed to open output file '" << outputFilename << "'\n";
    return ec.value();
  }
  out.os() << nvqppConfigs;
  out.keep();
  return 0;
}
