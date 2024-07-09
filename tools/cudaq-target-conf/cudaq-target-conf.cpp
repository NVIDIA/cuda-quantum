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
#include <fstream>

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
                   llvm::cl::value_desc("filename"));

static llvm::cl::opt<std::string>
    targetArgs("arg", llvm::cl::desc("Specify target CLI arguments"),
               llvm::cl::value_desc("string"));

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

enum TargetFeatureFlag : unsigned {
  flagsFP32 = 0x0001,
  flagsFP64 = 0x0002,
  flagsMgpu = 0x0004,
  flagsMqpu = 0x0008,
};

static std::unordered_map<std::string, TargetFeatureFlag> stringToFeatureFlag{
    {"fp32", flagsFP32},
    {"fp64", flagsFP64},
    {"mgpu", flagsMgpu},
    {"mqpu", flagsMqpu}};

namespace llvm {
namespace yaml {
template <>
struct ScalarBitSetTraits<TargetFeatureFlag> {
  static void bitset(IO &io, TargetFeatureFlag &value) {
    for (const auto &[k, v] : stringToFeatureFlag)
      io.bitSetCase(value, k.c_str(), v);
  }
};
} // namespace yaml
} // namespace llvm
LLVM_YAML_IS_SEQUENCE_VECTOR(TargetFeatureFlag)

enum ArgumentType { String, Int, UUID, FeatureFlag };
namespace llvm {
namespace yaml {
template <>
struct ScalarEnumerationTraits<ArgumentType> {
  static void enumeration(IO &io, ArgumentType &value) {
    io.enumCase(value, "string", String);
    io.enumCase(value, "integer", Int);
    io.enumCase(value, "uuid", UUID);
    io.enumCase(value, "option-flags", FeatureFlag);
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

struct SimulationBackendSetting {
  std::vector<std::string> values;
};

namespace llvm {
namespace yaml {
template <>
struct BlockScalarTraits<SimulationBackendSetting> {
  static void output(const SimulationBackendSetting &Value, void *Ctxt,
                     llvm::raw_ostream &OS) {
    std::size_t idx = 0;
    for (const auto &val : Value.values) {
      OS << val;
      if (idx != Value.values.size() - 1) {
        OS << ", ";
      }
      ++idx;
    }
  }

  static StringRef input(StringRef Scalar, void *Ctxt,
                         SimulationBackendSetting &Value) {
    llvm::SmallVector<llvm::StringRef> values;
    Scalar.split(values, ',', -1, false);
    for (const auto &val : values) {
      Value.values.emplace_back(val.trim().str());
    }
    return StringRef();
  }
};
} // namespace yaml
} // namespace llvm

struct ConditionalBuildConfig {
  std::string Condition;
  std::string CompileFlag;
  std::string LinkFlag;
};

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<ConditionalBuildConfig> {
  static void mapping(IO &io, ConditionalBuildConfig &info) {
    io.mapRequired("if", info.Condition);
    io.mapOptional("compiler-flag", info.CompileFlag);
    io.mapOptional("link-flag", info.LinkFlag);
  }
};
} // namespace yaml
} // namespace llvm
LLVM_YAML_IS_SEQUENCE_VECTOR(ConditionalBuildConfig)

struct BackendEndConfigEntry {
  std::optional<bool> GenTargetBackend;
  std::optional<bool> LibraryMode;
  std::string PlatformLoweringConfig;
  std::string CodegenEmission;
  std::string PostCodeGenPasses;
  std::string PlatformLibrary;
  std::string LibraryModeExecutionManager;
  std::string PlatformQpu;
  std::vector<std::string> PreprocessorDefines;
  std::vector<std::string> CompilerFlags;
  std::vector<std::string> LinkLibs;
  std::vector<std::string> LinkerFlags;
  SimulationBackendSetting SimulationBackend;
  std::vector<ConditionalBuildConfig> ConditionalBuildConfigs;
};

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<BackendEndConfigEntry> {
  static void mapping(IO &io, BackendEndConfigEntry &info) {
    io.mapOptional("gen-target-backend", info.GenTargetBackend);
    io.mapOptional("library-mode", info.LibraryMode);
    io.mapOptional("platform-lowering-config", info.PlatformLoweringConfig);
    io.mapOptional("codegen-emission", info.CodegenEmission);
    io.mapOptional("post-codegen-passes", info.PostCodeGenPasses);
    io.mapOptional("platform-library", info.PlatformLibrary);
    io.mapOptional("library-mode-execution-manager",
                   info.LibraryModeExecutionManager);
    io.mapOptional("platform-qpu", info.PlatformQpu);
    io.mapOptional("preprocessor-defines", info.PreprocessorDefines);
    io.mapOptional("compiler-flags", info.CompilerFlags);
    io.mapOptional("link-libs", info.LinkLibs);
    io.mapOptional("linker-flags", info.LinkerFlags);
    io.mapOptional("nvqir-simulation-backend", info.SimulationBackend);
    io.mapOptional("rules", info.ConditionalBuildConfigs);
  }
};
} // namespace yaml
} // namespace llvm

struct BackendFeatureMap {
  std::string Name;
  TargetFeatureFlag Flags;
  std::optional<bool> Default;
  BackendEndConfigEntry Config;
};

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<BackendFeatureMap> {
  static void mapping(IO &io, BackendFeatureMap &info) {
    io.mapRequired("name", info.Name);
    io.mapRequired("option-flags", info.Flags);
    io.mapOptional("default", info.Default);
    io.mapRequired("config", info.Config);
  }
};
} // namespace yaml
} // namespace llvm
LLVM_YAML_IS_SEQUENCE_VECTOR(BackendFeatureMap)

struct TargetConfig {
  std::string Name;
  std::string Description;
  std::string WarningMsg;
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
    io.mapOptional("warning", info.WarningMsg);
    io.mapOptional("target-arguments", info.TargetArguments);
    io.mapOptional("gpu-requirements", info.GpuRequired);
    io.mapOptional("config", info.BackendConfig);
    io.mapOptional("configuration-matrix", info.ConfigMap);
  }
};
} // namespace yaml
} // namespace llvm

std::string processSimBackendConfig(const std::string &targetName,
                                    const BackendEndConfigEntry &configValue) {
  std::stringstream output;
  if (configValue.GenTargetBackend.has_value()) {
    output << "GEN_TARGET_BACKEND="
           << (configValue.GenTargetBackend.value() ? "true" : "false") << "\n";
  }

  if (configValue.LibraryMode.has_value()) {
    output << "LIBRARY_MODE="
           << (configValue.LibraryMode.value() ? "true" : "false") << "\n";
  }

  if (!configValue.PlatformLoweringConfig.empty()) {
    output << "PLATFORM_LOWERING_CONFIG=\""
           << configValue.PlatformLoweringConfig << "\"\n";
  }
  if (!configValue.CodegenEmission.empty()) {
    output << "CODEGEN_EMISSION=" << configValue.CodegenEmission << "\n";
  }
  if (!configValue.PostCodeGenPasses.empty()) {
    output << "POST_CODEGEN_PASSES=\"" << configValue.PostCodeGenPasses
           << "\"\n";
  }
  if (!configValue.PlatformLibrary.empty()) {
    output << "PLATFORM_LIBRARY=" << configValue.PlatformLibrary << "\n";
  }
  if (!configValue.LibraryModeExecutionManager.empty()) {
    output << "LIBRARY_MODE_EXECUTION_MANAGER="
           << configValue.LibraryModeExecutionManager << "\n";
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

  if (!configValue.SimulationBackend.values.empty()) {
    output << "if [ -f \"${install_dir}/lib/libnvqir-"
           << configValue.SimulationBackend.values.front() << ".so\" ]; then\n";
    output << "  NVQIR_SIMULATION_BACKEND=\""
           << configValue.SimulationBackend.values.front() << "\"\n";

    for (std::size_t i = 1; i < configValue.SimulationBackend.values.size();
         ++i) {
      output << "elif [ -f \"${install_dir}/lib/libnvqir-"
             << configValue.SimulationBackend.values[i] << ".so\" ]; then\n";
      output << "  NVQIR_SIMULATION_BACKEND=\""
             << configValue.SimulationBackend.values[i] << "\"\n";
    }
    output << "else\n";
    output << "  error_exit=\"Unable to find NVQIR simulator lib for target "
           << targetName << ". Please check your installation.\"\n";
    output << "fi\n";
  }

  for (const auto &rule : configValue.ConditionalBuildConfigs) {
    output << "if [[ " << rule.Condition << " ]]; then\n";
    if (!rule.CompileFlag.empty()) {
      output << "  COMPILER_FLAGS=\"${COMPILER_FLAGS} " << rule.CompileFlag
             << "\"\n";
    }
    if (!rule.LinkFlag.empty()) {
      output << "  LINKER_FLAGS=\"${LINKER_FLAGS} " << rule.LinkFlag << "\"\n";
    }
    output << "fi\n";
  }

  return output.str();
}

std::string processRuntimeArgs(const TargetConfig &config,
                               const std::vector<std::string> &targetArgv) {
  std::stringstream output;
  if (config.BackendConfig.has_value()) {
    output << processSimBackendConfig(config.Name,
                                      config.BackendConfig.value());
  }

  unsigned featureFlag = 0;
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
                   << "' are: "
                   << "\n";
      for (const auto &argConfig : config.TargetArguments) {
        llvm::errs() << "  "
                     << "--" + config.Name + "-" + argConfig.KeyName;
        if (!argConfig.HelpString.empty()) {
          llvm::errs() << " (" << argConfig.HelpString << ")";
        }

        llvm::errs() << "\n";
      }
      abort();
    } else {
      if (iter->Type != ArgumentType::FeatureFlag) {
        if (!iter->PlatformArgKey.empty()) {
          platformExtraArgs << ";" << iter->PlatformArgKey << ";"
                            << targetArgv[idx + 1];
        }

        idx += 2;

      } else {
        const auto featureFlags = targetArgv[idx + 1];
        llvm::SmallVector<llvm::StringRef> flagStrs;
        llvm::StringRef(featureFlags).split(flagStrs, ',', -1, false);
        for (const auto &flag : flagStrs) {
          const auto iter = stringToFeatureFlag.find(flag.str());
          if (iter == stringToFeatureFlag.end()) {
            llvm::errs() << "Unknown  feature flag '" << flag << "'\n";
            abort();
          }
          featureFlag += iter->second;
        }

        idx += 2;
      }
    }
  }

  if (!config.ConfigMap.empty()) {
    const auto defaultFeatureIter = std::find_if(
        config.ConfigMap.begin(), config.ConfigMap.end(),
        [&](const BackendFeatureMap &entry) {
          return entry.Default.has_value() && entry.Default.value();
        });

    const uint64_t defaultFlag =
        (defaultFeatureIter != config.ConfigMap.end())
            ? static_cast<uint64_t>(defaultFeatureIter->Flags)
            : 0;

    const auto iter = [&]() {
      // If the command line set the feature flag, find it in the config map.
      // Otherwise, find the default.
      return featureFlag > 0
                 ? std::find_if(
                       config.ConfigMap.begin(), config.ConfigMap.end(),
                       [&](const BackendFeatureMap &entry) {
                         return featureFlag == entry.Flags ||
                                (featureFlag | defaultFlag) == entry.Flags;
                       })
                 : std::find_if(config.ConfigMap.begin(),
                                config.ConfigMap.end(),
                                [&](const BackendFeatureMap &entry) {
                                  return entry.Default.has_value() &&
                                         entry.Default.value();
                                });
    }();
    if (iter == config.ConfigMap.end()) {
      llvm::errs() << "Unable to find a config entry for feature flag value "
                   << featureFlag << ".\n";
      llvm::errs() << "This indicates the requested combination of features "
                      "is not supported.\n";
      abort();
    }
    output << processSimBackendConfig(config.Name, iter->Config);
  }
  const auto platformExtraArgsStr = platformExtraArgs.str();
  if (!platformExtraArgsStr.empty()) {
    output << "PLATFORM_EXTRA_ARGS=\"${PLATFORM_EXTRA_ARGS}"
           << platformExtraArgsStr << "\"\n";
  }
  return output.str();
}

/// @brief A utility function to check availability of Nvidia GPUs and return
/// their count.
int countGPUs() {
  int retCode = std::system("nvidia-smi >/dev/null 2>&1");
  if (0 != retCode) {
    LLVM_DEBUG(llvm::dbgs() << "nvidia-smi: command not found\n");
    return -1;
  }

  char tmpFile[] = "/tmp/.cmd.capture.XXXXXX";
  int fileDescriptor = mkstemp(tmpFile);
  if (-1 == fileDescriptor) {
    LLVM_DEBUG(llvm::dbgs()
               << "Failed to create a temporary file to capture output\n");
    return -1;
  }

  std::string command = "nvidia-smi -L 2>/dev/null | wc -l >> ";
  command.append(tmpFile);
  retCode = std::system(command.c_str());
  if (0 != retCode) {
    LLVM_DEBUG(llvm::dbgs()
               << "Encountered error while invoking 'nvidia-smi'\n");
    return -1;
  }

  std::stringstream buffer;
  buffer << std::ifstream(tmpFile).rdbuf();
  close(fileDescriptor);
  unlink(tmpFile);
  return std::stoi(buffer.str());
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

  // Verify GPU requirement
  if (config.GpuRequired && countGPUs() <= 0) {
    llvm::errs() << "Target '" << config.Name
                 << "' requires NVIDIA GPUs but none can be detected.";
    abort();
  }

  if (!config.WarningMsg.empty())
    llvm::outs() << BOLD << RED << "Warning: " << CLEAR << config.WarningMsg
                 << "\n";

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
