/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "llvm/Support/Allocator.h"
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
  bool IsPlatformConfigArg = false;
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
    io.mapOptional("platform-arg", info.IsPlatformConfigArg);
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
  std::string PlatformLibrary;
  std::string PlatformQpu;
  std::vector<std::string> PreprocessorDefines;
  std::vector<std::string> LinkLibs;
  std::string SimulationBackend;
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
    io.mapOptional("platform-library", info.PlatformLibrary);
    io.mapOptional("platform-qpu", info.PlatformQpu);
    io.mapOptional("preprocessor-defines", info.PreprocessorDefines);
    io.mapOptional("link-libs", info.LinkLibs);
    io.mapOptional("simulation-backend", info.SimulationBackend);
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
    io.mapRequired("description", info.Description);
    io.mapOptional("target-arguments", info.TargetArguments);
    io.mapOptional("gpu-requirements", info.GpuRequired);
    io.mapOptional("config", info.BackendConfig);
  }
};
} // namespace yaml
} // namespace llvm

std::optional<std::string> processRuntimeArgs(TargetConfig config, int argc,
                                              char **argv) {}

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
  return 0;
}
