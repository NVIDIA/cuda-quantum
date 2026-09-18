/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Resolves a target (by name, against the precompiled in-tree target database,
// or by YAML file for external/plugin targets) plus CLI target arguments into a
// flat file of `nvq++`-compatible bash `KEY=value` assignments. This is the
// successor to the retired `cudaq-target-conf` tool.

#include "cudaq/Target/TargetConfig.h"
#include "cudaq/Target/TargetDatabase.h"
#include "cudaq/Target/TargetPluginLibrary.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>

using namespace llvm;

std::string decodeBase64IfPrefixed(llvm::StringRef input) {
  if (!input.starts_with("base64_"))
    return input.str();

  if (input.size() <= 7)
    return "";

  auto encodedStr = input.substr(7);
  std::vector<char> decodedStr;
  if (auto err = llvm::decodeBase64(encodedStr, decodedStr)) {
    llvm::errs() << "DecodeBase64 error for '" << encodedStr << "' string.\n";
    abort();
  }
  return std::string(decodedStr.data(), decodedStr.size());
}

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

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "CUDA-Q Target Build Configuration Resolver\n");

  // In-tree targets are known to the precompiled target database at zero I/O &
  // parse cost. External/plugin targets are not part of that database, and are
  // resolved by loading their own compiled target plugin library.
  const cudaq::config::TargetConfig *builtin =
      cudaq::config::lookupBuiltinTarget(
          std::filesystem::path(inputConfigFile.getValue()).stem().string());

  cudaq::config::TargetConfig config;
  if (builtin) {
    config = *builtin;
  } else {
    auto pluginResult =
        cudaq::config::loadTargetPluginLibrary(inputConfigFile.getValue());
    if (!pluginResult.ok) {
      llvm::errs() << pluginResult.error << "\n";
      return 1;
    }
    config = pluginResult.config;
  }

  if (!config.WarningMsg.empty())
    llvm::outs() << BOLD << RED << "Warning: " << CLEAR << config.WarningMsg
                 << "\n";

  std::string targetArgsString = decodeBase64IfPrefixed(targetArgs);
  llvm::SmallVector<llvm::StringRef> args;
  llvm::StringRef(targetArgsString).split(args, ' ', -1, false);
  std::map<std::string, std::string> argsMap;
  if (args.size() > 0) {
    for (std::size_t idx = 0; idx < args.size() - 1; idx += 2) {
      std::string argKey = decodeBase64IfPrefixed(args[idx]);
      std::string argVal = decodeBase64IfPrefixed(args[idx + 1]);
      argsMap.insert({argKey, argVal});
    }
  }

  const auto nvqppConfigs = cudaq::config::processRuntimeArgs(config, argsMap);
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
