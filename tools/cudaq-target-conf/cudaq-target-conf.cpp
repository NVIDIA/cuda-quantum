/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/TargetConfig.h"
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
#include <sstream>
#include <unistd.h>

using namespace llvm;
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

static cl::opt<bool> skipGpuCheck(
    "skip-gpu-check",
    cl::desc("Skip NVIDIA check on target configuration that requires GPUs."),
    cl::init(false));

static constexpr const char BOLD[] = "\033[1m";
static constexpr const char RED[] = "\033[91m";
static constexpr const char CLEAR[] = "\033[0m";

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
  if (std::error_code ec = fileOrErr.getError()) {
    if (ec) {
      llvm::errs() << "Could not open input YML file: " << inputConfigFile
                   << "\n";
      llvm::errs() << "Error code: " << ec.message() << "\n";
      std::exit(ec.value());
    }
  }
  cudaq::config::TargetConfig config;
  llvm::yaml::Input Input(*(fileOrErr.get()));
  Input >> config;

  // Verify GPU requirement
  if (!skipGpuCheck && config.GpuRequired && countGPUs() <= 0) {
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

  const auto nvqppConfigs =
      cudaq::config::processRuntimeArgs(config, targetArgv);
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
