/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/InitLLVM.h"
#include <filesystem>
#include <vector>
namespace cudaq {
struct TargetPlatformConfig {
  // Do we need to compile target backend config source file?
  bool genTargetBackend = false;
  // Additional link flags, e.g., "-lcudaq-rest-qpu"
  std::vector<std::string> linkFlags;
  // Enable/disable library mode.
  bool libraryMode = true;
  // Extra args to be sent on as defines
  std::string platformExtraArgs;
  // Name of the nvqir backend (if any)
  // (to select the library to link)
  std::string nvqirSimulationBackend;
  // Name of the platform
  // (to select the platform library to link)
  // Optional: default platform if none specified
  std::string nvqirPlatform;
};
struct Target {
  const char *name;
  const char *description;
  using PlatformExtraArgsCtorFnTy =
      std::string (*)(const llvm::opt::InputArgList &cudaqOptions);
  PlatformExtraArgsCtorFnTy platformArgsCtor = nullptr;
  TargetPlatformConfig
  generateConfig(const llvm::opt::InputArgList &clOptions,
                 const std::filesystem::path &targetConfigSearchPath);
};

struct TargetRegistry {
  TargetRegistry() = delete;
  // Find a target by name, nullopt if not found
  static std::optional<Target> lookupTarget(const std::string &targetName);
  // Register a target
  static void
  registerTarget(const char *name, const char *desc,
                 Target::PlatformExtraArgsCtorFnTy argsCtorFn = nullptr);

private:
  // Note: thread-safety is not considered since we don't expect race conditions
  // here.
  static inline std::unordered_map<std::string, Target> registry;
};
// Register all targets
void registerAllTargets();
} // namespace cudaq