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
#include "llvm/Support/InitLLVM.h"
#include <filesystem>
#include <vector>

using ArgT = const char *;
using ArgvT = llvm::ArrayRef<ArgT>;
using ArgvStorage = llvm::SmallVector<ArgT, 256>;
using ArgvStorageBase = llvm::SmallVectorImpl<ArgT>;
using ExecCompileFuncT = int (*)(ArgvStorageBase &);

namespace cudaq {
/// Skeleton for CUDA Quantum-specific arguments (on top of regular Clang args)
struct CudaqArgs {
  using OptionList = std::vector<llvm::StringRef>;
  using MaybeOptionList = std::optional<OptionList>;

  ArgvStorage args;
  // Systematically detect our arguments
  // e.g., -cudaq-<option_flag> or -cudaq-<option_var_name>=<option_value>
  static constexpr llvm::StringRef cudaqOptionPrefix = "-cudaq-";
  static std::pair<CudaqArgs, ArgvStorage>
  filterArgs(const ArgvStorageBase &args) {
    CudaqArgs cudaqArgs;
    ArgvStorage rest;
    for (const auto &arg : args) {
      if (std::string_view(arg).starts_with(CudaqArgs::cudaqOptionPrefix)) {
        cudaqArgs.pushBack(arg);
      } else {
        rest.push_back(arg);
      }
    }

    return {cudaqArgs, rest};
  }
  // Detects the presence of an option in one of formats:
  // (1) -cudaq-"name"
  // (2) -cudaq-"name"="value"
  bool hasOption(llvm::StringRef opt) const {
    // TODO
    return getOptionInternal(opt).has_value();
  }

  // from option of form -cudaq-"name"="value" returns the "value"
  std::optional<llvm::StringRef> getOption(llvm::StringRef argName) const {
    if (auto opt = getOptionInternal(argName)) {
      if (auto [lhs, rhs] = opt->split('='); !rhs.empty())
        return rhs;
    }

    return std::nullopt;
  }

  // from option of form -cudaq-"name"="value1;value2;value3" returns list of
  // values
  MaybeOptionList getOptionsList(llvm::StringRef argName) const {
    // TODO
    return MaybeOptionList();
  }

  void pushBack(ArgT arg) { args.push_back(arg); }

private:
  std::optional<llvm::StringRef>
  getOptionInternal(llvm::StringRef argName) const {
    auto isOptWithName = [argName](llvm::StringRef arg) {
      return arg.drop_front(cudaqOptionPrefix.size()).startswith(argName);
    };

    if (auto it = llvm::find_if(args, isOptWithName); it != args.end())
      return llvm::StringRef(*it).drop_front(cudaqOptionPrefix.size());

    return std::nullopt;
  }
};

// Generic interface for target specific args parsing,
// e.g., implementation will consume its args and construct the
// PLATFORM_EXTRA_ARGS for compiling the backendConfig.cpp.
struct TargetPlatformArgs {
  struct Data {
    bool genTargetBackend = false;
    // Additional link flags, e.g., "-lcudaq-rest-qpu"
    std::vector<std::string> linkFlags;
    bool libraryMode = true;
    // Extra args to be sent on as defines
    std::string platformExtraArgs;
    std::string nvqirSimulationBackend;
    std::string nvqirPlatform;
  };
  virtual Data parsePlatformArgs(ArgvStorageBase &args) = 0;
};

std::shared_ptr<TargetPlatformArgs>
getTargetPlatformArgs(const std::string &targetName,
                      const std::filesystem::path &platformPath);
} // namespace cudaq