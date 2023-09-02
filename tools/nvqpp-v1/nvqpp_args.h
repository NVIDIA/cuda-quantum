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
// FIXME: Design a target parsing tool

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