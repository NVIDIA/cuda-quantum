/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/InitLLVM.h"

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

  // Detects the presence of an option in one of formats:
  // (1) -cudaq-"name"
  // (2) -cudaq-"name"="value"
  bool hasOption(llvm::StringRef opt) const {
    // TODO
    return false;
  }

  // from option of form -cudaq-"name"="value" returns the "value"
  std::optional<llvm::StringRef> getOption(llvm::StringRef opt) const {
    // TODO
    return std::optional<llvm::StringRef>();
  }

  // from option of form -cudaq-"name"="value1;value2;value3" returns list of
  // values
  MaybeOptionList getOptionsList(llvm::StringRef opt) const {
    // TODO
    return MaybeOptionList();
  }

  void pushBack(ArgT arg) {
    // TODO
  }
};
} // namespace cudaq