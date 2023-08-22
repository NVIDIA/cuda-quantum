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

using arg_t = const char *;
using argv_t = llvm::ArrayRef<arg_t>;
using argv_storage = llvm::SmallVector<arg_t, 256>;
using argv_storage_base = llvm::SmallVectorImpl<arg_t>;
using exec_compile_t = int (*)(argv_storage_base &);

namespace cudaq {

struct cudaq_args {
  using option_list = std::vector<llvm::StringRef>;
  using maybe_option_list = std::optional<option_list>;

  argv_storage args;

  // Detects the presence of an option in one of formats:
  // (1) -cudaq-"name"
  // (2) -cudaq-"name"="value"
  bool has_option(llvm::StringRef opt) const {
    // TODO
    return false;
  }

  // from option of form -cudaq-"name"="value" returns the "value"
  std::optional<llvm::StringRef> get_option(llvm::StringRef opt) const {
    // TODO
    return std::optional<llvm::StringRef>();
  }

  // from option of form -cudaq-"name"="value1;value2;value3" returns list of
  // values
  maybe_option_list get_options_list(llvm::StringRef opt) const {
    // TODO
    return maybe_option_list();
  }

  void push_back(arg_t arg) {
    // TODO
  }
};
} // namespace cudaq