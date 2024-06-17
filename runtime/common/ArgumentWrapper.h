/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <vector>

namespace cudaq {

struct ArgWrapper {
  mlir::ModuleOp mod;
  std::vector<std::string> callableNames;
  void *rawArgs = nullptr;
};

/// Holder of wrapped kernel `args`.
struct KernelArgsHolder {
  cudaq::ArgWrapper argsWrapper;
  // Info about the argsWrapper's rawArgs pointer.
  std::size_t argsSize;
  std::int32_t returnOffset;
};
} // namespace cudaq
