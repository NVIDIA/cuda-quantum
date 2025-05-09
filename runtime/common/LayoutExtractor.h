/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include <mlir/IR/BuiltinOps.h>
#include <string>
#include <utility>
#include <vector>

namespace cudaq {

/// Extracts data layout information from MLIR modules
class LayoutExtractor {
public:
  std::pair<std::size_t, std::vector<std::size_t>>
  extractLayout(const std::string &, const std::string &);

private:
  mlir::MLIRContext *createContext();
};

/// Free function
std::pair<std::size_t, std::vector<std::size_t>>
extractDataLayout(const std::string &, const std::string &);

} // namespace cudaq
