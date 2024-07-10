/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include <mutex>
#include <unordered_map>

using namespace mlir;

namespace cudaq {

/// @brief The JITExecutionCache is a utility class for
/// storing ExecutionEngine pointers keyed on the hash
/// for the string representation of the original MLIR ModuleOp.
class JITExecutionCache {
protected:
  std::unordered_map<std::size_t, ExecutionEngine *> cacheMap;
  std::mutex mutex;

public:
  JITExecutionCache() = default;
  ~JITExecutionCache();

  void cache(std::size_t hash, ExecutionEngine *);
  bool hasJITEngine(std::size_t hash);
  ExecutionEngine *getJITEngine(std::size_t hash);
};
} // namespace cudaq
