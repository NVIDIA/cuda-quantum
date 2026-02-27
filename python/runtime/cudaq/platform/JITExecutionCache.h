/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/JIT.h"
#include <list>
#include <mutex>
#include <unordered_map>

namespace cudaq {

/// @brief The JITExecutionCache is a utility class for
/// storing ExecutionEngine pointers keyed on the hash
/// for the string representation of the original MLIR ModuleOp.
class JITExecutionCache {
protected:
  // Implement a Least Recently Used cache based on the JIT hash.
  std::list<std::size_t> lruList;

  // A given JIT hash has an associated MapItemType, which contains pointers to
  // the execution engine and to the LRU iterator that is used to track which
  // engine is the least recently used.
  struct MapItemType {
    JitEngine execEngine;
    std::list<std::size_t>::iterator lruListIt;
  };
  std::unordered_map<std::size_t, MapItemType> cacheMap;

  std::mutex mutex;

public:
  JITExecutionCache() = default;
  ~JITExecutionCache();

  void cache(std::size_t hash, JitEngine);
  bool hasJITEngine(std::size_t hash);
  void deleteJITEngine(std::size_t hash);
  JitEngine getJITEngine(std::size_t hash);
  static JITExecutionCache &getJITCache();
};
} // namespace cudaq
