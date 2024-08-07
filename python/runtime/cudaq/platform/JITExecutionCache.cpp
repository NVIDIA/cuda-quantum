/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "JITExecutionCache.h"

using namespace mlir;

namespace cudaq {

JITExecutionCache::~JITExecutionCache() {
  std::scoped_lock<std::mutex> lock(mutex);
  for (auto &[k, v] : cacheMap)
    delete v;
  cacheMap.clear();
}
bool JITExecutionCache::hasJITEngine(std::size_t hashkey) {
  std::scoped_lock<std::mutex> lock(mutex);
  return cacheMap.count(hashkey);
}

void JITExecutionCache::cache(std::size_t hash, ExecutionEngine *jit) {
  std::scoped_lock<std::mutex> lock(mutex);
  cacheMap.insert({hash, jit});
}
ExecutionEngine *JITExecutionCache::getJITEngine(std::size_t hash) {
  std::scoped_lock<std::mutex> lock(mutex);
  return cacheMap.at(hash);
}
} // namespace cudaq
