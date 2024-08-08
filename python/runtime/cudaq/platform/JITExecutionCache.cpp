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

static constexpr int NUM_JIT_CACHE_ITEMS_TO_RETAIN = 100;

JITExecutionCache::~JITExecutionCache() {
  std::scoped_lock<std::mutex> lock(mutex);
  for (auto &[k, v] : cacheMap)
    delete v.execEngine;
  cacheMap.clear();
}
bool JITExecutionCache::hasJITEngine(std::size_t hashkey) {
  std::scoped_lock<std::mutex> lock(mutex);
  return cacheMap.count(hashkey);
}

void JITExecutionCache::cache(std::size_t hash, ExecutionEngine *jit) {
  std::scoped_lock<std::mutex> lock(mutex);

  lruList.push_back(hash);

  // If adding a new item would exceed our cache limit, then remove the least
  // recently used item (at the head of the list).
  if (cacheMap.size() >= NUM_JIT_CACHE_ITEMS_TO_RETAIN) {
    auto hashToRemove = lruList.begin();
    auto it = cacheMap.find(*hashToRemove);
    delete it->second.execEngine;
    lruList.erase(hashToRemove);
    cacheMap.erase(it);
  }

  cacheMap.insert({hash, {jit, std::prev(lruList.end())}});
}
ExecutionEngine *JITExecutionCache::getJITEngine(std::size_t hash) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto &item = cacheMap.at(hash);

  // Move item.lruListIt to the end of the list to indicate that it is being
  // used right now.
  lruList.splice(lruList.end(), lruList, item.lruListIt);

  return item.execEngine;
}
} // namespace cudaq
