/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "JITExecutionCache.h"

using namespace mlir;

namespace cudaq {

JITExecutionCache::~JITExecutionCache() {
  for (auto &[k, v] : cacheMap)
    delete v;
  cacheMap.clear();
}
bool JITExecutionCache::hasJITEngine(std::size_t hashkey) {
  return cacheMap.count(hashkey);
}

void JITExecutionCache::cache(std::size_t hash, ExecutionEngine *jit) {
  cacheMap.insert({hash, jit});
}
ExecutionEngine *JITExecutionCache::getJITEngine(std::size_t hash) {
  return cacheMap.at(hash);
}
} // namespace cudaq