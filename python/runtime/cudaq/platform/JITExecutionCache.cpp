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

JITExecutionCache::JITExecutionCache() {
  lruListHead.hash = 0;
  lruListHead.next = &lruListHead;
  lruListHead.prev = &lruListHead;
}

JITExecutionCache::~JITExecutionCache() {
  std::scoped_lock<std::mutex> lock(mutex);
  for (auto &[k, v] : cacheMap) {
    delete v.execEngine;
    delete v.lruNode;
  }
  cacheMap.clear();
  lruListHead.next = &lruListHead;
  lruListHead.prev = &lruListHead;
}

bool JITExecutionCache::hasJITEngine(std::size_t hashkey) {
  std::scoped_lock<std::mutex> lock(mutex);
  return cacheMap.count(hashkey);
}

void JITExecutionCache::cache(std::size_t hash, ExecutionEngine *jit) {
  std::scoped_lock<std::mutex> lock(mutex);

  if (cacheMap.contains(hash))
    return;

  // Make a new LRU node and insert it at the end of the doubly linked list.
  LRUNodeType *newNode = new LRUNodeType();
  newNode->hash = hash;
  newNode->next = &lruListHead;
  newNode->prev = lruListHead.prev;
  lruListHead.prev->next = newNode;
  lruListHead.prev = newNode;

  // If adding a new item would exceed our cache limit, then remove the least
  // recently used item (at the head of the list).
  if (cacheMap.size() >= NUM_JIT_CACHE_ITEMS_TO_RETAIN) {
    auto hashToRemove = lruListHead.next->hash;
    auto it = cacheMap.find(hashToRemove);
    lruListHead.next = lruListHead.next->next;
    delete it->second.execEngine;
    delete it->second.lruNode;
    cacheMap.erase(it);
  }

  cacheMap.insert({hash, {jit, newNode}});
}
ExecutionEngine *JITExecutionCache::getJITEngine(std::size_t hash) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto &item = cacheMap.at(hash);

  // Move item.lruNode to the end of the list to indicate that it is being used
  // right now.

  // First remove it from the list
  item.lruNode->prev->next = item.lruNode->next;
  item.lruNode->next->prev = item.lruNode->prev;

  // Then insert the item at the end of the list
  item.lruNode->next = &lruListHead;
  item.lruNode->prev = lruListHead.prev;
  lruListHead.prev->next = item.lruNode;
  lruListHead.prev = item.lruNode;

  return item.execEngine;
}
} // namespace cudaq
