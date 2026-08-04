/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompiledModuleCache.h"
#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace cudaq::detail {

CompiledModuleCache::Result CompiledModuleCache::getOrCompile(
    const Key &key, llvm::function_ref<SharedCompiledModule()> compile) {
  std::shared_ptr<CompilingEntry> compilingEntry;
  Role role;

  // Without this lock scope, two callers could both observe "Missing" and
  // become producers for the same compilation.
  {
    std::lock_guard<std::mutex> lock(cacheMutex);

    auto readyIter =
        std::find_if(readyEntries.begin(), readyEntries.end(),
                     [&](const ReadyEntry &entry) { return entry.key == key; });
    // A ready hit does not change insertion order (eviction is FIFO).
    if (readyIter != readyEntries.end())
      return {readyIter->module, Role::ReadyReader};

    auto compilingIter =
        std::find_if(compilingEntries.begin(), compilingEntries.end(),
                     [&](const std::shared_ptr<CompilingEntry> &entry) {
                       return entry->key == key;
                     });
    if (compilingIter != compilingEntries.end()) {
      // Join the existing attempt rather than duplicating its compilation.
      compilingEntry = *compilingIter;
      role = Role::Follower;
    } else {
      // Publish the "Compiling" state before releasing mutex. Every later
      // equivalent caller will therefore become a follower of this attempt.
      compilingEntry = std::make_shared<CompilingEntry>(key);
      compilingEntries.emplace_back(compilingEntry);
      role = Role::Producer;
    }
  } // `cacheMutex` is released here

  // Waiting while holding `cacheMutex` would deadlock.
  if (role == Role::Follower)
    return {compilingEntry->completion.get(), role};

  SharedCompiledModule module;
  // Written under the lock, destroyed after it: tearing down an evicted JIT
  // artifact can be heavy and must not block callers on unrelated keys.
  SharedCompiledModule evicted;
  try {
    // Only the producer invokes the callback. It runs without `cacheMutex`, so
    // unrelated keys can compile concurrently.
    module = compile();
    // Reject rather than publish a null artifact: every joined caller would
    // otherwise dereference it far from the failure. The catch path below
    // makes this attempt retryable.
    if (!module)
      throw std::logic_error(
          "CompiledModuleCache compile callback returned a null module");

    // Replace "Compiling" with "Ready" under one lock. No caller can observe a
    // "Missing" gap and incorrectly begin another compilation for this key.
    std::lock_guard<std::mutex> lock(cacheMutex);
    // Insert before evicting so an allocation failure preserves every existing
    // "Ready" entry; the catch path then makes this attempt retryable.
    // Publish the key snapshot that was claimed before compilation. The
    // callback is user-supplied and must not be able to change the identity of
    // this attempt by mutating the caller's original `key` object.
    readyEntries.emplace_back(compilingEntry->key, module);
    if (readyEntries.size() > maxReadyEntries) {
      evicted = std::move(readyEntries.front().module);
      if (cudaq::CompiledModule::debugMode()) {
        std::cout << "Evicting compiled module for " << evicted->getName()
                  << std::endl;
      }
      readyEntries.erase(readyEntries.begin());
    }
    std::erase(compilingEntries, compilingEntry);
  } catch (...) {
    auto error = std::current_exception();
    {
      std::lock_guard<std::mutex> lock(cacheMutex);
      // Removing the failed attempt makes a later call for this key eligible
      // to become a new producer.
      std::erase(compilingEntries, compilingEntry);
    }
    // Fulfill outside `cacheMutex` because making the future ready can wake all
    // followers immediately.
    compilingEntry->promise.set_exception(error);
    std::rethrow_exception(error);
  }

  // "Ready" is already visible before followers wake. New callers can therefore
  // reuse the module instead of joining a completed attempt.
  compilingEntry->promise.set_value(module);
  return {std::move(module), role};
}

CompiledModuleCache::CompiledModuleCache() {
  if (const auto *cacheSize = std::getenv("CUDAQ_COMPILED_MODULE_CACHE_SIZE")) {
    try {
      maxReadyEntries = std::stoi(cacheSize);
    } catch (const std::exception &) {
    }
  }
  if (cudaq::CompiledModule::debugMode()) {
    std::cout << "CompiledModuleCache constructor with maxReadyEntries = "
              << maxReadyEntries << std::endl;
    std::cout << "Size of CompiledModule: " << sizeof(CompiledModule)
              << std::endl;
  }
}

} // namespace cudaq::detail
