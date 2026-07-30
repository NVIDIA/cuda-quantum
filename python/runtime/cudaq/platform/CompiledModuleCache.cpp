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
#include <future>
#include <memory>
#include <optional>
#include <type_traits>

namespace cudaq::detail {

CompiledModuleCache::Result CompiledModuleCache::getOrCompile(
    const Key &key, llvm::function_ref<CompiledModule()> compile) {
  static_assert(std::is_nothrow_move_constructible_v<CompiledModule>);
  static_assert(std::is_nothrow_move_constructible_v<ReadyEntry>);
  static_assert(std::is_nothrow_move_assignable_v<ReadyEntry>);

  std::shared_ptr<CompilingEntry> compilingEntry;
  std::optional<std::promise<CompiledModule>> producerPromise;
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
      return {*readyIter->module, Role::ReadyReader};

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
      producerPromise.emplace();
      auto completion = producerPromise->get_future().share();
      compilingEntry =
          std::make_shared<CompilingEntry>(key, std::move(completion));
      compilingEntries.emplace_back(compilingEntry);
      role = Role::Producer;
    }
  } // `cacheMutex` is released here

  // Waiting while holding `cacheMutex` would deadlock.
  if (role == Role::Follower)
    return {compilingEntry->completion.get(), role};

  std::optional<CompiledModule> module;
  std::optional<CompiledModule> completionCopy;
  std::optional<ReadyEntry> ready;
  // Written under the lock, destroyed after it: tearing down an evicted JIT
  // artifact can be heavy and must not block callers on unrelated keys.
  std::unique_ptr<const CompiledModule> evicted;
  bool readyPublished = false;

  try {
    // Only the producer invokes the callback. It runs without `cacheMutex`, so
    // unrelated keys can compile concurrently.
    module.emplace(compile());

    // Stage every potentially throwing value copy before publication. The
    // ready entry is also allocated before taking `cacheMutex` so a large
    // module copy cannot extend the critical section.
    completionCopy.emplace(*module);
    // Use the immutable key snapshot claimed before compilation. The callback
    // may mutate the caller's original key, but it cannot change this attempt's
    // identity.
    ready.emplace(ReadyEntry{compilingEntry->key,
                             std::make_unique<const CompiledModule>(*module)});

    // Replace "Compiling" with "Ready" under one lock. No caller can observe a
    // "Missing" gap and incorrectly begin another compilation for this key.
    {
      std::lock_guard<std::mutex> lock(cacheMutex);

      // Insert before evicting so an allocation failure preserves every
      // existing "Ready" entry; the catch path then makes this attempt
      // retryable.
      readyEntries.emplace_back(std::move(*ready));
      readyPublished = true;

      if (readyEntries.size() > maxReadyEntries) {
        evicted = std::move(readyEntries.front().module);
        readyEntries.erase(readyEntries.begin());
      }

      // Exactly one in-flight entry must be replaced by this "Ready" entry.
      if (std::erase(compilingEntries, compilingEntry) != 1)
        std::terminate();
    }
  } catch (...) {
    // Once the "Ready" entry is visible, rolling back only the in-flight state
    // would corrupt the atomic "Compiling" -> "Ready" transition.
    if (readyPublished)
      std::terminate();

    auto error = std::current_exception();
    {
      std::lock_guard<std::mutex> lock(cacheMutex);
      // Removing the failed attempt makes a later call for this key eligible
      // to become a new producer.
      if (std::erase(compilingEntries, compilingEntry) != 1)
        std::terminate();
    }

    // Fulfill outside `cacheMutex` because making the future ready can wake all
    // followers immediately.
    try {
      producerPromise->set_exception(error);
    } catch (...) {
      std::terminate();
    }
    std::rethrow_exception(error);
  }

  // "Ready" is already visible before followers wake. With one valid
  // producer-owned promise and a nothrow module move, failure here is an
  // invariant violation; no rollback is valid after publication.
  try {
    producerPromise->set_value(std::move(*completionCopy));
  } catch (...) {
    std::terminate();
  }

  // Lookup eviction occurred under the lock, but potentially expensive JIT
  // teardown does not.
  evicted.reset();

  return {std::move(*module), role};
}

} // namespace cudaq::detail
