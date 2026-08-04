/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompiledModule.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::detail {

/// Thread-safe, per-kernel cache of immutable compiled modules.
///
/// Calls for the same key share one compilation attempt. One caller produces
/// the artifact while concurrent callers join that attempt; calls for different
/// keys may compile concurrently. Successful artifacts remain usable through
/// shared ownership even after their cache entry is evicted.
///
/// The cache retains up to four ready artifacts to bound the memory held by JIT
/// engines. Eviction is first-in, first-out: a ready hit deliberately does not
/// refresh insertion order, keeping the hot path read-only and publication
/// order deterministic under concurrency. Outstanding compilation attempts are
/// neither counted toward the ready-entry limit nor evicted.
///
/// Inherits `std::enable_shared_from_this` so the Python binding layer reuses
/// the native `shared_ptr` control block when a cache crosses the language
/// boundary. Without it, nanobind substitutes a Python-backed control block
/// whose deletion logic must acquire the GIL — unsafe once asynchronous worker
/// threads hold the last reference.
class CompiledModuleCache
    : public std::enable_shared_from_this<CompiledModuleCache> {
public:
  /// All compile-time inputs that determine whether an artifact can be reused.
  struct Key {
    /// Uniqued entry-point name of the kernel being compiled.
    std::string kernelName;
    /// Hash of the target configuration used for compilation.
    std::size_t targetHash = 0;
    /// Digest of the resolved program, including compile-time dependencies.
    std::array<std::uint8_t, 32> programDigest = {};

    bool operator==(const Key &) const = default;
  };

  /// Immutable compiled module shared by the cache and all callers using it.
  using SharedCompiledModule = std::shared_ptr<const CompiledModule>;

  /// Role this caller played while resolving a `getOrCompile()` request.
  enum class Role {
    /// This caller ran the compilation callback and published its result.
    Producer,
    /// This caller found and joined an existing compilation attempt.
    Follower,
    /// This caller found an artifact that had already been published.
    ReadyReader
  };

  /// Compiled module returned by `getOrCompile()` and this caller's role.
  struct Result {
    /// Non-null compiled module owned independently of cache eviction.
    SharedCompiledModule module;
    Role role;
  };

  CompiledModuleCache();
  ~CompiledModuleCache() = default;

  /// A cache has shared identity and must not be copied or relocated. Share it
  /// with std::shared_ptr<CompiledModuleCache> instead.
  CompiledModuleCache(const CompiledModuleCache &) = delete;
  CompiledModuleCache &operator=(const CompiledModuleCache &) = delete;
  CompiledModuleCache(CompiledModuleCache &&) = delete;
  CompiledModuleCache &operator=(CompiledModuleCache &&) = delete;

  /// Return the compiled artifact for \p key, invoking \p compile only when
  /// this caller claims a missing entry.
  ///
  /// The check for an existing entry and installation of a new outstanding
  /// attempt form one atomic cache operation. The cache mutex is not held while
  /// invoking \p compile or waiting for another caller's attempt. This class
  /// does not manage the Python GIL; Python launch paths must release it before
  /// calling this function. The callback is invoked synchronously and is never
  /// retained beyond this function call.
  ///
  /// On success, the produced artifact is published before this function
  /// returns. If \p compile throws, concurrent callers joining the same attempt
  /// observe that exception, the failed entry is removed, and a later call may
  /// retry compilation.
  ///
  /// The callback must only create the compiled artifact; callers execute the
  /// kernel after this function returns. It must return a non-null module —
  /// a null result is rejected with an exception (observed by all joined
  /// callers) rather than published. It must not call back into this cache
  /// for the same key: the nested call would join the caller's own attempt
  /// and deadlock waiting for it.
  Result getOrCompile(const Key &key,
                      llvm::function_ref<SharedCompiledModule()> compile);

private:
  /// A successfully published key/module pair. Ready entries are immutable
  /// after insertion and are the only entries subject to FIFO eviction.
  struct ReadyEntry {
    Key key;
    SharedCompiledModule module;
  };

  /// Completion state shared by every caller participating in one compilation
  /// attempt. The producer completes the promise once; followers wait on the
  /// shared future and observe either the same module or the same exception.
  struct CompilingEntry {
    /// `std::promise` permits `get_future()` only once, so create the copyable
    /// `shared_future` before publishing this entry to any follower.
    explicit CompilingEntry(Key compilationKey)
        : key(std::move(compilationKey)),
          completion(promise.get_future().share()) {}

    /// Retained independently of the producer's stack frame so callers can
    /// continue matching this attempt while compilation runs without the lock.
    Key key;
    /// Single-producer write end of the completion channel.
    std::promise<SharedCompiledModule> promise;
    /// Multi-follower read end of the completion channel.
    std::shared_future<SharedCompiledModule> completion;
  };

  /// Bound only completed artifacts; compiling entries must remain discoverable
  /// so a later equivalent caller cannot start duplicate work.
  std::size_t maxReadyEntries = 4;

  /// One mutex protects both collections, making the Missing -> Compiling and
  /// Compiling -> Ready transitions atomic.
  std::mutex cacheMutex;
  /// In insertion order so erasing the first element implements FIFO eviction.
  std::vector<ReadyEntry> readyEntries;
  /// `shared_ptr` keeps an attempt alive after the producer releases
  /// `cacheMutex` and until it publishes either a module or an exception.
  std::vector<std::shared_ptr<CompilingEntry>> compilingEntries;
};

} // namespace cudaq::detail
