/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#include "cuda_memory.h"
#include <cmath>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <utility>

namespace cudaq::detail {

/// \file propagator_cache.h
/// \brief LRU cache for PWC propagator matrices.

/// \brief Quantize timestep to 1 ps precision for stable hashing.
/// \param dt_ns Timestep in nanoseconds.
/// \return Quantized integer (picoseconds).
inline int64_t quantize_timestep(double dt_ns) {
  return static_cast<int64_t>(std::round(dt_ns * 1000.0)); // 0.001 ns = 1 ps
}

/// \brief Cache entry for PWC propagators.
///
/// Stores both U = exp(-i·0.5·dt·H) and U² = U × U for Magnus CF4.
struct PropagatorCacheEntry {
  CudaComplexMemory U;         ///< exp(-i·0.5·dt·H).
  CudaComplexMemory U_squared; ///< U².
  size_t dim = 0;              ///< Matrix dimension.
  size_t last_use_time = 0;    ///< Timestamp for LRU.

  PropagatorCacheEntry() = default;
  PropagatorCacheEntry(PropagatorCacheEntry &&) = default;
  PropagatorCacheEntry &operator=(PropagatorCacheEntry &&) = default;
};

/// \brief LRU cache for PWC propagators with O(1) lookup and eviction.
///
/// Eliminates redundant matrix exponentials for piecewise-constant
/// Hamiltonians by caching (H_signature, dt) → propagator mappings.
class PropagatorLRUCache {
public:
  /// \brief Construct cache with given capacity.
  /// \param max_entries Maximum number of cached propagators.
  explicit PropagatorLRUCache(size_t max_entries = 32)
      : max_entries_(max_entries) {}

  /// \brief Build cache key from Hamiltonian signature and timestep.
  /// \param H_signature Hamiltonian hash.
  /// \param dt_ns Timestep in nanoseconds.
  /// \return Combined cache key.
  [[nodiscard]] size_t make_key(size_t H_signature, double dt_ns) const {
    // Combine H_signature and quantized dt using FNV-1a
    size_t h = H_signature;
    constexpr size_t fnv_prime = 1099511628211ULL;
    int64_t dt_quantized = quantize_timestep(dt_ns);
    h ^= static_cast<size_t>(dt_quantized);
    h *= fnv_prime;
    return h;
  }

  /// \brief Lookup entry by key.
  /// \param key Cache key from make_key().
  /// \return Pointer to entry, or nullptr if not found.
  PropagatorCacheEntry *get(size_t key) {
    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
      ++cache_misses_;
      return nullptr;
    }

    // O(1) LRU update: move to front
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.list_iter);
    ++cache_hits_;

    return &it->second.entry;
  }

  /// \brief Insert new entry (evicts LRU if at capacity).
  /// \param key Cache key.
  /// \param dim Matrix dimension.
  /// \return Reference to the new cache entry.
  PropagatorCacheEntry &insert(size_t key, size_t dim) {
    if (cache_map_.size() >= max_entries_) {
      evict_lru();
    }

    // Insert at front of LRU list
    lru_list_.push_front(key);

    auto &cache_entry = cache_map_[key];
    cache_entry.entry.dim = dim;
    cache_entry.entry.U.reallocate(dim * dim);
    cache_entry.entry.U_squared.reallocate(dim * dim);
    cache_entry.list_iter = lru_list_.begin();

    return cache_entry.entry;
  }

  /// \brief Check if key exists in cache.
  [[nodiscard]] bool contains(size_t key) const {
    return cache_map_.find(key) != cache_map_.end();
  }

  /// \brief Clear all cached entries and reset statistics.
  void clear() {
    cache_map_.clear();
    lru_list_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
  }

  /// \brief Get current number of cached entries.
  [[nodiscard]] size_t size() const { return cache_map_.size(); }

  /// \brief Get maximum cache capacity.
  [[nodiscard]] size_t capacity() const { return max_entries_; }

  /// \brief Get cache hit count.
  [[nodiscard]] size_t hits() const { return cache_hits_; }

  /// \brief Get cache miss count.
  [[nodiscard]] size_t misses() const { return cache_misses_; }

  /// \brief Get cache hit rate (0.0 to 1.0).
  [[nodiscard]] double hit_rate() const {
    size_t total = cache_hits_ + cache_misses_;
    return (total > 0) ? static_cast<double>(cache_hits_) / total : 0.0;
  }

private:
  /// \brief Evict least recently used entry.
  void evict_lru() {
    if (lru_list_.empty())
      return;

    // O(1) eviction: remove least recently used
    size_t lru_key = lru_list_.back();
    lru_list_.pop_back();
    cache_map_.erase(lru_key);
  }

  /// Internal map entry with LRU list iterator.
  struct MapEntry {
    PropagatorCacheEntry entry;
    std::list<size_t>::iterator list_iter;
  };

  std::unordered_map<size_t, MapEntry> cache_map_; ///< Key → entry map.
  std::list<size_t> lru_list_; ///< LRU order (front = MRU, back = LRU).
  size_t max_entries_;         ///< Maximum cache size.
  size_t cache_hits_ = 0;      ///< Hit counter.
  size_t cache_misses_ = 0;    ///< Miss counter.
};

} // namespace cudaq::detail
