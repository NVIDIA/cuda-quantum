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
#include <vector>

namespace cudaq::pulse {

/// \file hamiltonian_cache.h
/// \brief LRU cache for Hamiltonian matrices.

/// \brief Quantize a double to fixed bins for stable hashing.
/// \param value Input value.
/// \param quantum Bin size (default 1e-6 = 1 MHz).
/// \return Quantized integer.
inline int64_t quantize_param(double value, double quantum = 1e-6) {
  return static_cast<int64_t>(std::round(value / quantum));
}

/// \brief Cache entry holding host and device copies of a Hamiltonian.
struct HamiltonianCacheEntry {
  std::vector<cuDoubleComplex> host_matrix; ///< CPU copy.
  CudaComplexMemory device_matrix;          ///< GPU copy.
  size_t dim = 0;                           ///< Matrix dimension.
  size_t last_use_time = 0;                 ///< Timestamp for LRU.

  HamiltonianCacheEntry() = default;
  HamiltonianCacheEntry(HamiltonianCacheEntry &&) = default;
  HamiltonianCacheEntry &operator=(HamiltonianCacheEntry &&) = default;
};

/// \brief LRU cache for Hamiltonians with O(1) lookup and eviction.
///
/// Key is a quantized signature built from drive parameters. Uses std::list
/// for LRU tracking (move-to-front on access).
class HamiltonianLRUCache {
public:
  /// \brief Construct cache with given capacity.
  /// \param max_entries Maximum number of cached Hamiltonians.
  explicit HamiltonianLRUCache(size_t max_entries = 16)
      : max_entries_(max_entries) {}

  /// \brief Lookup entry by signature.
  /// \param signature Quantized Hamiltonian key.
  /// \return Pointer to entry, or nullptr if not found.
  HamiltonianCacheEntry *get(size_t signature) {
    auto it = cache_map_.find(signature);
    if (it == cache_map_.end()) {
      return nullptr;
    }

    // O(1) LRU update: move to front of list
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second.list_iter);

    return &it->second.entry;
  }

  /// \brief Insert new entry (evicts LRU if at capacity).
  /// \param signature Quantized Hamiltonian key.
  /// \param dim Matrix dimension.
  /// \return Reference to the new cache entry.
  HamiltonianCacheEntry &insert(size_t signature, size_t dim) {
    // Evict if at capacity
    if (cache_map_.size() >= max_entries_) {
      evict_lru();
    }

    // Insert at front of LRU list
    lru_list_.push_front(signature);

    auto &cache_entry = cache_map_[signature];
    cache_entry.entry.dim = dim;
    cache_entry.entry.host_matrix.resize(dim * dim);
    cache_entry.entry.device_matrix.reallocate(dim * dim);
    cache_entry.list_iter = lru_list_.begin();

    return cache_entry.entry;
  }

  /// \brief Check if signature exists in cache.
  [[nodiscard]] bool contains(size_t signature) const {
    return cache_map_.find(signature) != cache_map_.end();
  }

  /// \brief Clear all cached entries.
  void clear() {
    cache_map_.clear();
    lru_list_.clear();
  }

  /// \brief Get current number of cached entries.
  [[nodiscard]] size_t size() const { return cache_map_.size(); }

  /// \brief Get maximum cache capacity.
  [[nodiscard]] size_t capacity() const { return max_entries_; }

private:
  /// \brief Evict least recently used entry.
  void evict_lru() {
    if (lru_list_.empty())
      return;

    // O(1) eviction: remove least recently used (back of list)
    size_t lru_key = lru_list_.back();
    lru_list_.pop_back();
    cache_map_.erase(lru_key);
  }

  /// Internal map entry with LRU list iterator.
  struct MapEntry {
    HamiltonianCacheEntry entry;
    std::list<size_t>::iterator list_iter;
  };

  std::unordered_map<size_t, MapEntry> cache_map_; ///< Key → entry map.
  std::list<size_t> lru_list_; ///< LRU order (front = MRU, back = LRU).
  size_t max_entries_;         ///< Maximum cache size.
};

/// \brief Build quantized signature from drive parameters.
///
/// Quantizes amplitudes/frequencies to 1 MHz precision to avoid cache misses
/// from floating-point noise.
///
/// \param qubits Qubit indices involved.
/// \param amplitudes Drive amplitudes (GHz).
/// \param frequencies Drive frequencies (GHz).
/// \param duration_samples Segment duration.
/// \return Hash signature for cache lookup.
inline size_t build_hamiltonian_signature(
    const std::vector<int> &qubits, const std::vector<double> &amplitudes,
    const std::vector<double> &frequencies, int duration_samples) {

  // FNV-1a hash
  size_t h = 14695981039346656037ULL;
  constexpr size_t fnv_prime = 1099511628211ULL;

  auto combine = [&h, fnv_prime](int64_t v) {
    h ^= static_cast<size_t>(v);
    h *= fnv_prime;
  };

  combine(duration_samples);
  for (auto q : qubits) {
    combine(q);
  }
  for (auto a : amplitudes) {
    combine(quantize_param(a, 1e-6)); // 1 MHz precision
  }
  for (auto f : frequencies) {
    combine(quantize_param(f, 1e-6)); // 1 MHz precision
  }

  return h;
}

} // namespace cudaq::pulse
