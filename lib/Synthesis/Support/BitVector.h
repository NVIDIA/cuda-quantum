/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace cudaq::synth {

// ---------------------------------------------------------------------------
// SIMD policy concept
// ---------------------------------------------------------------------------
//
// A policy is a struct that provides:
//   - storage_type: the underlying register / array type
//   - static methods: zero, constant, load, extract, bit_xor, bit_and, bit_not
//
// This lets BasicBitBlock<P> stay entirely free of #ifdef noise. The single
// #ifdef lives in the BitBlock type alias at the bottom of this file.

// clang-format off
template <typename P>
concept simd_policy =
  requires { typename P::storage_type; } &&
  requires(typename P::storage_type a,
           const std::array<int32_t, 8> &lanes, int32_t v) {
    { P::zero() } noexcept -> std::same_as<typename P::storage_type>;
    { P::constant(v) } noexcept -> std::same_as<typename P::storage_type>;
    { P::load(lanes) } noexcept -> std::same_as<typename P::storage_type>;
    { P::extract(a) } noexcept -> std::same_as<std::array<int32_t, 8>>;
    { P::bit_xor(a, a) } noexcept -> std::same_as<typename P::storage_type>;
    { P::bit_and(a, a) } noexcept -> std::same_as<typename P::storage_type>;
    { P::bit_not(a) } noexcept -> std::same_as<typename P::storage_type>;
  };
// clang-format on

// ---------------------------------------------------------------------------
// ScalarPolicy
// ---------------------------------------------------------------------------
//
// Portable fallback. Uses an alignas(32) array<int32_t,8> so that the
// in-memory layout is identical to the AVX2 path and serialised bitvectors
// can be freely exchanged between builds.

struct ScalarPolicy {
  struct alignas(32) storage_type {
    std::array<int32_t, 8> lanes;
  };

  static storage_type zero() noexcept { return storage_type{.lanes = {}}; }

  static storage_type constant(int32_t v) noexcept {
    storage_type s;
    s.lanes.fill(v);
    return s;
  }

  static storage_type load(const std::array<int32_t, 8> &arr) noexcept {
    storage_type s;
    s.lanes = arr;
    return s;
  }

  static std::array<int32_t, 8> extract(const storage_type &s) noexcept {
    return s.lanes;
  }

  static storage_type bit_xor(storage_type a, storage_type b) noexcept {
    storage_type result;
    for (int i = 0; i < 8; ++i)
      result.lanes[i] = a.lanes[i] ^ b.lanes[i];
    return result;
  }

  static storage_type bit_and(storage_type a, storage_type b) noexcept {
    storage_type result;
    for (int i = 0; i < 8; ++i)
      result.lanes[i] = a.lanes[i] & b.lanes[i];
    return result;
  }

  static storage_type bit_not(storage_type a) noexcept {
    storage_type result;
    for (int i = 0; i < 8; ++i)
      result.lanes[i] = ~a.lanes[i];
    return result;
  }
};

static_assert(simd_policy<ScalarPolicy>);

// ---------------------------------------------------------------------------
// AvxPolicy
// ---------------------------------------------------------------------------
//
// Uses AVX2 256-bit integer intrinsics. The struct is only defined when the
// translation unit is compiled with -mavx2 (i.e. __AVX2__ is defined).

#if defined(__AVX2__)
struct AvxPolicy {
  using storage_type = __m256i;

  static storage_type zero() noexcept { return _mm256_setzero_si256(); }

  static storage_type constant(int32_t v) noexcept {
    return _mm256_set1_epi32(v);
  }

  // SAFETY: BitBlock is alignas(32), so the std::array reference here comes
  // from extract() which hands out a freshly constructed array (stack), or
  // from a user-supplied 32-byte-aligned array. We use loadu for safety with
  // arbitrary alignment on the load path; store uses storeu likewise.
  static storage_type load(const std::array<int32_t, 8> &arr) noexcept {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(arr.data()));
  }

  static std::array<int32_t, 8> extract(storage_type v) noexcept {
    alignas(32) std::array<int32_t, 8> arr;
    _mm256_store_si256(reinterpret_cast<__m256i *>(arr.data()), v);
    return arr;
  }

  static storage_type bit_xor(storage_type a, storage_type b) noexcept {
    return _mm256_xor_si256(a, b);
  }

  static storage_type bit_and(storage_type a, storage_type b) noexcept {
    return _mm256_and_si256(a, b);
  }

  static storage_type bit_not(storage_type a) noexcept {
    // AVX2 has no integer NOT; XOR with all-ones is the idiom.
    return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
  }
};

static_assert(simd_policy<AvxPolicy>);
#endif // __AVX2__

// ---------------------------------------------------------------------------
// NeonPolicy (stub)
// ---------------------------------------------------------------------------
//
// Uses two 128-bit NEON registers (int32x4_t) to represent one 256-bit block.
// Full implementation is deferred; this stub satisfies the concept so that
// the type alias below compiles on AArch64 toolchains.

#if defined(__ARM_NEON)
struct NeonPolicy {
  struct storage_type {
    int32x4_t lo;
    int32x4_t hi;
  };

  static storage_type zero() noexcept {
    return {vdupq_n_s32(0), vdupq_n_s32(0)};
  }

  static storage_type constant(int32_t v) noexcept {
    return {vdupq_n_s32(v), vdupq_n_s32(v)};
  }

  static storage_type load(const std::array<int32_t, 8> &arr) noexcept {
    return {vld1q_s32(arr.data()), vld1q_s32(arr.data() + 4)};
  }

  static std::array<int32_t, 8> extract(storage_type v) noexcept {
    std::array<int32_t, 8> arr;
    vst1q_s32(arr.data(), v.lo);
    vst1q_s32(arr.data() + 4, v.hi);
    return arr;
  }

  static storage_type bit_xor(storage_type a, storage_type b) noexcept {
    return {veorq_s32(a.lo, b.lo), veorq_s32(a.hi, b.hi)};
  }

  static storage_type bit_and(storage_type a, storage_type b) noexcept {
    return {vandq_s32(a.lo, b.lo), vandq_s32(a.hi, b.hi)};
  }

  static storage_type bit_not(storage_type a) noexcept {
    return {vmvnq_s32(a.lo), vmvnq_s32(a.hi)};
  }
};

static_assert(simd_policy<NeonPolicy>);
#endif // __ARM_NEON

// ---------------------------------------------------------------------------
// BasicBitBlock<Policy>
// ---------------------------------------------------------------------------
//
// A 256-bit block of bits parameterised over a SIMD policy. Holds a single
// policy storage_type and delegates every operation to the policy's static
// methods. Trivially copyable for all concrete policies.

template <simd_policy Policy>
class BasicBitBlock {
public:
  // -- Factories --

  static BasicBitBlock zero() noexcept { return BasicBitBlock{Policy::zero()}; }

  static BasicBitBlock constant(int32_t v) noexcept {
    return BasicBitBlock{Policy::constant(v)};
  }

  static BasicBitBlock
  from_lanes(const std::array<int32_t, 8> &lanes) noexcept {
    return BasicBitBlock{Policy::load(lanes)};
  }

  // -- Observers --

  std::array<int32_t, 8> extract() const noexcept {
    return Policy::extract(inner_);
  }

  // -- Operators --

  BasicBitBlock &operator^=(BasicBitBlock rhs) noexcept {
    inner_ = Policy::bit_xor(inner_, rhs.inner_);
    return *this;
  }

  BasicBitBlock &operator&=(BasicBitBlock rhs) noexcept {
    inner_ = Policy::bit_and(inner_, rhs.inner_);
    return *this;
  }

  BasicBitBlock operator~() const noexcept {
    return BasicBitBlock{Policy::bit_not(inner_)};
  }

  friend BasicBitBlock operator^(BasicBitBlock a, BasicBitBlock b) noexcept {
    return a ^= b;
  }

  friend BasicBitBlock operator&(BasicBitBlock a, BasicBitBlock b) noexcept {
    return a &= b;
  }

private:
  explicit BasicBitBlock(typename Policy::storage_type inner) noexcept
      : inner_(inner) {}

  typename Policy::storage_type inner_;
};

// ---------------------------------------------------------------------------
// BitBlock type alias -- the single #ifdef in this file
// ---------------------------------------------------------------------------

#if defined(__AVX2__)
using BitBlock = BasicBitBlock<AvxPolicy>;
#elif defined(__ARM_NEON)
using BitBlock = BasicBitBlock<NeonPolicy>;
#else
using BitBlock = BasicBitBlock<ScalarPolicy>;
#endif

// ---------------------------------------------------------------------------
// BitVector
// ---------------------------------------------------------------------------

/// Dynamically-sized bitvector backed by std::vector<BitBlock>. Capacity is
/// always a multiple of kBlockSize (256) bits. The stored size rounds up.
class BitVector {
public:
  static constexpr size_t kLanes = 8;
  static constexpr size_t kLaneSize = 32;   // bits per int32_t lane
  static constexpr size_t kBlockSize = 256; // bits per BitBlock

  // -- Construction --

  /// Allocate a bitvector with capacity for at least nb_bits bits (all zero).
  explicit BitVector(size_t nb_bits) : blocks_(init_blocks(nb_bits)) {}

  /// Allocate a bitvector with exactly nb_blocks blocks (all zero).
  static BitVector from_blocks(size_t nb_blocks) {
    BitVector bv(0);
    bv.blocks_.resize(nb_blocks, BitBlock::zero());
    return bv;
  }

  /// Build from a span of int64_t values. Every 4 values fill one 256-bit
  /// block (lane order: lo32 of values[0], hi32 of values[0], lo32 of
  /// values[1], hi32 of values[1], ...). Pads to block boundary with zeros.
  static BitVector from_integers(std::span<const int64_t> values) {
    // Each int64_t contributes 2 int32_t lanes; 4 int64_t values fill one
    // block.
    const size_t nb_lanes = values.size() * 2;
    const size_t nb_blocks = (nb_lanes + kLanes - 1) / kLanes;
    BitVector bv = from_blocks(nb_blocks);

    size_t lane_idx = 0;
    size_t block_idx = 0;
    std::array<int32_t, 8> arr{};

    for (int64_t v : values) {
      arr[lane_idx++] =
          static_cast<int32_t>(static_cast<uint64_t>(v) & 0xFFFF'FFFFu);
      arr[lane_idx++] = static_cast<int32_t>(static_cast<uint64_t>(v) >> 32);
      if (lane_idx == kLanes) {
        bv.blocks_[block_idx++] = BitBlock::from_lanes(arr);
        arr = {};
        lane_idx = 0;
      }
    }
    if (lane_idx > 0)
      bv.blocks_[block_idx] = BitBlock::from_lanes(arr);

    return bv;
  }

  /// Build directly from a span of raw int32_t lanes. Length must be a
  /// multiple of kLanes (8). Useful for serialisation / interop.
  static BitVector from_lanes(std::span<const int32_t> lanes) {
    assert(lanes.size() % kLanes == 0 &&
           "lanes.size() must be a multiple of 8");
    const size_t nb_blocks = lanes.size() / kLanes;
    BitVector bv = from_blocks(nb_blocks);
    for (size_t i = 0; i < nb_blocks; ++i) {
      std::array<int32_t, 8> arr;
      for (size_t j = 0; j < kLanes; ++j)
        arr[j] = lanes[i * kLanes + j];
      bv.blocks_[i] = BitBlock::from_lanes(arr);
    }
    return bv;
  }

  // -- Capacity --

  /// Bit capacity (always a multiple of kBlockSize).
  size_t size() const noexcept { return blocks_.size() * kBlockSize; }

  size_t block_count() const noexcept { return blocks_.size(); }

  // -- Single-bit access --

  /// Return the value of bit `bit`.
  bool get(size_t bit) const {
    assert(bit < size() && "bit is out of range");
    const size_t block_idx = bit / kBlockSize;
    const size_t lane_idx = (bit % kBlockSize) / kLaneSize;
    const size_t bit_in_lane = bit % kLaneSize;
    return (blocks_[block_idx].extract()[lane_idx] >> bit_in_lane) & 1;
  }

  /// Flip bit `bit` (XOR with 1).
  void xor_bit(size_t bit) {
    assert(bit < size() && "bit is out of range");
    const size_t block_idx = bit / kBlockSize;
    const size_t lane_idx = (bit % kBlockSize) / kLaneSize;
    const size_t bit_in_lane = bit % kLaneSize;
    std::array<int32_t, 8> arr{};
    arr[lane_idx] = static_cast<int32_t>(1u << bit_in_lane);
    blocks_[block_idx] ^= BitBlock::from_lanes(arr);
  }

  // -- Bit scanning --

  /// Return the index of the first set bit, or std::nullopt if all zero.
  std::optional<size_t> first_one() const {
    for (size_t i = 0; i < blocks_.size(); ++i) {
      const auto arr = blocks_[i].extract();
      for (size_t j = 0; j < kLanes; ++j) {
        const auto lane = static_cast<uint32_t>(arr[j]);
        if (lane != 0)
          return i * kBlockSize + j * kLaneSize + std::countr_zero(lane);
      }
    }
    return std::nullopt;
  }

  /// Return indices of all set bits with index < nb_bits.
  std::vector<size_t> all_ones(size_t nb_bits) const {
    std::vector<size_t> result;
    for (size_t i = 0; i < blocks_.size(); ++i) {
      const auto arr = blocks_[i].extract();
      for (size_t j = 0; j < kLanes; ++j) {
        auto lane = static_cast<uint32_t>(arr[j]);
        const size_t base = i * kBlockSize + j * kLaneSize;
        while (lane != 0) {
          const size_t bit_pos = base + std::countr_zero(lane);
          if (bit_pos >= nb_bits)
            return result;
          result.push_back(bit_pos);
          lane &= lane - 1; // Kernighan: clear lowest set bit
        }
      }
    }
    return result;
  }

  // -- Bulk operations --

  BitVector &operator^=(const BitVector &rhs) {
    assert(blocks_.size() == rhs.blocks_.size());
    for (size_t i = 0; i < blocks_.size(); ++i)
      blocks_[i] ^= rhs.blocks_[i];
    return *this;
  }

  BitVector &operator&=(const BitVector &rhs) {
    assert(blocks_.size() == rhs.blocks_.size());
    for (size_t i = 0; i < blocks_.size(); ++i)
      blocks_[i] &= rhs.blocks_[i];
    return *this;
  }

  friend BitVector operator^(BitVector a, const BitVector &b) { return a ^= b; }
  friend BitVector operator&(BitVector a, const BitVector &b) { return a &= b; }

  /// Return a copy with every bit flipped.
  BitVector operator~() const {
    BitVector result = *this;
    result.negate();
    return result;
  }

  /// Flip every bit in-place (XOR with all-ones).
  void negate() {
    const BitBlock ones = BitBlock::constant(-1);
    for (auto &block : blocks_)
      block ^= ones;
  }

  // -- Extension --

  /// XOR the bits in `bits` starting at bit position `start_bit`, growing
  /// the vector if necessary. This mirrors the Rust extend_vec semantics.
  void extend(std::span<const bool> bits, size_t start_bit) {
    size_t block_idx = start_bit / kBlockSize;
    size_t lane_idx = (start_bit % kBlockSize) / kLaneSize;
    size_t bit_in_lane = start_bit % kLaneSize;
    std::array<int32_t, 8> arr{};

    for (bool val : bits) {
      if (bit_in_lane == kLaneSize) {
        bit_in_lane = 0;
        ++lane_idx;
        if (lane_idx == kLanes) {
          lane_idx = 0;
          if (block_idx < blocks_.size())
            blocks_[block_idx] ^= BitBlock::from_lanes(arr);
          ++block_idx;
          blocks_.push_back(BitBlock::zero());
          arr = {};
        }
      }
      if (val)
        arr[lane_idx] ^= static_cast<int32_t>(1u << bit_in_lane);
      ++bit_in_lane;
    }

    // Flush the partial array for the final (possibly partial) block.
    while (block_idx >= blocks_.size())
      blocks_.push_back(BitBlock::zero());
    blocks_[block_idx] ^= BitBlock::from_lanes(arr);
  }

  // -- Conversions --

  /// Decompose every block into individual bit values (capacity bits total).
  std::vector<bool> to_bool_vec() const {
    std::vector<bool> result;
    result.reserve(size());
    for (size_t i = 0; i < blocks_.size(); ++i) {
      const auto arr = blocks_[i].extract();
      for (size_t j = 0; j < kLanes; ++j) {
        const auto lane = static_cast<uint32_t>(arr[j]);
        for (size_t k = 0; k < kLaneSize; ++k)
          result.push_back((lane >> k) & 1u);
      }
    }
    return result;
  }

  /// Pack pairs of adjacent int32_t lanes into int64_t values.
  /// Each block yields 4 int64_t values; total size = block_count * 4.
  std::vector<int64_t> to_integer_vec() const {
    std::vector<int64_t> result;
    result.reserve(blocks_.size() * 4);
    for (size_t i = 0; i < blocks_.size(); ++i) {
      const auto arr = blocks_[i].extract();
      for (size_t k = 0; k < 4; ++k) {
        const uint64_t lo = static_cast<uint32_t>(arr[k * 2]);
        const uint64_t hi = static_cast<uint32_t>(arr[k * 2 + 1]);
        result.push_back(static_cast<int64_t>(lo | (hi << 32)));
      }
    }
    return result;
  }

  // -- Popcount --

  /// Return the total number of set bits across all blocks.
  int32_t popcount() const {
    int32_t total = 0;
    for (size_t i = 0; i < blocks_.size(); ++i) {
      const auto arr = blocks_[i].extract();
      for (size_t j = 0; j < kLanes; ++j)
        total += std::popcount(static_cast<uint32_t>(arr[j]));
    }
    return total;
  }

  // -- Raw block access --

  std::span<const BitBlock> blocks() const noexcept { return blocks_; }
  std::span<BitBlock> blocks() noexcept { return blocks_; }

private:
  std::vector<BitBlock> blocks_;

  static std::vector<BitBlock> init_blocks(size_t nb_bits) {
    const size_t count = nb_bits / kBlockSize + 1;
    return std::vector<BitBlock>(count, BitBlock::zero());
  }
};

} // namespace cudaq::synth
