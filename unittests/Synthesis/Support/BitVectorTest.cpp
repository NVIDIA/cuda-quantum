/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Support/BitVector.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace cudaq::synth {

// ===========================================================================
// Helpers
// ===========================================================================

// Build a known 8-lane array.
static std::array<int32_t, 8> make_lanes(std::initializer_list<int32_t> vals) {
  std::array<int32_t, 8> arr{};
  size_t i = 0;
  for (int32_t v : vals)
    arr[i++] = v;
  return arr;
}

// ===========================================================================
// ScalarPolicy -- tested explicitly so the scalar path is exercised even when
// the build is compiled with AVX2 or NEON enabled.
// ===========================================================================

using ScalarBlock = BasicBitBlock<ScalarPolicy>;

TEST(ScalarPolicy, Zero) {
  auto b = ScalarBlock::zero();
  for (int32_t lane : b.extract())
    EXPECT_EQ(lane, 0);
}

TEST(ScalarPolicy, Constant) {
  auto b = ScalarBlock::constant(0xDEAD'BEEF);
  for (int32_t lane : b.extract())
    EXPECT_EQ(lane, static_cast<int32_t>(0xDEAD'BEEF));
}

TEST(ScalarPolicy, FromLanesExtractRoundtrip) {
  const auto orig = make_lanes({1, 2, 3, 4, 5, 6, 7, 8});
  auto b = ScalarBlock::from_lanes(orig);
  EXPECT_EQ(b.extract(), orig);
}

TEST(ScalarPolicy, XorAssign) {
  auto a = ScalarBlock::constant(0x0F0F'0F0F);
  auto b = ScalarBlock::constant(0x00FF'00FF);
  a ^= b;
  for (int32_t lane : a.extract())
    EXPECT_EQ(lane, 0x0F0F'0F0F ^ 0x00FF'00FF);
}

TEST(ScalarPolicy, AndAssign) {
  auto a = ScalarBlock::constant(0x0F0F'0F0F);
  auto b = ScalarBlock::constant(0x00FF'00FF);
  a &= b;
  for (int32_t lane : a.extract())
    EXPECT_EQ(lane, 0x0F0F'0F0F & 0x00FF'00FF);
}

TEST(ScalarPolicy, Not) {
  auto a = ScalarBlock::constant(0x0000'FFFF);
  auto n = ~a;
  for (int32_t lane : n.extract())
    EXPECT_EQ(lane, ~0x0000'FFFF);
}

TEST(ScalarPolicy, FreeOperators) {
  auto a = ScalarBlock::constant(0xAAAA'AAAA);
  auto b = ScalarBlock::constant(0x5555'5555);
  auto xored = a ^ b;
  auto anded = a & b;
  for (int32_t lane : xored.extract())
    EXPECT_EQ(lane, static_cast<int32_t>(0xAAAA'AAAA ^ 0x5555'5555));
  for (int32_t lane : anded.extract())
    EXPECT_EQ(lane, static_cast<int32_t>(0xAAAA'AAAA & 0x5555'5555));
}

// ===========================================================================
// BitBlock (active ISA alias) -- same ops via the high-level alias
// ===========================================================================

TEST(BitBlock, ZeroExtractIsAllZero) {
  auto b = BitBlock::zero();
  for (int32_t lane : b.extract())
    EXPECT_EQ(lane, 0);
}

TEST(BitBlock, ConstantAllLanesSame) {
  auto b = BitBlock::constant(42);
  for (int32_t lane : b.extract())
    EXPECT_EQ(lane, 42);
}

TEST(BitBlock, FromLanesRoundtrip) {
  const auto orig = make_lanes({10, 20, 30, 40, 50, 60, 70, 80});
  EXPECT_EQ(BitBlock::from_lanes(orig).extract(), orig);
}

TEST(BitBlock, XorIsItsOwnInverse) {
  auto a = BitBlock::from_lanes(make_lanes({1, 2, 3, 4, 5, 6, 7, 8}));
  auto orig = a.extract();
  a ^= a;
  for (int32_t lane : a.extract())
    EXPECT_EQ(lane, 0);
  (void)orig;
}

TEST(BitBlock, NotDoubleNegation) {
  auto a = BitBlock::from_lanes(make_lanes({1, -1, 0x7FFF'FFFF, 0}));
  EXPECT_EQ((~~a).extract(), a.extract());
}

// ===========================================================================
// BitVector construction
// ===========================================================================

TEST(BitVector, DefaultConstructionAllZero) {
  BitVector bv(256);
  EXPECT_EQ(bv.block_count(), 2u);  // (256/256)+1 = 2
  EXPECT_EQ(bv.popcount(), 0);
}

TEST(BitVector, FromBlocks) {
  auto bv = BitVector::from_blocks(3);
  EXPECT_EQ(bv.block_count(), 3u);
  EXPECT_EQ(bv.size(), 768u);
}

TEST(BitVector, FromIntegersRoundtrip) {
  // Pack 8 int64_t values (fills 2 blocks of 256 bits each).
  std::vector<int64_t> in = {1LL, -1LL, 0x0F0F'0F0F'0F0F'0F0FLL,
                              0x1234'5678'9ABC'DEF0LL,
                              0, INT64_MIN, INT64_MAX, 42LL};
  auto bv = BitVector::from_integers(in);
  auto out = bv.to_integer_vec();
  ASSERT_EQ(out.size(), in.size());
  for (size_t i = 0; i < in.size(); ++i)
    EXPECT_EQ(out[i], in[i]) << "mismatch at index " << i;
}

TEST(BitVector, FromLanesRoundtrip) {
  std::vector<int32_t> lanes(16, 0);
  for (int i = 0; i < 16; ++i)
    lanes[i] = i + 1;
  auto bv = BitVector::from_lanes(lanes);
  ASSERT_EQ(bv.block_count(), 2u);
  auto b0 = bv.blocks()[0].extract();
  for (int j = 0; j < 8; ++j)
    EXPECT_EQ(b0[j], j + 1);
  auto b1 = bv.blocks()[1].extract();
  for (int j = 0; j < 8; ++j)
    EXPECT_EQ(b1[j], j + 9);
}

// ===========================================================================
// Single-bit access: get / xor_bit
// ===========================================================================

TEST(BitVector, GetXorBitBit0) {
  BitVector bv(256);
  EXPECT_FALSE(bv.get(0));
  bv.xor_bit(0);
  EXPECT_TRUE(bv.get(0));
  bv.xor_bit(0);
  EXPECT_FALSE(bv.get(0));
}

TEST(BitVector, GetXorBitLastInLane) {
  // Bit 31 = last bit of lane 0 in block 0.
  BitVector bv(256);
  bv.xor_bit(31);
  EXPECT_TRUE(bv.get(31));
  EXPECT_FALSE(bv.get(30));
  EXPECT_FALSE(bv.get(32));
}

TEST(BitVector, GetXorBitFirstInSecondLane) {
  // Bit 32 = first bit of lane 1 in block 0.
  BitVector bv(256);
  bv.xor_bit(32);
  EXPECT_FALSE(bv.get(31));
  EXPECT_TRUE(bv.get(32));
  EXPECT_FALSE(bv.get(33));
}

TEST(BitVector, GetXorBitLastInFirstBlock) {
  // Bit 255 = last bit in block 0.
  BitVector bv(512);
  bv.xor_bit(255);
  EXPECT_TRUE(bv.get(255));
  EXPECT_FALSE(bv.get(254));
  EXPECT_FALSE(bv.get(256));
}

TEST(BitVector, GetXorBitFirstInSecondBlock) {
  // Bit 256 = first bit of block 1.
  BitVector bv(512);
  bv.xor_bit(256);
  EXPECT_FALSE(bv.get(255));
  EXPECT_TRUE(bv.get(256));
  EXPECT_FALSE(bv.get(257));
}

// ===========================================================================
// Bulk operations: ^=, &=, negate
// ===========================================================================

TEST(BitVector, XorAssignSelfIsZero) {
  BitVector bv(256);
  bv.xor_bit(0);
  bv.xor_bit(100);
  bv ^= bv;
  EXPECT_EQ(bv.popcount(), 0);
}

TEST(BitVector, AndAssignWithZeroIsZero) {
  BitVector a(256);
  BitVector b(256);
  a.xor_bit(7);
  a.xor_bit(255);
  a &= b;
  EXPECT_EQ(a.popcount(), 0);
}

TEST(BitVector, AndAssignPreservesIntersection) {
  BitVector a(256);
  BitVector b(256);
  a.xor_bit(10);
  a.xor_bit(20);
  b.xor_bit(20);
  b.xor_bit(30);
  a &= b;
  EXPECT_TRUE(a.get(20));
  EXPECT_FALSE(a.get(10));
  EXPECT_FALSE(a.get(30));
}

TEST(BitVector, NegateFlipsAllBits) {
  BitVector bv(256);
  bv.negate();
  EXPECT_EQ(bv.popcount(), static_cast<int32_t>(bv.size()));
  bv.negate();
  EXPECT_EQ(bv.popcount(), 0);
}

TEST(BitVector, FreeXorOperator) {
  BitVector a(256);
  BitVector b(256);
  a.xor_bit(5);
  b.xor_bit(5);
  b.xor_bit(10);
  auto c = a ^ b;
  EXPECT_FALSE(c.get(5));
  EXPECT_TRUE(c.get(10));
}

TEST(BitVector, FreeAndOperator) {
  BitVector a(256);
  BitVector b(256);
  a.xor_bit(3);
  a.xor_bit(7);
  b.xor_bit(7);
  b.xor_bit(9);
  auto c = a & b;
  EXPECT_FALSE(c.get(3));
  EXPECT_TRUE(c.get(7));
  EXPECT_FALSE(c.get(9));
}

// ===========================================================================
// Bit scanning: first_one / all_ones
// ===========================================================================

TEST(BitVector, FirstOneEmptyIsNullopt) {
  BitVector bv(512);
  EXPECT_EQ(bv.first_one(), std::nullopt);
}

TEST(BitVector, FirstOneBit0) {
  BitVector bv(256);
  bv.xor_bit(0);
  EXPECT_EQ(bv.first_one(), std::optional<size_t>(0));
}

TEST(BitVector, FirstOneMiddleOfSecondBlock) {
  BitVector bv(512);
  bv.xor_bit(300);
  EXPECT_EQ(bv.first_one(), std::optional<size_t>(300));
}

TEST(BitVector, FirstOneSelectsSmallest) {
  BitVector bv(512);
  bv.xor_bit(200);
  bv.xor_bit(50);
  bv.xor_bit(300);
  EXPECT_EQ(bv.first_one(), std::optional<size_t>(50));
}

TEST(BitVector, AllOnesEmptyVector) {
  BitVector bv(256);
  EXPECT_TRUE(bv.all_ones(256).empty());
}

TEST(BitVector, AllOnesKnownPattern) {
  BitVector bv(256);
  bv.xor_bit(0);
  bv.xor_bit(31);
  bv.xor_bit(32);
  bv.xor_bit(255);
  auto ones = bv.all_ones(256);
  ASSERT_EQ(ones.size(), 4u);
  EXPECT_EQ(ones[0], 0u);
  EXPECT_EQ(ones[1], 31u);
  EXPECT_EQ(ones[2], 32u);
  EXPECT_EQ(ones[3], 255u);
}

TEST(BitVector, AllOnesRespectsNbBitsLimit) {
  BitVector bv(512);
  bv.xor_bit(10);
  bv.xor_bit(300);  // beyond the limit
  auto ones = bv.all_ones(256);
  ASSERT_EQ(ones.size(), 1u);
  EXPECT_EQ(ones[0], 10u);
}

TEST(BitVector, AllOnesMultiBlock) {
  BitVector bv(512);
  bv.xor_bit(1);
  bv.xor_bit(255);
  bv.xor_bit(256);
  bv.xor_bit(511);
  auto ones = bv.all_ones(512);
  ASSERT_EQ(ones.size(), 4u);
  EXPECT_EQ(ones[0], 1u);
  EXPECT_EQ(ones[1], 255u);
  EXPECT_EQ(ones[2], 256u);
  EXPECT_EQ(ones[3], 511u);
}

// ===========================================================================
// Conversions: to_bool_vec / to_integer_vec
// ===========================================================================

TEST(BitVector, ToBoolVecSize) {
  BitVector bv(512);
  EXPECT_EQ(bv.to_bool_vec().size(), bv.size());
}

TEST(BitVector, ToBoolVecSetBit) {
  BitVector bv(256);
  bv.xor_bit(13);
  auto bools = bv.to_bool_vec();
  for (size_t i = 0; i < bools.size(); ++i)
    EXPECT_EQ(bools[i], i == 13) << "at index " << i;
}

TEST(BitVector, ToIntegerVecSize) {
  auto bv = BitVector::from_blocks(3);
  // 3 blocks × 4 int64_t each = 12
  EXPECT_EQ(bv.to_integer_vec().size(), 12u);
}

TEST(BitVector, IntegerVecZeroRoundtrip) {
  std::vector<int64_t> zeros(8, 0);
  auto bv = BitVector::from_integers(zeros);
  auto out = bv.to_integer_vec();
  ASSERT_EQ(out.size(), 8u);
  for (auto v : out)
    EXPECT_EQ(v, 0);
}

// ===========================================================================
// Popcount
// ===========================================================================

TEST(BitVector, PopcountZero) {
  BitVector bv(256);
  EXPECT_EQ(bv.popcount(), 0);
}

TEST(BitVector, PopcountAllOnes) {
  BitVector bv(256);
  bv.negate();
  EXPECT_EQ(bv.popcount(), static_cast<int32_t>(bv.size()));
}

TEST(BitVector, PopcountSingleBit) {
  BitVector bv(512);
  bv.xor_bit(123);
  EXPECT_EQ(bv.popcount(), 1);
}

TEST(BitVector, PopcountMultipleBits) {
  BitVector bv(256);
  bv.xor_bit(0);
  bv.xor_bit(1);
  bv.xor_bit(255);
  EXPECT_EQ(bv.popcount(), 3);
}

TEST(BitVector, PopcountKnownLaneValue) {
  // Set lane 0 of block 0 to 0b1010'1010 = 0xAA (4 set bits).
  std::array<int32_t, 8> lanes{};
  lanes[0] = 0xAA;
  auto bv = BitVector::from_lanes(lanes);
  EXPECT_EQ(bv.popcount(), 4);
}

// ===========================================================================
// extend
// ===========================================================================

TEST(BitVector, ExtendFromZero) {
  // std::vector<bool> uses bit packing so it is incompatible with
  // std::span<const bool>; use std::array<bool, N> instead.
  BitVector bv(256);
  const std::array<bool, 4> bits = {true, false, true, true};
  bv.extend(bits, 0);
  EXPECT_TRUE(bv.get(0));
  EXPECT_FALSE(bv.get(1));
  EXPECT_TRUE(bv.get(2));
  EXPECT_TRUE(bv.get(3));
  EXPECT_FALSE(bv.get(4));
}

TEST(BitVector, ExtendAcrossLaneBoundary) {
  // Start at bit 30 so the extension crosses the lane boundary at bit 32.
  BitVector bv(256);
  const std::array<bool, 5> bits = {false, true, false, true, false};
  bv.extend(bits, 30);
  // bits[0]=false → bit 30, bits[1]=true → bit 31, bits[2]=false → bit 32,
  // bits[3]=true  → bit 33, bits[4]=false → bit 34
  EXPECT_FALSE(bv.get(30));
  EXPECT_TRUE(bv.get(31));
  EXPECT_FALSE(bv.get(32));
  EXPECT_TRUE(bv.get(33));
  EXPECT_FALSE(bv.get(34));
}

TEST(BitVector, ExtendAcrossBlockBoundary) {
  // Start at bit 254 so the extension crosses the block boundary at 256.
  BitVector bv(512);
  const std::array<bool, 4> bits = {true, false, true, false};
  bv.extend(bits, 254);
  EXPECT_TRUE(bv.get(254));
  EXPECT_FALSE(bv.get(255));
  EXPECT_TRUE(bv.get(256));
  EXPECT_FALSE(bv.get(257));
}

} // namespace cudaq::synth
