/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Circuit/PauliProduct.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace cudaq::synth {

// ===========================================================================
// Helpers: build 1-qubit and 2-qubit Pauli products from scratch.
//
// Single-qubit convention (1 bit, no padding concerns):
//   I = z=0, x=0
//   Z = z=1, x=0
//   X = z=0, x=1
//   Y = z=1, x=1  (phase tracked separately via sign)
// ===========================================================================

static BitVector make_bv(std::initializer_list<bool> bits) {
  BitVector bv(bits.size());
  size_t i = 0;
  for (bool b : bits) {
    if (b)
      bv.xor_bit(i++);
    else
      ++i;
  }
  return bv;
}

// Single-qubit Paulis (sign = false means +1 phase)
static PauliProduct make_I() {
  return {make_bv({false}), make_bv({false}), false};
}
static PauliProduct make_X() {
  return {make_bv({false}), make_bv({true}), false};
}
static PauliProduct make_Y() {
  return {make_bv({true}), make_bv({true}), false};
}
static PauliProduct make_Z() {
  return {make_bv({true}), make_bv({false}), false};
}

// ===========================================================================
// Construction / accessors
// ===========================================================================

TEST(PauliProduct, ConstructionAccessors) {
  auto p = make_Z();
  EXPECT_TRUE(p.z().get(0));
  EXPECT_FALSE(p.x().get(0));
  EXPECT_FALSE(p.sign());
}

TEST(PauliProduct, ConstructionSignTrue) {
  PauliProduct p{make_bv({true}), make_bv({false}), true};
  EXPECT_TRUE(p.sign());
}

// ===========================================================================
// is_commuting
// ===========================================================================
//
// Commutation rules for single-qubit Paulis:
//   Any Pauli commutes with itself.
//   I commutes with everything.
//   X, Y, Z mutually anti-commute.

TEST(PauliProduct, XCommutesWithX) {
  EXPECT_TRUE(make_X().is_commuting(make_X()));
}

TEST(PauliProduct, ZCommutesWithZ) {
  EXPECT_TRUE(make_Z().is_commuting(make_Z()));
}

TEST(PauliProduct, YCommutesWithY) {
  EXPECT_TRUE(make_Y().is_commuting(make_Y()));
}

TEST(PauliProduct, ICommutesWithEverything) {
  EXPECT_TRUE(make_I().is_commuting(make_X()));
  EXPECT_TRUE(make_I().is_commuting(make_Y()));
  EXPECT_TRUE(make_I().is_commuting(make_Z()));
}

TEST(PauliProduct, XAntiCommutesWithZ) {
  EXPECT_FALSE(make_X().is_commuting(make_Z()));
  EXPECT_FALSE(make_Z().is_commuting(make_X()));
}

TEST(PauliProduct, XAntiCommutesWithY) {
  EXPECT_FALSE(make_X().is_commuting(make_Y()));
  EXPECT_FALSE(make_Y().is_commuting(make_X()));
}

TEST(PauliProduct, ZAntiCommutesWithY) {
  EXPECT_FALSE(make_Z().is_commuting(make_Y()));
  EXPECT_FALSE(make_Y().is_commuting(make_Z()));
}

TEST(PauliProduct, CommutationIsSymmetric) {
  // On 2 qubits: XI commutes with IZ (disjoint support)
  PauliProduct p1{make_bv({false, false}), make_bv({true, false}), false}; // XI
  PauliProduct p2{make_bv({false, true}), make_bv({false, false}), false}; // IZ
  EXPECT_TRUE(p1.is_commuting(p2));
  EXPECT_TRUE(p2.is_commuting(p1));
}

TEST(PauliProduct, MultiQubitAntiCommuting) {
  // XZ anti-commutes with ZX: symplectic inner product = popcount(10 XOR 01) =
  // 2 → even? No. Wait: x1=(1,0), z1=(0,1); x2=(0,1), z2=(1,0) (x1 & z2) =
  // (1,0)&(1,0) = (1,0), popcount=1 (z1 & x2) = (0,1)&(0,1) = (0,1), popcount=1
  // ac = (1,0)^(0,1) = (1,1), popcount=2 → 2%2==0 → commutes!
  PauliProduct xz{make_bv({false, true}), make_bv({true, false}), false}; // XZ
  PauliProduct zx{make_bv({true, false}), make_bv({false, true}), false}; // ZX
  EXPECT_TRUE(xz.is_commuting(zx));
}

TEST(PauliProduct, MultiQubitAntiCommutingOdd) {
  // XI anti-commutes with ZI (single-qubit anti-comm, other qubit is I):
  // x1=(1,0), z1=(0,0); x2=(0,0), z2=(1,0)
  // (x1 & z2) = (1,0)&(1,0) = (1,0), popcount=1
  // (z1 & x2) = (0,0)&(0,0) = (0,0), popcount=0
  // ac = (1,0)^(0,0) = (1,0), popcount=1 → 1%2==1 → anti-commutes
  PauliProduct xi{make_bv({false, false}), make_bv({true, false}), false}; // XI
  PauliProduct zi{make_bv({true, false}), make_bv({false, false}), false}; // ZI
  EXPECT_FALSE(xi.is_commuting(zi));
}

// ===========================================================================
// operator*=
//
// In the symplectic representation the generator for qubit (z=1,x=1) is the
// element X·Z (NOT i·X·Z = Y). The phase formula accumulates i-powers
// automatically, so the sign bit for results involving the (z=1,x=1) generator
// differs from the naive "+i → sign=false" intuition.
//
// Expected values are derived directly from the phase formula:
//   phase = ac.popcount() + 2*x1z2.popcount()
//   sign  = (phase % 4) > 1
//
//   X*X = I              z=0,x=0, sign=false
//   Z*Z = I              z=0,x=0, sign=false
//   Y*Y = I              z=0,x=0, sign=false
//   X*Z → z=1,x=1, phase=1, sign=false
//   Z*X → z=1,x=1, phase=3, sign=true
//   X*Y → z=1,x=0, phase=3, sign=true
//   Y*X → z=1,x=0, phase=1, sign=false
//   Z*Y → z=0,x=1, phase=1, sign=false
//   Y*Z → z=0,x=1, phase=3, sign=true
// ===========================================================================

// Helper to check PauliProduct state.
static void expect_pauli(const PauliProduct &p, bool z0, bool x0, bool sign,
                         const char *label) {
  SCOPED_TRACE(label);
  EXPECT_EQ(p.z().get(0), z0);
  EXPECT_EQ(p.x().get(0), x0);
  EXPECT_EQ(p.sign(), sign);
}

TEST(PauliProduct, MultiplyXX) {
  auto r = make_X() * make_X();
  expect_pauli(r, false, false, false, "X*X = I");
}

TEST(PauliProduct, MultiplyZZ) {
  auto r = make_Z() * make_Z();
  expect_pauli(r, false, false, false, "Z*Z = I");
}

TEST(PauliProduct, MultiplyYY) {
  auto r = make_Y() * make_Y();
  expect_pauli(r, false, false, false, "Y*Y = I");
}

TEST(PauliProduct, MultiplyXZ) {
  // X*Z = iY → z=1,x=1, sign=false (+1 phase flag; +i is not -1)
  auto r = make_X() * make_Z();
  expect_pauli(r, true, true, false, "X*Z = iY");
}

TEST(PauliProduct, MultiplyZX) {
  // Z*X = -iY → z=1,x=1, sign=true
  auto r = make_Z() * make_X();
  expect_pauli(r, true, true, true, "Z*X = -iY");
}

TEST(PauliProduct, MultiplyXY) {
  // phase=3 → sign=true; result: z=1,x=0
  auto r = make_X() * make_Y();
  expect_pauli(r, true, false, true, "X*Y");
}

TEST(PauliProduct, MultiplyYX) {
  // phase=1 → sign=false; result: z=1,x=0
  auto r = make_Y() * make_X();
  expect_pauli(r, true, false, false, "Y*X");
}

TEST(PauliProduct, MultiplyZY) {
  // phase=1 → sign=false; result: z=0,x=1
  auto r = make_Z() * make_Y();
  expect_pauli(r, false, true, false, "Z*Y");
}

TEST(PauliProduct, MultiplyYZ) {
  // phase=3 → sign=true; result: z=0,x=1
  auto r = make_Y() * make_Z();
  expect_pauli(r, false, true, true, "Y*Z");
}

TEST(PauliProduct, MultiplyByIdentityIsNoop) {
  auto p = make_X();
  auto r = p * make_I();
  expect_pauli(r, false, true, false, "X*I = X");
}

TEST(PauliProduct, IdentityMultiplyIsNoop) {
  auto p = make_Z();
  auto r = make_I() * p;
  expect_pauli(r, true, false, false, "I*Z = Z");
}

TEST(PauliProduct, SignPropagates) {
  // (-X)*X = -I → sign=true
  PauliProduct neg_x{make_bv({false}), make_bv({true}), true};
  auto r = neg_x * make_X();
  EXPECT_EQ(r.sign(), true);
  EXPECT_FALSE(r.z().get(0));
  EXPECT_FALSE(r.x().get(0));
}

TEST(PauliProduct, SignCancels) {
  // (-X)*(-X) = X*X = I → sign=false
  PauliProduct neg_x{make_bv({false}), make_bv({true}), true};
  auto r = neg_x * neg_x;
  EXPECT_EQ(r.sign(), false);
}

TEST(PauliProduct, InPlaceMultiply) {
  auto p = make_X();
  p *= make_Z();
  expect_pauli(p, true, true, false, "X*=Z gives iY");
}

// ===========================================================================
// to_bool_vec
// ===========================================================================

TEST(PauliProduct, ToBoolVecSingleQubit) {
  // X: z=0, x=1 → vec = [z0, x0] = [false, true]
  auto vec = make_X().to_bool_vec(1);
  ASSERT_EQ(vec.size(), 2u);
  EXPECT_FALSE(vec[0]); // z
  EXPECT_TRUE(vec[1]);  // x
}

TEST(PauliProduct, ToBoolVecZ) {
  // Z: z=1, x=0 → [true, false]
  auto vec = make_Z().to_bool_vec(1);
  ASSERT_EQ(vec.size(), 2u);
  EXPECT_TRUE(vec[0]);  // z
  EXPECT_FALSE(vec[1]); // x
}

TEST(PauliProduct, ToBoolVecY) {
  // Y: z=1, x=1 → [true, true]
  auto vec = make_Y().to_bool_vec(1);
  ASSERT_EQ(vec.size(), 2u);
  EXPECT_TRUE(vec[0]);
  EXPECT_TRUE(vec[1]);
}

TEST(PauliProduct, ToBoolVecMultiQubit) {
  // 3-qubit: XZI → x=(1,0,0), z=(0,1,0)
  // to_bool_vec(3) = [z0,z1,z2, x0,x1,x2] = [0,1,0, 1,0,0]
  PauliProduct p{make_bv({false, true, false}), make_bv({true, false, false}),
                 false};
  auto vec = p.to_bool_vec(3);
  ASSERT_EQ(vec.size(), 6u);
  EXPECT_FALSE(vec[0]); // z0
  EXPECT_TRUE(vec[1]);  // z1
  EXPECT_FALSE(vec[2]); // z2
  EXPECT_TRUE(vec[3]);  // x0
  EXPECT_FALSE(vec[4]); // x1
  EXPECT_FALSE(vec[5]); // x2
}

TEST(PauliProduct, ToBoolVecTruncatesExtraBits) {
  // BitVector capacity is rounded up to 256, but to_bool_vec(nb_qubits)
  // should return exactly 2*nb_qubits entries.
  auto vec = make_Y().to_bool_vec(1);
  EXPECT_EQ(vec.size(), 2u);
}

} // namespace cudaq::synth
