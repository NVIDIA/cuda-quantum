/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Circuit/CliffordCircuit.h"
#include "Circuit/Tableau.h"

#include <gtest/gtest.h>

using namespace cudaq::synth;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Verify that the identity tableau has stabilizer Z_i at column i and
// destabilizer X_i at column i+n, all with sign=false.
static void expect_identity(const Tableau &tab) {
  const size_t n = tab.num_qubits();
  for (size_t i = 0; i < n; ++i) {
    // stabilizer i: Z_i
    PauliProduct stab = tab.extract_pauli_product(i);
    EXPECT_FALSE(stab.sign()) << "stab " << i << " sign";
    for (size_t q = 0; q < n; ++q) {
      EXPECT_EQ(stab.z().get(q), q == i) << "stab " << i << " z[" << q << "]";
      EXPECT_FALSE(stab.x().get(q)) << "stab " << i << " x[" << q << "]";
    }
    // destabilizer i: X_i
    PauliProduct destab = tab.extract_pauli_product(i + n);
    EXPECT_FALSE(destab.sign()) << "destab " << i << " sign";
    for (size_t q = 0; q < n; ++q) {
      EXPECT_FALSE(destab.z().get(q)) << "destab " << i << " z[" << q << "]";
      EXPECT_EQ(destab.x().get(q), q == i) << "destab " << i << " x[" << q << "]";
    }
  }
}

// Apply a CliffordCircuit to a Tableau by appending each gate.
static void apply_circuit(Tableau &tab, const CliffordCircuit &circ) {
  for (const CliffordGate &g : circ) {
    switch (g.kind) {
    case CliffordGateKind::H:  tab.append_h(g.qubit0);  break;
    case CliffordGateKind::S:  tab.append_s(g.qubit0);  break;
    case CliffordGateKind::X:  tab.append_x(g.qubit0);  break;
    case CliffordGateKind::Z:  tab.append_z(g.qubit0);  break;
    case CliffordGateKind::CX: tab.append_cx(g.qubit0, g.qubit1); break;
    case CliffordGateKind::CZ: tab.append_cz(g.qubit0, g.qubit1); break;
    }
  }
}

// ---------------------------------------------------------------------------
// Identity construction
// ---------------------------------------------------------------------------

TEST(TableauIdentity, SingleQubit) {
  Tableau tab(1);
  ASSERT_EQ(tab.num_qubits(), 1u);
  expect_identity(tab);
}

TEST(TableauIdentity, TwoQubits) {
  Tableau tab(2);
  ASSERT_EQ(tab.num_qubits(), 2u);
  expect_identity(tab);
}

TEST(TableauIdentity, FourQubits) {
  Tableau tab(4);
  expect_identity(tab);
}

// ---------------------------------------------------------------------------
// Single-qubit appends
// ---------------------------------------------------------------------------

// append_x: X†Z_iX = -Z_i (stabilizer gets sign flip), X†X_iX = X_i (no change)
TEST(TableauAppend, X_FlipsStabSign) {
  Tableau tab(2);
  tab.append_x(0);
  // Stabilizer 0 should now be -Z_0
  PauliProduct s0 = tab.extract_pauli_product(0);
  EXPECT_TRUE(s0.sign());
  EXPECT_TRUE(s0.z().get(0));
  EXPECT_FALSE(s0.x().get(0));
  // Stabilizer 1 should be unchanged
  PauliProduct s1 = tab.extract_pauli_product(1);
  EXPECT_FALSE(s1.sign());
  EXPECT_TRUE(s1.z().get(1));
}

// append_z: Z†X_iZ = -X_i
TEST(TableauAppend, Z_FlipsDestabSign) {
  Tableau tab(2);
  tab.append_z(0);
  // Destabilizer 0 should now be -X_0
  PauliProduct d0 = tab.extract_pauli_product(2);
  EXPECT_TRUE(d0.sign());
  EXPECT_FALSE(d0.z().get(0));
  EXPECT_TRUE(d0.x().get(0));
}

// append_s: S†Z_iS = Z_i (unchanged), S†X_iS = Y_i
TEST(TableauAppend, S_TransformsX_to_Y) {
  Tableau tab(1);
  tab.append_s(0);
  // Stabilizer 0: S†Z_0S = Z_0, unchanged
  PauliProduct s0 = tab.extract_pauli_product(0);
  EXPECT_FALSE(s0.sign());
  EXPECT_TRUE(s0.z().get(0));
  EXPECT_FALSE(s0.x().get(0));
  // Destabilizer 0: S†X_0S = Y_0 = (z=1,x=1,sign=false)
  PauliProduct d0 = tab.extract_pauli_product(1);
  EXPECT_FALSE(d0.sign());
  EXPECT_TRUE(d0.z().get(0));
  EXPECT_TRUE(d0.x().get(0));
}

// append_h: H†Z_0H = X_0, H†X_0H = Z_0
TEST(TableauAppend, H_SwapsXandZ) {
  Tableau tab(1);
  tab.append_h(0);
  // Stabilizer 0: H†Z_0H = X_0
  PauliProduct s0 = tab.extract_pauli_product(0);
  EXPECT_FALSE(s0.sign());
  EXPECT_FALSE(s0.z().get(0));
  EXPECT_TRUE(s0.x().get(0));
  // Destabilizer 0: H†X_0H = Z_0
  PauliProduct d0 = tab.extract_pauli_product(1);
  EXPECT_FALSE(d0.sign());
  EXPECT_TRUE(d0.z().get(0));
  EXPECT_FALSE(d0.x().get(0));
}

// append_v implements exp(-i*pi/4 * X) = (I - iX)/sqrt(2).
// Conjugation action: Z -> -Y (sign flips, becomes Y), X -> X (unchanged).
TEST(TableauAppend, V_TransformsGenerators) {
  Tableau tab(1);
  tab.append_v(0);
  // Stabilizer 0 was Z_0. After V: Z -> -Y (z=1, x=1, sign=true)
  PauliProduct s0 = tab.extract_pauli_product(0);
  EXPECT_TRUE(s0.sign());
  EXPECT_TRUE(s0.z().get(0));
  EXPECT_TRUE(s0.x().get(0));
  // Destabilizer 0 was X_0. After V: X -> X (unchanged)
  PauliProduct d0 = tab.extract_pauli_product(1);
  EXPECT_FALSE(d0.sign());
  EXPECT_FALSE(d0.z().get(0));
  EXPECT_TRUE(d0.x().get(0));
}

// Applying H twice returns to identity.
TEST(TableauAppend, H_Squared_IsIdentity) {
  Tableau tab(2);
  tab.append_h(0);
  tab.append_h(0);
  expect_identity(tab);
}

// Applying X twice returns to identity.
TEST(TableauAppend, X_Squared_IsIdentity) {
  Tableau tab(3);
  tab.append_x(1);
  tab.append_x(1);
  expect_identity(tab);
}

// Applying S four times returns to identity.
TEST(TableauAppend, S_Fourth_Power_IsIdentity) {
  Tableau tab(2);
  for (int k = 0; k < 4; ++k)
    tab.append_s(0);
  expect_identity(tab);
}

// ---------------------------------------------------------------------------
// Two-qubit appends
// ---------------------------------------------------------------------------

// CX Heisenberg-picture action (append = right-multiply):
//   Z_ctrl -> Z_ctrl  (unchanged)
//   Z_targ -> Z_ctrl*Z_targ  (Z on target spreads to control row)
//   X_ctrl -> X_ctrl*X_targ  (X on control spreads to target row)
//   X_targ -> X_targ  (unchanged)
TEST(TableauAppend, CX_UpdatesGenerators) {
  Tableau tab(2);
  tab.append_cx(0, 1);
  // Stabilizer 0 (was Z_0): Z_ctrl unchanged -> still Z_0
  PauliProduct s0 = tab.extract_pauli_product(0);
  EXPECT_FALSE(s0.sign());
  EXPECT_TRUE(s0.z().get(0));
  EXPECT_FALSE(s0.z().get(1));
  // Stabilizer 1 (was Z_1): Z_targ -> Z_ctrl*Z_targ = Z_0*Z_1
  PauliProduct s1 = tab.extract_pauli_product(1);
  EXPECT_FALSE(s1.sign());
  EXPECT_TRUE(s1.z().get(0));
  EXPECT_TRUE(s1.z().get(1));
  // Destabilizer 0 (was X_0): X_ctrl -> X_ctrl*X_targ = X_0*X_1
  PauliProduct d0 = tab.extract_pauli_product(2);
  EXPECT_FALSE(d0.sign());
  EXPECT_TRUE(d0.x().get(0));
  EXPECT_TRUE(d0.x().get(1));
  // Destabilizer 1 (was X_1): X_targ unchanged -> still X_1
  PauliProduct d1 = tab.extract_pauli_product(3);
  EXPECT_FALSE(d1.sign());
  EXPECT_FALSE(d1.x().get(0));
  EXPECT_TRUE(d1.x().get(1));
}

// CX applied twice is identity.
TEST(TableauAppend, CX_Squared_IsIdentity) {
  Tableau tab(3);
  tab.append_cx(0, 2);
  tab.append_cx(0, 2);
  expect_identity(tab);
}

// CZ applied twice is identity.
TEST(TableauAppend, CZ_Squared_IsIdentity) {
  Tableau tab(2);
  tab.append_cz(0, 1);
  tab.append_cz(0, 1);
  expect_identity(tab);
}

// ---------------------------------------------------------------------------
// Prepend operations
// ---------------------------------------------------------------------------

TEST(TableauPrepend, X_FlipsStabSign) {
  Tableau tab(2);
  tab.prepend_x(0);
  // Signs bit for stabilizer 0 should be flipped.
  PauliProduct s0 = tab.extract_pauli_product(0);
  EXPECT_TRUE(s0.sign());
}

TEST(TableauPrepend, Z_FlipsDestabSign) {
  Tableau tab(2);
  tab.prepend_z(1);
  // Signs bit for destabilizer 1 should be flipped.
  PauliProduct d1 = tab.extract_pauli_product(3);
  EXPECT_TRUE(d1.sign());
}

TEST(TableauPrepend, H_SwapsColumns) {
  Tableau tab(2);
  // After prepend_h(0): stabilizer 0 <-> destabilizer 0
  tab.prepend_h(0);
  PauliProduct s0 = tab.extract_pauli_product(0);
  EXPECT_FALSE(s0.sign());
  EXPECT_FALSE(s0.z().get(0));
  EXPECT_TRUE(s0.x().get(0));
  PauliProduct d0 = tab.extract_pauli_product(2);
  EXPECT_FALSE(d0.sign());
  EXPECT_TRUE(d0.z().get(0));
  EXPECT_FALSE(d0.x().get(0));
}

TEST(TableauPrepend, S_MultipliesDestabByStab) {
  Tableau tab(1);
  // prepend_s(0): destab = destab * stab
  // Initial: stab=Z_0, destab=X_0
  // X_0 * Z_0 = (z=1,x=1,sign=true) in our convention
  tab.prepend_s(0);
  PauliProduct d0 = tab.extract_pauli_product(1);
  // X*Z product sign depends on PauliProduct multiply
  EXPECT_TRUE(d0.z().get(0));
  EXPECT_TRUE(d0.x().get(0));
}

// Prepend H twice = identity
TEST(TableauPrepend, H_Squared_IsIdentity) {
  Tableau tab(3);
  tab.prepend_h(1);
  tab.prepend_h(1);
  expect_identity(tab);
}

// Prepend CX twice = identity
TEST(TableauPrepend, CX_Squared_IsIdentity) {
  Tableau tab(3);
  tab.prepend_cx(0, 1);
  tab.prepend_cx(0, 1);
  expect_identity(tab);
}

// ---------------------------------------------------------------------------
// Extract / insert round-trip
// ---------------------------------------------------------------------------

TEST(TableauExtractInsert, RoundTrip) {
  Tableau tab(3);
  // Build a known Pauli product: qubit 0=Y, qubit 1=X, qubit 2=Z, sign=true
  BitVector z(3);
  z.xor_bit(0); // Y has z=1
  z.xor_bit(2); // Z has z=1
  BitVector x(3);
  x.xor_bit(0); // Y has x=1
  x.xor_bit(1); // X has x=1
  PauliProduct p(std::move(z), std::move(x), true);

  tab.insert_pauli_product(p, 0);
  PauliProduct extracted = tab.extract_pauli_product(0);

  EXPECT_TRUE(extracted.sign());
  EXPECT_TRUE(extracted.z().get(0));
  EXPECT_FALSE(extracted.z().get(1));
  EXPECT_TRUE(extracted.z().get(2));
  EXPECT_TRUE(extracted.x().get(0));
  EXPECT_TRUE(extracted.x().get(1));
  EXPECT_FALSE(extracted.x().get(2));
}

// insert_pauli_product *writes* a PauliProduct into a column (it sets
// each bit to match p, XOR-ing only where they differ). Inserting an
// all-zero Pauli into column 0 (which holds Z_0) overwrites it with I.
TEST(TableauExtractInsert, InsertOverwritesColumn) {
  Tableau tab(2);
  // Column 0 starts as Z_0: z[0]=1, x[0]=0, sign=false.
  BitVector z(2);
  BitVector x(2);
  PauliProduct zeros(std::move(z), std::move(x), false);
  tab.insert_pauli_product(zeros, 0);
  PauliProduct extracted = tab.extract_pauli_product(0);
  // Column 0 now stores the all-zero Pauli.
  EXPECT_FALSE(extracted.sign());
  EXPECT_FALSE(extracted.z().get(0));
  EXPECT_FALSE(extracted.x().get(0));
  EXPECT_FALSE(extracted.z().get(1));
  EXPECT_FALSE(extracted.x().get(1));
}

