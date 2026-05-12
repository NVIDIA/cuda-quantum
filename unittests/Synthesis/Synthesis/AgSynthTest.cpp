/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Synthesis/AgSynth.h"

#include <gtest/gtest.h>

using namespace cudaq::synth;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void expect_identity(const Tableau &tab) {
  const size_t n = tab.num_qubits();
  for (size_t i = 0; i < n; ++i) {
    PauliProduct stab = tab.extract_pauli_product(i);
    EXPECT_FALSE(stab.sign()) << "stab " << i << " sign";
    for (size_t q = 0; q < n; ++q) {
      EXPECT_EQ(stab.z().get(q), q == i) << "stab " << i << " z[" << q << "]";
      EXPECT_FALSE(stab.x().get(q)) << "stab " << i << " x[" << q << "]";
    }
    PauliProduct destab = tab.extract_pauli_product(i + n);
    EXPECT_FALSE(destab.sign()) << "destab " << i << " sign";
    for (size_t q = 0; q < n; ++q) {
      EXPECT_FALSE(destab.z().get(q)) << "destab " << i << " z[" << q << "]";
      EXPECT_EQ(destab.x().get(q), q == i)
          << "destab " << i << " x[" << q << "]";
    }
  }
}

static void apply_circuit(Tableau &tab, const CliffordCircuit &circ) {
  for (const CliffordGate &g : circ) {
    switch (g.kind) {
    case CliffordGateKind::H:
      tab.append_h(g.qubit0);
      break;
    case CliffordGateKind::S:
      tab.append_s(g.qubit0);
      break;
    case CliffordGateKind::X:
      tab.append_x(g.qubit0);
      break;
    case CliffordGateKind::Z:
      tab.append_z(g.qubit0);
      break;
    case CliffordGateKind::CX:
      tab.append_cx(g.qubit0, g.qubit1);
      break;
    case CliffordGateKind::CZ:
      tab.append_cz(g.qubit0, g.qubit1);
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Synthesis round-trip
// ---------------------------------------------------------------------------

// After synthesizing and applying the inverse circuit, the tableau becomes
// the identity.
static void test_synthesis_roundtrip(Tableau &tab) {
  const size_t n = tab.num_qubits();
  CliffordCircuit inv = ag_synth_inverse(tab);
  apply_circuit(tab, inv);
  expect_identity(tab);
  EXPECT_EQ(tab.num_qubits(), n);
}

TEST(AgSynthSynthesis, IdentityRoundTrip) {
  Tableau tab(2);
  test_synthesis_roundtrip(tab);
}

TEST(AgSynthSynthesis, SingleH_RoundTrip) {
  Tableau tab(2);
  tab.append_h(0);
  test_synthesis_roundtrip(tab);
}

TEST(AgSynthSynthesis, SingleS_RoundTrip) {
  Tableau tab(2);
  tab.append_s(1);
  test_synthesis_roundtrip(tab);
}

TEST(AgSynthSynthesis, CX_RoundTrip) {
  Tableau tab(2);
  tab.append_cx(0, 1);
  test_synthesis_roundtrip(tab);
}

TEST(AgSynthSynthesis, HSH_RoundTrip) {
  Tableau tab(3);
  tab.append_h(0);
  tab.append_s(0);
  tab.append_h(0);
  test_synthesis_roundtrip(tab);
}

TEST(AgSynthSynthesis, BellPrep_RoundTrip) {
  Tableau tab(2);
  tab.append_h(0);
  tab.append_cx(0, 1);
  test_synthesis_roundtrip(tab);
}

TEST(AgSynthSynthesis, ThreeQubit_RoundTrip) {
  Tableau tab(3);
  tab.append_h(0);
  tab.append_cx(0, 1);
  tab.append_cx(1, 2);
  tab.append_s(2);
  tab.append_h(1);
  test_synthesis_roundtrip(tab);
}

// Forward circuit round-trip: apply ag_synth() to identity and verify result
// matches the original tableau.
TEST(AgSynthSynthesis, ForwardCircuit_MatchesOriginal) {
  Tableau original(2);
  original.append_h(0);
  original.append_cx(0, 1);

  CliffordCircuit fwd = ag_synth(original);

  Tableau rebuilt(2);
  apply_circuit(rebuilt, fwd);

  const size_t n = original.num_qubits();
  for (size_t col = 0; col < 2 * n; ++col) {
    PauliProduct p_orig = original.extract_pauli_product(col);
    PauliProduct p_rebuilt = rebuilt.extract_pauli_product(col);
    EXPECT_EQ(p_orig.sign(), p_rebuilt.sign()) << "col=" << col;
    for (size_t q = 0; q < n; ++q) {
      EXPECT_EQ(p_orig.z().get(q), p_rebuilt.z().get(q))
          << "col=" << col << " z[" << q << "]";
      EXPECT_EQ(p_orig.x().get(q), p_rebuilt.x().get(q))
          << "col=" << col << " x[" << q << "]";
    }
  }
}
