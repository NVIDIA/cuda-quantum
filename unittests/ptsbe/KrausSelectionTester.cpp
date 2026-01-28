/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/KrausSelection.h"

using namespace cudaq;

CUDAQ_TEST(KrausSelectionTest, DefaultConstruction) {
  KrausSelection selection;
  EXPECT_EQ(selection.circuit_location, 0);
  EXPECT_TRUE(selection.qubits.empty());
  EXPECT_TRUE(selection.op_name.empty());
  EXPECT_EQ(static_cast<std::size_t>(selection.kraus_operator_index), 0);
}

CUDAQ_TEST(KrausSelectionTest, ParameterizedConstruction) {
  KrausSelection selection(5,                    // circuit_location
                           {0, 1},               // qubits
                           "cx",                 // op_name
                           KrausOperatorIndex{2} // kraus_operator_index
  );

  EXPECT_EQ(selection.circuit_location, 5);
  EXPECT_EQ(selection.qubits.size(), 2);
  EXPECT_EQ(selection.qubits[0], 0);
  EXPECT_EQ(selection.qubits[1], 1);
  EXPECT_EQ(selection.op_name, "cx");
  EXPECT_EQ(static_cast<std::size_t>(selection.kraus_operator_index), 2);
}

CUDAQ_TEST(KrausSelectionTest, Equality) {
  KrausSelection sel1(0, {0}, "h", KrausOperatorIndex{1});
  KrausSelection sel2(0, {0}, "h", KrausOperatorIndex{1});
  KrausSelection sel3(0, {0}, "h", KrausOperatorIndex{2}); // Different index
  KrausSelection sel4(1, {0}, "h", KrausOperatorIndex{1}); // Different location

  EXPECT_TRUE(sel1 == sel2);
  EXPECT_FALSE(sel1 == sel3);
  EXPECT_FALSE(sel1 == sel4);
}

CUDAQ_TEST(KrausSelectionTest, KrausOperatorIndexIdentity) {
  EXPECT_EQ(static_cast<std::size_t>(KrausOperatorIndex::IDENTITY), 0);

  KrausOperatorIndex idx{3};
  EXPECT_EQ(static_cast<std::size_t>(idx), 3);
}

CUDAQ_TEST(KrausSelectionTest, SingleQubitNoise) {
  KrausSelection h_noise(0,                    // circuit_location
                         {0},                  // Single qubit
                         "h",                  // op_name
                         KrausOperatorIndex{1} // X error
  );

  EXPECT_EQ(h_noise.qubits.size(), 1);
  EXPECT_EQ(h_noise.qubits[0], 0);
  EXPECT_EQ(h_noise.op_name, "h");
}

CUDAQ_TEST(KrausSelectionTest, TwoQubitNoise) {
  KrausSelection cx_noise(5,                    // circuit_location
                          {0, 1},               // Both qubits affected
                          "cx",                 // op_name
                          KrausOperatorIndex{7} // IX error
  );

  EXPECT_EQ(cx_noise.qubits.size(), 2);
  EXPECT_EQ(cx_noise.qubits[0], 0);
  EXPECT_EQ(cx_noise.qubits[1], 1);
  EXPECT_EQ(static_cast<std::size_t>(cx_noise.kraus_operator_index), 7);
}

CUDAQ_TEST(KrausSelectionTest, CopySemantics) {
  KrausSelection original(1, {0, 1}, "cx", KrausOperatorIndex{3});

  KrausSelection copied = original;
  EXPECT_TRUE(copied == original);
  EXPECT_EQ(copied.qubits[0], 0);

  KrausSelection assigned;
  assigned = original;
  EXPECT_TRUE(assigned == original);
}

CUDAQ_TEST(KrausSelectionTest, MoveSemantics) {
  KrausSelection original(1, {0, 1, 2}, "ccx", KrausOperatorIndex{5});

  KrausSelection moved = std::move(original);
  EXPECT_EQ(moved.circuit_location, 1);
  EXPECT_EQ(moved.qubits.size(), 3);
  EXPECT_EQ(moved.op_name, "ccx");
}

CUDAQ_TEST(KrausSelectionTest, ConstexprEquality) {
  KrausSelection sel1(0, {0}, "h", KrausOperatorIndex{1});
  KrausSelection sel2(0, {0}, "h", KrausOperatorIndex{1});
  KrausSelection sel3(1, {0}, "h", KrausOperatorIndex{1});

  EXPECT_TRUE(sel1 == sel2);
  EXPECT_FALSE(sel1 == sel3);
}
