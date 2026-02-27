/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
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
  EXPECT_EQ(selection.kraus_operator_index, 0);
  EXPECT_FALSE(selection.is_error);
}

CUDAQ_TEST(KrausSelectionTest, ParameterizedConstruction) {
  KrausSelection selection(5,      // circuit_location
                           {0, 1}, // qubits
                           "cx",   // op_name
                           2,      // kraus_operator_index
                           true    // is_error
  );

  EXPECT_EQ(selection.circuit_location, 5);
  EXPECT_EQ(selection.qubits.size(), 2);
  EXPECT_EQ(selection.qubits[0], 0);
  EXPECT_EQ(selection.qubits[1], 1);
  EXPECT_EQ(selection.op_name, "cx");
  EXPECT_EQ(selection.kraus_operator_index, 2);
  EXPECT_TRUE(selection.is_error);
}

CUDAQ_TEST(KrausSelectionTest, Equality) {
  KrausSelection sel1(0, {0}, "h", 1, true);
  KrausSelection sel2(0, {0}, "h", 1, true);
  KrausSelection sel3(0, {0}, "h", 2, true);  // Different index
  KrausSelection sel4(1, {0}, "h", 1, true);  // Different location
  KrausSelection sel5(0, {0}, "h", 1, false); // Different is_error

  EXPECT_TRUE(sel1 == sel2);
  EXPECT_FALSE(sel1 == sel3);
  EXPECT_FALSE(sel1 == sel4);
  EXPECT_FALSE(sel1 == sel5);
}

CUDAQ_TEST(KrausSelectionTest, IsErrorDefaultsFalse) {
  KrausSelection sel(0, {0}, "h", 0);
  EXPECT_FALSE(sel.is_error);

  KrausSelection sel_error(0, {0}, "h", 3, true);
  EXPECT_TRUE(sel_error.is_error);
}

CUDAQ_TEST(KrausSelectionTest, SingleQubitNoise) {
  KrausSelection h_noise(0,   // circuit_location
                         {0}, // Single qubit
                         "h", // op_name
                         1,   // X error
                         true // is_error
  );

  EXPECT_EQ(h_noise.qubits.size(), 1);
  EXPECT_EQ(h_noise.qubits[0], 0);
  EXPECT_EQ(h_noise.op_name, "h");
  EXPECT_TRUE(h_noise.is_error);
}

CUDAQ_TEST(KrausSelectionTest, TwoQubitNoise) {
  KrausSelection cx_noise(5,      // circuit_location
                          {0, 1}, // Both qubits affected
                          "cx",   // op_name
                          7,      // IX error
                          true    // is_error
  );

  EXPECT_EQ(cx_noise.qubits.size(), 2);
  EXPECT_EQ(cx_noise.qubits[0], 0);
  EXPECT_EQ(cx_noise.qubits[1], 1);
  EXPECT_EQ(cx_noise.kraus_operator_index, 7);
  EXPECT_TRUE(cx_noise.is_error);
}

CUDAQ_TEST(KrausSelectionTest, CopySemantics) {
  KrausSelection original(1, {0, 1}, "cx", 3, true);

  KrausSelection copied = original;
  EXPECT_TRUE(copied == original);
  EXPECT_EQ(copied.qubits[0], 0);

  KrausSelection assigned;
  assigned = original;
  EXPECT_TRUE(assigned == original);
}

CUDAQ_TEST(KrausSelectionTest, MoveSemantics) {
  KrausSelection original(1, {0, 1, 2}, "ccx", 5, true);

  KrausSelection moved = std::move(original);
  EXPECT_EQ(moved.circuit_location, 1);
  EXPECT_EQ(moved.qubits.size(), 3);
  EXPECT_EQ(moved.op_name, "ccx");
  EXPECT_TRUE(moved.is_error);
}

CUDAQ_TEST(KrausSelectionTest, ConstexprEquality) {
  KrausSelection sel1(0, {0}, "h", 1, true);
  KrausSelection sel2(0, {0}, "h", 1, true);
  KrausSelection sel3(1, {0}, "h", 1, true);

  EXPECT_TRUE(sel1 == sel2);
  EXPECT_FALSE(sel1 == sel3);
}
