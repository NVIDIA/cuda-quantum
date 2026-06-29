/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

#include <cmath>
#include <iostream>
#include <vector>

// IMPORTANT: loop-count annotations are part of this test contract. Kernel
// names ending in `_expected_N_loop(s)` tell the quake fake mock server how
// many `cc.loop` operations must be present in the submitted client payload
// before the server does its own execution-time full unroll.

// Expected remaining payload loops: none. Broadcast X over the whole qvector
// must lower to static wire operations in the submitted payload.
__qpu__ int broadcast_x_qvector_expected_0_loops() {
  cudaq::qvector q(3);
  x(q);

  int result = 0;
  auto bits = mz(q);
  if (bits[0])
    result |= 1;
  if (bits[1])
    result |= 2;
  if (bits[2])
    result |= 4;
  return result;
}

// Expected remaining payload loops: none. The whole-register control must be
// expanded before wire assignment so all control wires are static.
__qpu__ int whole_register_control_x_expected_0_loops() {
  cudaq::qvector ctrls(3);
  cudaq::qubit target;

  x(ctrls);
  x<cudaq::ctrl>(ctrls, target);
  return mz(target);
}

// Expected remaining payload loop: the outer pass loop. The loops that index
// quantum data (`q[i]`) are unrolled so wire IDs are static in the payload.
__qpu__ int loop_payload_stress_expected_1_loop() {
  constexpr int width = 4;
  cudaq::qvector q(width);

  // KEEP: this classical repeat loop does not index quantum data directly.
  for (int pass = 0; pass < 2; ++pass)
    // UNROLL: this loop indexes `q[i]`, so wire IDs must become static.
    for (int i = 0; i < width; ++i)
      x(q[i]);

  int i = 0;
  // UNROLL: this loop indexes `q[i]`, so wire IDs must become static.
  while (i < width) {
    x(q[i]);
    ++i;
  }

  int result = 0;
  if (mz(q[0]))
    result |= 1;
  if (mz(q[1]))
    result |= 2;
  if (mz(q[2]))
    result |= 4;
  if (mz(q[3]))
    result |= 8;

  return result;
}

// Expected remaining payload loop: the loop over measurement results. This loop
// does not access quantum data, so it should stay rolled.
__qpu__ int measurement_result_loop_expected_1_loop() {
  constexpr int width = 4;
  cudaq::qvector q(width);

  x(q[0]);
  x(q[2]);

  auto bits = mz(q);
  int result = 0;
  // KEEP: this loop only indexes measurement results, not quantum data.
  for (int i = 0; i < width; ++i)
    if (bits[i])
      result |= 1 << i;

  return result;
}

// Expected remaining payload loops: outer, group, classical accumulation, and
// measurement-result accumulation. The quantum-data loop over `q[wire]` is
// unrolled for static wire IDs.
__qpu__ int nested_mixed_loop_payload_expected_4_loops() {
  constexpr int width = 4;
  cudaq::qvector q(width);

  int classical = 0;
  // KEEP: this classical outer loop does not index quantum data directly.
  for (int outer = 0; outer < 1; ++outer)
    // KEEP: this classical group loop does not index quantum data directly.
    for (int group = 0; group < 2; ++group) {
      // UNROLL: this loop indexes `q[wire]`, so wire IDs must become static.
      for (int wire = 0; wire < width; ++wire)
        x(q[wire]);

      // KEEP: this loop is classical accumulation only.
      for (int i = 0; i < 3; ++i)
        classical += group + i;
    }

  x(q[0]);

  auto bits = mz(q);
  int result = classical;
  // KEEP: this loop only iterates over measurement results.
  for (auto bit : bits)
    if (bit)
      result += 16;

  return result;
}

// Expected remaining payload loop: the phase-kickback repeat loop. This is the
// fixed-wire core of phase estimation: repeated controlled unitary powers on a
// prepared eigenstate. The loop should stay rolled because it uses fixed
// qubits.
__qpu__ int fixed_wire_phase_estimation_expected_1_loop() {
  cudaq::qubit counting, eigenstate;

  x(eigenstate);
  h(counting);

  // KEEP: repeated controlled rotations over fixed qubits should stay rolled.
  for (int power = 0; power < 3; ++power)
    r1<cudaq::ctrl>(M_PI, counting, eigenstate);

  h(counting);
  return mz(counting);
}

// Expected remaining payload loops: none. The triangular inner bound depends on
// the outer induction variable, and both loops index quantum data. Both loops
// must unroll so the client payload contains only static wire operations.
__qpu__ int parent_dependent_triangular_expected_0_loops() {
  constexpr int width = 4;
  cudaq::qvector q(width);

  x(q[0]);

  // UNROLL: `i` indexes quantum data and bounds the inner loop.
  for (int i = 0; i < width - 1; ++i)
    // UNROLL: `j` indexes quantum data with a parent-dependent lower bound.
    for (int j = i + 1; j < width; ++j)
      x<cudaq::ctrl>(q[i], q[j]);

  int result = 0;
  if (mz(q[0]))
    result |= 1;
  if (mz(q[1]))
    result |= 2;
  if (mz(q[2]))
    result |= 4;
  if (mz(q[3]))
    result |= 8;

  return result;
}

// Expected remaining payload loop: the outer repeat loop. The inner triangular
// parent-dependent quantum loops must unroll, but the outer repeat does not
// index quantum data directly and should stay rolled.
__qpu__ int repeated_parent_dependent_triangular_expected_1_loop() {
  constexpr int width = 4;
  cudaq::qvector q(width);

  x(q[0]);

  // KEEP: this repeat loop does not index quantum data directly.
  for (int pass = 0; pass < 2; ++pass)
    // UNROLL: `i` indexes quantum data and bounds the inner loop.
    for (int i = 0; i < width - 1; ++i)
      // UNROLL: `j` indexes quantum data with a parent-dependent lower bound.
      for (int j = i + 1; j < width; ++j)
        x<cudaq::ctrl>(q[i], q[j]);

  int result = 0;
  if (mz(q[0]))
    result |= 1;
  if (mz(q[1]))
    result |= 2;
  if (mz(q[2]))
    result |= 4;
  if (mz(q[3]))
    result |= 8;

  return result;
}

// Expected remaining payload loops: none. The middle loop does not directly
// index quantum data, but the inner loop indexes q[outer]. Unrolling must
// propagate through the non-blocking middle loop so every quantum access has a
// static wire ID in the submitted payload.
__qpu__ int transitive_parent_index_access_expected_0_loops() {
  constexpr int width = 3;
  cudaq::qvector q(width);

  // UNROLL: `outer` is used by the innermost quantum access q[outer].
  for (int outer = 0; outer < width; ++outer)
    // UNROLL: this loop is not directly blocking, but it encloses the
    // parent-dependent quantum access exposed by unrolling the inner loop.
    for (int middle = 0; middle < 3; ++middle)
      // UNROLL: this loop contains the q[outer] access that initially blocks
      // wire conversion.
      for (int inner = 0; inner < 3; ++inner)
        x(q[outer]);

  int result = 0;
  if (mz(q[0]))
    result |= 1;
  if (mz(q[1]))
    result |= 2;
  if (mz(q[2]))
    result |= 4;

  return result;
}

static bool checkResults(const std::vector<int> &results, int expected,
                         const char *name) {
  for (auto result : results) {
    if (result != expected) {
      std::cerr << name << " expected " << expected << ", got " << result
                << "\n";
      return false;
    }
  }
  return true;
}

int main() {
  if (!checkResults(cudaq::run(3, broadcast_x_qvector_expected_0_loops), 7,
                    "broadcast_x_qvector_expected_0_loops"))
    return 1;

  if (!checkResults(cudaq::run(3, whole_register_control_x_expected_0_loops), 1,
                    "whole_register_control_x_expected_0_loops"))
    return 1;

  if (!checkResults(cudaq::run(3, loop_payload_stress_expected_1_loop), 15,
                    "loop_payload_stress_expected_1_loop"))
    return 1;

  if (!checkResults(cudaq::run(3, measurement_result_loop_expected_1_loop), 5,
                    "measurement_result_loop_expected_1_loop"))
    return 1;

  if (!checkResults(cudaq::run(3, nested_mixed_loop_payload_expected_4_loops),
                    25, "nested_mixed_loop_payload_expected_4_loops"))
    return 1;

  if (!checkResults(cudaq::run(3, fixed_wire_phase_estimation_expected_1_loop),
                    1, "fixed_wire_phase_estimation_expected_1_loop"))
    return 1;

  if (!checkResults(cudaq::run(3, parent_dependent_triangular_expected_0_loops),
                    3, "parent_dependent_triangular_expected_0_loops"))
    return 1;

  if (!checkResults(
          cudaq::run(3, repeated_parent_dependent_triangular_expected_1_loop),
          5, "repeated_parent_dependent_triangular_expected_1_loop"))
    return 1;

  if (!checkResults(
          cudaq::run(3, transitive_parent_index_access_expected_0_loops), 7,
          "transitive_parent_index_access_expected_0_loops"))
    return 1;

  std::cout << "Loop payload stress passed.\n";
  return 0;
}
