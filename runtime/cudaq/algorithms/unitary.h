/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Trace.h"
#include "cudaq/algorithms/draw.h"
#include "cudaq/utils/matrix.h"
#include "nvqir/Gates.h"
#include <iostream>

namespace cudaq::contrib {

/// @brief Build a controlled-gate unitary: identity in all control blocks
/// except the “all controls = 1” block, which is replaced by the original gate.
/// @param gate The base unitary matrix of the target gate.
/// @param num_controls The number of control qubits.
/// @returns A complex_matrix representing the controlled version of the gate.
inline complex_matrix make_controlled_unitary(const complex_matrix &gate,
                                              std::size_t num_controls) {
  auto gdim = gate.rows();
  auto ctrl_dim = 1ULL << num_controls;
  auto new_dim = ctrl_dim * gdim;
  // Start with identity of full size
  complex_matrix M = complex_matrix::identity(new_dim);
  // Offset of the “all controls = 1” block
  auto offset = (ctrl_dim - 1) * gdim;
  // Overwrite bottom-right block with gate
  for (std::size_t i = 0; i < gdim; ++i)
    for (std::size_t j = 0; j < gdim; ++j)
      M(offset + i, offset + j) = gate(i, j);
  return M;
}

/// @brief Expand an m-qubit gate to an n-qubit system by tensor product with
/// identities and permuting qubits into the specified positions.
/// @param gate The unitary matrix acting on m qubits.
/// @param num_qudits The total number of qubits in the full system.
/// @param qudit_indices A vector of size m giving the target qubit indices in
/// the full system.
/// @returns A complex_matrix representing the expanded gate on the full n-qubit
/// system.
inline complex_matrix
expand_gate_to_system(const complex_matrix &gate, std::size_t num_qudits,
                      const std::vector<std::size_t> &qudit_indices) {
  // number of qubits this gate acts on
  std::size_t m = qudit_indices.size();

  // 1) Build U0 = gate ⊗ I ⊗ ... ⊗ I, with gate on qubits [0..m-1]
  std::vector<complex_matrix> facs;
  facs.reserve(1 + num_qudits - m);
  facs.push_back(gate);
  for (std::size_t k = m; k < num_qudits; ++k)
    facs.push_back(complex_matrix::identity(2));
  auto U0 = kronecker(facs.begin(), facs.end());

  // 2) Build permutation P that moves qubits [0..m−1] → qudit_indices[]
  std::size_t dim = 1ULL << num_qudits;
  // 2a) build a map old_index → new_index
  std::vector<std::size_t> perm(num_qudits);
  for (std::size_t k = 0; k < m; ++k)
    perm[k] = qudit_indices[k];
  // mark used targets
  std::vector<bool> used(num_qudits, false);
  for (auto idx : qudit_indices)
    used[idx] = true;
  // collect the free new slots
  std::vector<std::size_t> free_pos;
  free_pos.reserve(num_qudits - m);
  for (std::size_t i = 0; i < num_qudits; ++i)
    if (!used[i])
      free_pos.push_back(i);
  // assign the remaining old qubits (m..n−1) into those slots
  for (std::size_t k = m; k < num_qudits; ++k)
    perm[k] = free_pos[k - m];

  // 2b) build P by permuting computational‐basis indices (fixing endian)
  complex_matrix P(dim, dim);
  // Interpret bit-0 of the index as the MSB, build P accordingly.
  for (std::size_t col = 0; col < dim; ++col) {
    std::size_t row = 0;
    // i = 0 → MSB, i = num_qudits-1 → LSB
    for (std::size_t i = 0; i < num_qudits; ++i) {
      auto srcBitPos = num_qudits - 1 - i;
      if ((col >> srcBitPos) & 1ULL) {
        auto dstBitPos = num_qudits - 1 - perm[i];
        row |= (1ULL << dstBitPos);
      }
    }
    P(row, col) = 1.0;
  }

  // 3) Embed: full_gate = P * U0 * P⁻¹
  auto result = P * U0 * P.adjoint();
  return result;
}

/// @brief Construct the full system unitary from a Trace of quantum
/// instructions.
/// @param trace The Trace object recording the sequence of quantum operations.
/// @returns A complex_matrix representing the overall unitary of the traced
/// kernel.
inline complex_matrix unitary_from_trace(const Trace &trace) {
  auto num_qubits = trace.getNumQudits();
  std::size_t dim = 1ULL << num_qubits;
  complex_matrix U = complex_matrix::identity(dim);

  for (const auto &inst : trace) {
    auto gate_name = nvqir::getGateNameFromString(inst.name);
    auto gate_vec = nvqir::getGateByName<double>(gate_name, inst.params);
    std::size_t gate_dim = static_cast<std::size_t>(std::sqrt(gate_vec.size()));
    complex_matrix gate(gate_vec, {gate_dim, gate_dim});

    // If there are control qubits, build the controlled-unitary
    if (!inst.controls.empty()) {
      gate = make_controlled_unitary(gate, inst.controls.size());
      gate_dim = gate.rows();
    }

    // Get vector of all qubit indices that this gate operates on
    // The control qubits are expected to be at the start of the vector,
    std::vector<std::size_t> inst_qubits;
    inst_qubits.reserve(inst.controls.size() + inst.targets.size());
    for (const auto &control : inst.controls) {
      inst_qubits.push_back(control.id);
    }
    for (const auto &target : inst.targets) {
      inst_qubits.push_back(target.id);
    }

    // Expand gate to full system
    auto full_gate = expand_gate_to_system(gate, num_qubits, inst_qubits);

    // Left-multiply the system unitary
    U = full_gate * U;
  }
  return U;
}

/// @brief Execute a quantum kernel and return its unitary as a complex_matrix.
/// @tparam QuantumKernel The functor type of the quantum kernel.
/// @param kernel The quantum kernel to execute.
/// @param args Arguments to pass to the kernel.
/// @returns The full system unitary as a complex_matrix.
template <typename QuantumKernel, typename... Args>
complex_matrix get_unitary_cmat(QuantumKernel &&kernel, Args &&...args) {
  auto trace = traceFromKernel(kernel, std::forward<Args>(args)...);
  return unitary_from_trace(trace);
}

} // namespace cudaq::contrib
