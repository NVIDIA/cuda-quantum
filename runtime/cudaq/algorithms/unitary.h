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
#include "cudaq/operators/matrix.h"
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

/// @brief Apply an m-qubit gate in place to an n-qubit unitary.
/// @param U The 2^n x 2^n unitary, modified in place.
/// @param gate The 2^m x 2^m gate matrix.
/// @param num_qubits The total number of qubits in the full system.
/// @param qubit_indices A vector of size m giving the qubit indices the gate
/// acts on in the full system.
inline void apply_gate_in_place(complex_matrix &U, const complex_matrix &gate,
                                std::size_t num_qubits,
                                const std::vector<std::size_t> &qubit_indices) {
  using value_type = std::complex<double>;
  const std::size_t m = qubit_indices.size();
  const std::size_t gdim = 1ULL << m;
  const std::size_t dim = 1ULL << num_qubits;

  // Bit position of each affected qubit in the system index.
  std::vector<std::size_t> bp(m);
  std::size_t affected_mask = 0;
  for (std::size_t k = 0; k < m; ++k) {
    bp[k] = num_qubits - 1 - qubit_indices[k];
    affected_mask |= (1ULL << bp[k]);
  }

  // Local row-major copy of the gate so the inner loop is a tight indexed
  // multiply-add without going through complex_matrix accessors.
  std::vector<value_type> g(gdim * gdim);
  for (std::size_t a = 0; a < gdim; ++a)
    for (std::size_t b = 0; b < gdim; ++b)
      g[a * gdim + b] = gate(a, b);

  // scatter[a] deposits the m bits of a into the bp[] positions. Bit (m-1-k)
  // of a is the k-th qubit's value (MSB-first within the gate, matching the
  // gate's row/column encoding).
  std::vector<std::size_t> scatter(gdim, 0);
  for (std::size_t a = 0; a < gdim; ++a) {
    std::size_t s = 0;
    for (std::size_t k = 0; k < m; ++k)
      if ((a >> (m - 1 - k)) & 1ULL)
        s |= (1ULL << bp[k]);
    scatter[a] = s;
  }

  value_type *udata = U.get_data(complex_matrix::order::row_major);

  std::vector<value_type> col_old(gdim);
  std::vector<value_type> col_new(gdim);
  std::vector<std::size_t> rows(gdim);

  // Iterate over all "base" row indices that have zero in every affected bit.
  // For each base, the 2^m affected rows are base | scatter[a].
  for (std::size_t base = 0; base < dim; ++base) {
    if (base & affected_mask)
      continue;
    for (std::size_t a = 0; a < gdim; ++a)
      rows[a] = base | scatter[a];

    for (std::size_t j = 0; j < dim; ++j) {
      for (std::size_t b = 0; b < gdim; ++b)
        col_old[b] = udata[rows[b] * dim + j];
      for (std::size_t a = 0; a < gdim; ++a) {
        value_type acc(0.0, 0.0);
        for (std::size_t b = 0; b < gdim; ++b)
          acc += g[a * gdim + b] * col_old[b];
        col_new[a] = acc;
      }
      for (std::size_t a = 0; a < gdim; ++a)
        udata[rows[a] * dim + j] = col_new[a];
    }
  }
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
    if (!inst.controls.empty())
      gate = make_controlled_unitary(gate, inst.controls.size());

    // Get vector of all qubit indices that this gate operates on.
    // The control qubits are expected to be at the start of the vector.
    std::vector<std::size_t> inst_qubits;
    inst_qubits.reserve(inst.controls.size() + inst.targets.size());
    for (const auto &control : inst.controls)
      inst_qubits.push_back(control.id);
    for (const auto &target : inst.targets)
      inst_qubits.push_back(target.id);

    apply_gate_in_place(U, gate, num_qubits, inst_qubits);
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
