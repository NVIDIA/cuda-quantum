/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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

namespace cudaq::details {

inline complex_matrix
expand_gate_to_system(const complex_matrix &gate, std::size_t num_qudits,
                      const std::vector<std::size_t> &qudit_indices) {
  // Build a list of matrices for each qubit: gate if one of the participating
  // qudits, identity otherwise.
  std::vector<complex_matrix> factors;
  // std::size_t gate_idx = 0;
  for (std::size_t q = 0; q < num_qudits; ++q) {
    if (std::find(qudit_indices.begin(), qudit_indices.end(), q) !=
        qudit_indices.end()) {
      factors.push_back(gate);
    } else {
      factors.push_back(complex_matrix::identity(2));
    }
  }
  // Kronecker product of all factors
  auto full_unitary = kronecker(factors.begin(), factors.end());
  // TODO: swap
  return full_unitary;
}

inline complex_matrix unitary_from_trace(const Trace &trace) {
  auto num_qubits = trace.getNumQudits();
  std::size_t dim = 1ULL << num_qubits;
  complex_matrix U = complex_matrix::identity(dim);

  for (const auto &inst : trace) {
    auto gate_name = nvqir::getGateNameFromString(inst.name);
    auto gate_vec = nvqir::getGateByName<double>(gate_name, inst.params);
    std::size_t gate_dim = static_cast<std::size_t>(std::sqrt(gate_vec.size()));
    complex_matrix gate(gate_vec, {gate_dim, gate_dim});

    // Get vector of all qubit indices that this gate operates on
    std::vector<std::size_t> inst_qubits;
    inst_qubits.reserve(inst.targets.size());
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

template <typename QuantumKernel, typename... Args>
complex_matrix get_unitary_cmat(QuantumKernel &&kernel, Args &&...args) {
  auto trace = traceFromKernel(kernel, std::forward<Args>(args)...);
  return unitary_from_trace(trace);
}

template <typename QuantumKernel, typename... Args>
std::vector<std::complex<double>> get_unitary(QuantumKernel &&kernel,
                                              Args &&...args) {
  auto U = get_unitary_cmat(std::forward<QuantumKernel>(kernel),
                            std::forward<Args>(args)...);
  // Flatten to row-major std::vector
  std::vector<std::complex<double>> result;
  result.reserve(U.rows() * U.cols());
  for (std::size_t i = 0; i < U.rows(); ++i)
    for (std::size_t j = 0; j < U.cols(); ++j)
      result.push_back(U(i, j));
  return result;
}

} // namespace cudaq::details
