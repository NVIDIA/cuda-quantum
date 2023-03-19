/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once
#include "observe.h"
#include <cudaq.h>
#include <cudaq/spin_op.h>

namespace cudaq {
namespace __internal__ {

struct base_qite_ansatz {
  void operator()(const int N, const double step_size,
                  std::vector<cudaq::spin_op> &aOps) __qpu__ {
    cudaq::qreg q(N);
    for (auto &a : aOps) {
      for (std::size_t i = 0; i < a.n_terms(); i++) {
        auto term = a[i];
        if (!term.is_identity()) {
          exp(q, 0.5 * step_size, term);
        }
      }
    }
  }
};

std::vector<cudaq::spin_op> generatePauliPermutation(int in_nbQubits) {
  const int nbPermutations = std::pow(4, in_nbQubits);
  std::vector<cudaq::spin_op> opsList;
  opsList.reserve(nbPermutations);

  const std::vector<std::function<cudaq::spin_op(int)>> pauliOps{
      [](int i) { return cudaq::spin::x(i); },
      [](int i) { return cudaq::spin::y(i); },
      [](int i) { return cudaq::spin::z(i); }};
  const auto addQubitPauli = [&opsList, &pauliOps](int in_qubitIdx) {
    const auto currentOpListSize = opsList.size();
    for (std::size_t i = 0; i < currentOpListSize; ++i) {
      auto &currentOp = opsList[i];
      for (const auto &pauliOp : pauliOps) {
        const auto newOp = currentOp * pauliOp(in_qubitIdx);
        opsList.emplace_back(newOp);
      }
    }
  };

  opsList = {cudaq::spin_op(), cudaq::spin::x(0), cudaq::spin::y(0),
             cudaq::spin::z(0)};
  for (int i = 1; i < in_nbQubits; ++i) {
    addQubitPauli(i);
  }

  return opsList;
}

template <typename Kernel>
double qiteEvolve(cudaq::spin_op h, const double m_step_size,
                  std::vector<cudaq::spin_op> &aOps) {
  [[maybe_unused]] auto energy =
      cudaq::observe(Kernel{}, h, (int)h.n_qubits(), m_step_size, aOps);
  auto pauliOps = generatePauliPermutation(h.n_qubits());
  std::vector<double> sig_exps{1.0}; // 4**N of these
  for (std::size_t i = 1; i < pauliOps.size(); i++) {
    auto val = cudaq::observe(Kernel{}, pauliOps[i], (int)h.n_qubits(),
                              m_step_size, aOps);
    sig_exps.push_back(val);
  }

  return 0.0;
}

} // namespace __internal__

std::vector<double> qite(cudaq::spin_op h, const int steps,
                         const double step_size) {
  std::vector<cudaq::spin_op> aOps;
  std::vector<double> energies;
  for (int i = 0; i < steps; i++) {
    auto e = __internal__::qiteEvolve<__internal__::base_qite_ansatz>(
        h, step_size, aOps);
    energies.push_back(e);
  }
  return energies;
}
} // namespace cudaq
