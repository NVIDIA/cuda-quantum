/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/builder/kernel_builder.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/utils/cudaq_utils.h"

namespace cudaq {
using SingleIndices = std::vector<std::size_t>;
using DoubleIndices = std::vector<std::size_t>;
using Excitations =
    std::tuple<std::vector<SingleIndices>, std::vector<DoubleIndices>>;

Excitations generateExcitations(std::size_t nElectrons, std::size_t nOrbitals) {
  std::vector<double> sz(nOrbitals);
  for (auto &i : cudaq::range(nOrbitals))
    sz[i] = (i % 2 == 0) ? 0.5 : -0.5;

  std::vector<SingleIndices> singles;
  for (auto &r : cudaq::range(nElectrons))
    for (std::size_t p = nElectrons; p < nOrbitals; p++) {
      if (sz[p] - sz[r] == 0)
        singles.emplace_back(SingleIndices{r, p});
    }

  std::vector<DoubleIndices> doubles;
  for (auto &s : cudaq::range(nElectrons - 1))
    for (std::size_t r = s + 1; r < nElectrons; r++)
      for (std::size_t q = nElectrons; q < nOrbitals - 1; q++)
        for (std::size_t p = q + 1; p < nOrbitals; p++)
          if (sz[p] + sz[q] - sz[r] - sz[s] == 0)
            doubles.emplace_back(DoubleIndices{s, r, q, p});

  return std::make_tuple(singles, doubles);
}

template <typename KernelBuilder>
void singletExcitation(KernelBuilder &&kernel, QuakeValue &qubits,
                       QuakeValue &theta, const SingleIndices &indices) {
  auto r = indices.front();
  auto p = indices.back();
  kernel.exp_pauli(0.5 * theta, "YX", qubits[r], qubits[p]);
  kernel.exp_pauli(-0.5 * theta, "XY", qubits[r], qubits[p]);
}

__qpu__ void singletExcitation(cudaq::qspan<> qubits, double theta,
                               const SingleIndices &indices) {
  auto r = indices.front();
  auto p = indices.back();
  exp_pauli(theta / 2., "YX", qubits[r], qubits[p]);
  exp_pauli(-theta / 2., "XY", qubits[r], qubits[p]);
}

template <typename KernelBuilder>
void doubletExcitation(KernelBuilder &kernel, QuakeValue &qubits,
                       QuakeValue &theta, const DoubleIndices &d1,
                       const DoubleIndices &d2) {
  auto s = d1.front();
  auto r = d1.back();
  auto q = d2.front();
  auto p = d2.back();

  // layer 1
  kernel.exp_pauli((1. / 8.) * theta, "XXYX", qubits[s], qubits[r], qubits[q],
                   qubits[p]);
  // layer 2
  kernel.exp_pauli((1. / 8.) * theta, "YXYY", qubits[s], qubits[r], qubits[q],
                   qubits[p]);
  // layer 3
  kernel.exp_pauli((1. / 8.) * theta, "XYYY", qubits[s], qubits[r], qubits[q],
                   qubits[p]);
  // layer 4
  kernel.exp_pauli((1. / 8.) * theta, "XXXY", qubits[s], qubits[r], qubits[q],
                   qubits[p]);
  // layer 5
  kernel.exp_pauli((-1. / 8.) * theta, "YXXX", qubits[s], qubits[r], qubits[q],
                   qubits[p]);
  // layer 6
  kernel.exp_pauli((-1. / 8.) * theta, "XYXX", qubits[s], qubits[r], qubits[q],
                   qubits[p]);
  // layer 7
  kernel.exp_pauli((-1. / 8.) * theta, "YYYX", qubits[s], qubits[r], qubits[q],
                   qubits[p]);
  // layer 8
  kernel.exp_pauli((-1. / 8.) * theta, "YYXY", qubits[s], qubits[r], qubits[q],
                   qubits[p]);
}

__qpu__ void doubletExcitation(cudaq::qspan<> qubits, double theta,
                               const DoubleIndices &d1,
                               const DoubleIndices &d2) {
  auto s = d1.front();
  auto r = d1.back();
  auto q = d2.front();
  auto p = d2.back();

  // layer 1
  exp_pauli((1. / 8.) * theta, "XXYX", qubits[s], qubits[r], qubits[q],
            qubits[p]);
  // layer 2
  exp_pauli((1. / 8.) * theta, "YXYY", qubits[s], qubits[r], qubits[q],
            qubits[p]);
  // layer 3
  exp_pauli((1. / 8.) * theta, "XYYY", qubits[s], qubits[r], qubits[q],
            qubits[p]);
  // layer 4
  exp_pauli((1. / 8.) * theta, "XXXY", qubits[s], qubits[r], qubits[q],
            qubits[p]);
  // layer 5
  exp_pauli((-1. / 8.) * theta, "YXXX", qubits[s], qubits[r], qubits[q],
            qubits[p]);
  // layer 6
  exp_pauli((-1. / 8.) * theta, "XYXX", qubits[s], qubits[r], qubits[q],
            qubits[p]);
  // layer 7
  exp_pauli((-1. / 8.) * theta, "YYYX", qubits[s], qubits[r], qubits[q],
            qubits[p]);
  // layer 8
  exp_pauli((-1. / 8.) * theta, "YYXY", qubits[s], qubits[r], qubits[q],
            qubits[p]);
}

std::size_t uccsd_num_parameters(std::size_t nElectrons, std::size_t nQubits) {
  auto [singles, doubles] = generateExcitations(nElectrons, nQubits);
  return singles.size() + doubles.size();
}

/// @brief Generate the unitary coupled cluster singlet doublet ansatz on the
/// given number of qubits and electrons. This function creates the ansatz on an
/// existing kernel_builder instance. It takes a vector of rotation
/// parameters as input as a QuakeValue.
template <typename KernelBuilder>
void uccsd(KernelBuilder &kernel, QuakeValue &qubits, QuakeValue &thetas,
           std::size_t nElectrons, std::size_t nOrbitals) {
  auto [singles, doubles] = generateExcitations(nElectrons, nOrbitals);

  // doubles
  std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>
      double_processed;
  for (auto &el : doubles) {
    auto s = el[0];
    auto r = el[1];
    auto q = el[2];
    auto p = el[3];
    std::vector<std::size_t> d1, d2;
    for (std::size_t i = s; i < r + 1; i++)
      d1.emplace_back(i);
    for (std::size_t i = q; i < p + 1; i++)
      d2.emplace_back(i);
    double_processed.emplace_back(d1, d2);
  }

  for (std::size_t c = singles.size(); auto &el : double_processed) {
    auto t = thetas[c++];
    doubletExcitation(kernel, qubits, t, el.first, el.second);
  }

  // singles
  for (std::size_t i = 0; auto &single : singles) {
    auto t = thetas[i++];
    singletExcitation(kernel, qubits, t, single);
  }
}

/// @brief Generate the unitary coupled cluster singlet doublet ansatz on the
/// given number of qubits and electrons. Takes a vector of rotation
/// parameters as input, the size of which must correspond to the output of
/// the `uccsd_num_parameters` function.
__qpu__ void uccsd(cudaq::qspan<> qubits, std::vector<double> thetas,
                   std::size_t nElectrons) {
  auto [singles, doubles] = generateExcitations(nElectrons, qubits.size());

  // doubles
  std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>
      double_processed;
  for (auto &el : doubles) {
    auto s = el[0];
    auto r = el[1];
    auto q = el[2];
    auto p = el[3];
    std::vector<std::size_t> d1, d2;
    for (std::size_t i = s; i < r + 1; i++)
      d1.emplace_back(i);
    for (std::size_t i = q; i < p + 1; i++)
      d2.emplace_back(i);
    double_processed.emplace_back(d1, d2);
  }

  for (std::size_t c = singles.size(); auto &el : double_processed)
    doubletExcitation(qubits, thetas[c++], el.first, el.second);

  // singles
  for (std::size_t i = 0; auto &single : singles)
    singletExcitation(qubits, thetas[i++], single);
}

} // namespace cudaq
