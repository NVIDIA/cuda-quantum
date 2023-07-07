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

  kernel.rx(-M_PI_2, qubits[r]);
  kernel.h(qubits[p]);

  std::vector<std::pair<std::size_t, std::size_t>> cnots;

  for (std::size_t i = r; i < p; i++)
    cnots.emplace_back(i, i + 1);

  auto reversed = cnots;
  std::reverse(reversed.begin(), reversed.end());

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz(0.5 * theta, qubits[p]);

  for (auto &[i, j] : reversed)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rx(M_PI_2, qubits[r]);
  kernel.h(qubits[p]);

  kernel.h(qubits[r]);
  kernel.rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz(-0.5 * theta, qubits[p]);

  for (auto &[i, j] : reversed)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.h(qubits[r]);
  kernel.rx(M_PI_2, qubits[p]);
}

__qpu__ void singletExcitation(cudaq::qspan<> qubits, double theta,
                               const SingleIndices &indices) {
  auto r = indices.front();
  auto p = indices.back();

  rx(-M_PI_2, qubits[r]);
  h(qubits[p]);

  std::vector<std::pair<std::size_t, std::size_t>> cnots;

  for (std::size_t i = r; i < p; i++)
    cnots.emplace_back(i, i + 1);

  auto reversed = cnots;
  std::reverse(reversed.begin(), reversed.end());

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(theta / 2., qubits[p]);

  for (auto &[i, j] : reversed)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rx(M_PI_2, qubits[r]);
  h(qubits[p]);

  h(qubits[r]);
  rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(-theta / 2., qubits[p]);

  for (auto &[i, j] : reversed)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  h(qubits[r]);
  rx(M_PI_2, qubits[p]);
}

template <typename KernelBuilder>
void doubletExcitation(KernelBuilder &kernel, QuakeValue &qubits,
                       QuakeValue &theta, const DoubleIndices &d1,
                       const DoubleIndices &d2) {
  auto s = d1.front();
  auto r = d1.back();
  auto q = d2.front();
  auto p = d2.back();

  std::vector<std::pair<std::size_t, std::size_t>> cnots;
  for (auto &i : cudaq::range(d1.size() - 1))
    cnots.emplace_back(d1[i], d1[i + 1]);

  cnots.emplace_back(r, q);

  for (auto &i : cudaq::range(d2.size() - 1))
    cnots.emplace_back(d2[i], d2[i + 1]);

  auto reversed_cnots = cnots;
  std::reverse(reversed_cnots.begin(), reversed_cnots.end());

  kernel.h(qubits[s]);
  kernel.h(qubits[r]);
  kernel.rx(-M_PI_2, qubits[q]);
  kernel.h(qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz((1. / 8.0) * theta, qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.h(qubits[s]);
  kernel.h(qubits[r]);
  kernel.rx(M_PI_2, qubits[q]);
  kernel.h(qubits[p]);

  // layer 2
  kernel.rx(-M_PI_2, qubits[s]);
  kernel.h(qubits[r]);
  kernel.rx(-M_PI_2, qubits[q]);
  kernel.rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz((1. / 8.0) * theta, qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rx(M_PI_2, qubits[s]);
  kernel.h(qubits[r]);
  kernel.rx(M_PI_2, qubits[q]);
  kernel.rx(M_PI_2, qubits[p]);

  // layer 3
  kernel.h(qubits[s]);
  kernel.rx(-M_PI_2, qubits[r]);
  kernel.rx(-M_PI_2, qubits[q]);
  kernel.rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz((1. / 8.0) * theta, qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.h(qubits[s]);
  kernel.rx(M_PI_2, qubits[r]);
  kernel.rx(M_PI_2, qubits[q]);
  kernel.rx(M_PI_2, qubits[p]);

  // layer 4
  kernel.h(qubits[s]);
  kernel.h(qubits[r]);
  kernel.h(qubits[q]);
  kernel.rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz((1. / 8.0) * theta, qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.h(qubits[s]);
  kernel.h(qubits[r]);
  kernel.h(qubits[q]);
  kernel.rx(M_PI_2, qubits[p]);

  // layer 5
  kernel.rx(-M_PI_2, qubits[s]);
  kernel.h(qubits[r]);
  kernel.h(qubits[q]);
  kernel.h(qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz((-1. / 8.0) * theta, qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rx(M_PI_2, qubits[s]);
  kernel.h(qubits[r]);
  kernel.h(qubits[q]);
  kernel.h(qubits[p]);

  // layer 6
  kernel.h(qubits[s]);
  kernel.rx(-M_PI_2, qubits[r]);
  kernel.h(qubits[q]);
  kernel.h(qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz((-1. / 8.0) * theta, qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.h(qubits[s]);
  kernel.rx(M_PI_2, qubits[r]);
  kernel.h(qubits[q]);
  kernel.h(qubits[p]);

  // layer 7
  kernel.rx(-M_PI_2, qubits[s]);
  kernel.rx(-M_PI_2, qubits[r]);
  kernel.rx(-M_PI_2, qubits[q]);
  kernel.h(qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz((-1. / 8.0) * theta, qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rx(M_PI_2, qubits[s]);
  kernel.rx(M_PI_2, qubits[r]);
  kernel.rx(M_PI_2, qubits[q]);
  kernel.h(qubits[p]);

  // layer 8
  kernel.rx(-M_PI_2, qubits[s]);
  kernel.rx(-M_PI_2, qubits[r]);
  kernel.h(qubits[q]);
  kernel.rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rz((-1. / 8.0) * theta, qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[j]);

  kernel.rx(M_PI_2, qubits[s]);
  kernel.rx(M_PI_2, qubits[r]);
  kernel.h(qubits[q]);
  kernel.rx(M_PI_2, qubits[p]);
}

__qpu__ void doubletExcitation(cudaq::qspan<> qubits, double theta,
                               const DoubleIndices &d1,
                               const DoubleIndices &d2) {
  auto s = d1.front();
  auto r = d1.back();
  auto q = d2.front();
  auto p = d2.back();

  std::vector<std::pair<std::size_t, std::size_t>> cnots;
  for (auto &i : cudaq::range(d1.size() - 1))
    cnots.emplace_back(d1[i], d1[i + 1]);

  cnots.emplace_back(r, q);

  for (auto &i : cudaq::range(d2.size() - 1))
    cnots.emplace_back(d2[i], d2[i + 1]);

  auto reversed_cnots = cnots;
  std::reverse(reversed_cnots.begin(), reversed_cnots.end());

  h(qubits[s]);
  h(qubits[r]);
  rx(-M_PI_2, qubits[q]);
  h(qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(theta / 8., qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  h(qubits[s]);
  h(qubits[r]);
  rx(M_PI_2, qubits[q]);
  h(qubits[p]);

  // layer 2
  rx(-M_PI_2, qubits[s]);
  h(qubits[r]);
  rx(-M_PI_2, qubits[q]);
  rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(theta / 8., qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rx(M_PI_2, qubits[s]);
  h(qubits[r]);
  rx(M_PI_2, qubits[q]);
  rx(M_PI_2, qubits[p]);

  // layer 3
  h(qubits[s]);
  rx(-M_PI_2, qubits[r]);
  rx(-M_PI_2, qubits[q]);
  rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(theta / 8., qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  h(qubits[s]);
  rx(M_PI_2, qubits[r]);
  rx(M_PI_2, qubits[q]);
  rx(M_PI_2, qubits[p]);

  // layer 4
  h(qubits[s]);
  h(qubits[r]);
  h(qubits[q]);
  rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(theta / 8., qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  h(qubits[s]);
  h(qubits[r]);
  h(qubits[q]);
  rx(M_PI_2, qubits[p]);

  // layer 5
  rx(-M_PI_2, qubits[s]);
  h(qubits[r]);
  h(qubits[q]);
  h(qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(-theta / 8., qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rx(M_PI_2, qubits[s]);
  h(qubits[r]);
  h(qubits[q]);
  h(qubits[p]);

  // layer 6
  h(qubits[s]);
  rx(-M_PI_2, qubits[r]);
  h(qubits[q]);
  h(qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(-theta / 8., qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  h(qubits[s]);
  rx(M_PI_2, qubits[r]);
  h(qubits[q]);
  h(qubits[p]);

  // layer 7
  rx(-M_PI_2, qubits[s]);
  rx(-M_PI_2, qubits[r]);
  rx(-M_PI_2, qubits[q]);
  h(qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(-theta / 8., qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rx(M_PI_2, qubits[s]);
  rx(M_PI_2, qubits[r]);
  rx(M_PI_2, qubits[q]);
  h(qubits[p]);

  // layer 8
  rx(-M_PI_2, qubits[s]);
  rx(-M_PI_2, qubits[r]);
  h(qubits[q]);
  rx(-M_PI_2, qubits[p]);

  for (auto &[i, j] : cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rz(-theta / 8., qubits[p]);

  for (auto &[i, j] : reversed_cnots)
    x<cudaq::ctrl>(qubits[i], qubits[j]);

  rx(M_PI_2, qubits[s]);
  rx(M_PI_2, qubits[r]);
  h(qubits[q]);
  rx(M_PI_2, qubits[p]);
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
