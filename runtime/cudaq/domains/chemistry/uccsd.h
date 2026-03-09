/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

/// @brief An excitation_list is a vector of lists of indices.
using excitation_list = std::vector<std::vector<std::size_t>>;

/// @brief UCCSD excitations for single and double excitations.
struct excitations {
  excitation_list singles_alpha;
  excitation_list singles_beta;
  excitation_list doubles_mixed;
  excitation_list doubles_alpha;
  excitation_list doubles_beta;
};

/// @brief Given the number of electrons and qubits that make up the
/// system, return the single and double excitation indices.
excitations get_uccsd_excitations(std::size_t numElectrons,
                                  std::size_t numQubits) {
  std::make_signed_t<std::size_t> numSpatialOrbs = numQubits / 2;
  // check rounding
  std::make_signed_t<std::size_t> numOccupied = std::ceil(numElectrons / 2);
  auto numVirtual = numSpatialOrbs - numOccupied;
  excitation_list singlesAlpha, singlesBeta, doublesMixed, doublesAlpha,
      doublesBeta;
  std::vector<std::size_t> occupiedAlpha, virtualAlpha, occupiedBeta,
      virtualBeta;

  if (numElectrons % 2 != 0) {
    for (auto i : cudaq::range(numOccupied))
      occupiedAlpha.push_back(i * 2);

    for (auto i : cudaq::range(numVirtual))
      virtualAlpha.push_back(i * 2 + numElectrons + 1);

    for (auto i : cudaq::range(numOccupied - 1))
      occupiedBeta.push_back(i * 2 + 1);

    virtualBeta.push_back(2 * numOccupied - 1);
    for (auto i : cudaq::range(numVirtual))
      virtualBeta.push_back(i * 2 + numElectrons + 2);

  } else {
    for (auto i : cudaq::range(numOccupied))
      occupiedAlpha.push_back(i * 2);

    for (auto i : cudaq::range(numVirtual))
      virtualAlpha.push_back(i * 2 + numElectrons);

    for (auto i : cudaq::range(numOccupied))
      occupiedBeta.push_back(i * 2 + 1);

    for (auto i : cudaq::range(numVirtual))
      virtualBeta.push_back(i * 2 + numElectrons + 1);
  }

  for (auto p : occupiedAlpha)
    for (auto q : virtualAlpha)
      singlesAlpha.push_back({p, q});

  for (auto p : occupiedBeta)
    for (auto q : virtualBeta)
      singlesBeta.push_back({p, q});

  for (auto p : occupiedAlpha)
    for (auto q : occupiedBeta)
      for (auto r : virtualBeta)
        for (auto s : virtualAlpha)
          doublesMixed.push_back({p, q, r, s});

  std::make_signed_t<std::size_t> numOccAlpha = occupiedAlpha.size();
  std::make_signed_t<std::size_t> numOccBeta = occupiedBeta.size();
  std::make_signed_t<std::size_t> numVirtAlpha = virtualAlpha.size();
  std::make_signed_t<std::size_t> numVirtBeta = virtualBeta.size();

  for (auto p : cudaq::range(numOccAlpha - 1))
    for (auto q = p + 1; q < numOccAlpha; q++)
      for (auto r : cudaq::range(numVirtAlpha - 1))
        for (auto s = r + 1; s < numVirtAlpha; s++)
          doublesAlpha.push_back({occupiedAlpha[p], occupiedAlpha[q],
                                  virtualAlpha[r], virtualAlpha[s]});

  for (auto p : cudaq::range(numOccBeta - 1))
    for (auto q = p + 1; q < numOccBeta; q++)
      for (auto r : cudaq::range(numVirtBeta - 1))
        for (auto s = r + 1; s < numVirtBeta; s++)
          doublesBeta.push_back({occupiedBeta[p], occupiedBeta[q],
                                 virtualBeta[r], virtualBeta[s]});

  return excitations{singlesAlpha, singlesBeta, doublesMixed, doublesAlpha,
                     doublesBeta};
}

/// @brief Return the number of UCCSD ansatz parameters.
auto uccsd_num_parameters(std::size_t numElectrons, std::size_t numQubits) {
  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      get_uccsd_excitations(numElectrons, numQubits);
  return singlesAlpha.size() + singlesBeta.size() + doublesMixed.size() +
         doublesAlpha.size() + doublesBeta.size();
}

__qpu__ void singleExcitation(cudaq::qview<> qubits, std::size_t pOcc,
                              std::size_t qVirt, double theta) {
  // Y_p X_q
  rx(M_PI_2, qubits[pOcc]);
  h(qubits[qVirt]);

  for (std::size_t i = pOcc; i < qVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.5 * theta, qubits[qVirt]);

  for (std::size_t i = qVirt; i > pOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[qVirt]);
  rx(-M_PI_2, qubits[pOcc]);

  // -X_p Y_q
  h(qubits[pOcc]);
  rx(M_PI_2, qubits[qVirt]);

  for (std::size_t i = pOcc; i < qVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.5 * theta, qubits[qVirt]);

  for (std::size_t i = qVirt; i > pOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[qVirt]);
  h(qubits[pOcc]);
}

template <typename Kernel>
void singleExcitation(Kernel &kernel, QuakeValue &qubits, std::size_t pOcc,
                      std::size_t qVirt, QuakeValue &theta) {
  // Y_p X_q
  kernel.rx(M_PI_2, qubits[pOcc]);
  kernel.h(qubits[qVirt]);

  for (std::size_t i = pOcc; i < qVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.5 * theta, qubits[qVirt]);

  for (std::size_t i = qVirt; i > pOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.h(qubits[qVirt]);
  kernel.rx(-M_PI_2, qubits[pOcc]);

  // -X_p Y_q
  kernel.h(qubits[pOcc]);
  kernel.rx(M_PI_2, qubits[qVirt]);

  for (std::size_t i = pOcc; i < qVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.5 * theta, qubits[qVirt]);

  for (std::size_t i = qVirt; i > pOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.rx(-M_PI_2, qubits[qVirt]);
  kernel.h(qubits[pOcc]);
}

template <typename Kernel>
void doubleExcitation(Kernel &kernel, QuakeValue &qubits, std::size_t pOcc,
                      std::size_t qOcc, std::size_t rVirt, std::size_t sVirt,
                      QuakeValue &theta) {
  std::size_t iOcc = 0, jOcc = 0, aVirt = 0, bVirt = 0;
  double multiplier = 1.;
  if ((pOcc < qOcc) && (rVirt < sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = rVirt;
    bVirt = sVirt;
  } else if ((pOcc > qOcc) && (rVirt > sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = sVirt;
    bVirt = rVirt;
  } else if ((pOcc < qOcc) && (rVirt > sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = sVirt;
    bVirt = rVirt;
    multiplier = -1.;
  } else if ((pOcc > qOcc) && (rVirt < sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = rVirt;
    bVirt = sVirt;
    multiplier = -1.;
  }

  kernel.h(qubits[iOcc]);
  kernel.h(qubits[jOcc]);
  kernel.h(qubits[aVirt]);
  kernel.rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  kernel.rx(-M_PI_2, qubits[bVirt]);
  kernel.h(qubits[aVirt]);

  kernel.rx(M_PI_2, qubits[aVirt]);
  kernel.h(qubits[bVirt]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.rx(-M_PI_2, qubits[aVirt]);
  kernel.h(qubits[jOcc]);

  kernel.rx(M_PI_2, qubits[jOcc]);
  kernel.h(qubits[aVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  kernel.h(qubits[bVirt]);
  kernel.h(qubits[aVirt]);

  kernel.rx(M_PI_2, qubits[aVirt]);
  kernel.rx(M_PI_2, qubits[bVirt]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.rx(-M_PI_2, qubits[jOcc]);
  kernel.h(qubits[iOcc]);

  kernel.rx(M_PI_2, qubits[iOcc]);
  kernel.h(qubits[jOcc]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  kernel.rx(-M_PI_2, qubits[bVirt]);
  kernel.rx(-M_PI_2, qubits[aVirt]);

  kernel.h(qubits[aVirt]);
  kernel.h(qubits[bVirt]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.h(qubits[bVirt]);
  kernel.h(qubits[jOcc]);

  kernel.rx(M_PI_2, qubits[jOcc]);
  kernel.rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  kernel.rx(-M_PI_2, qubits[bVirt]);
  kernel.h(qubits[aVirt]);

  kernel.rx(M_PI_2, qubits[aVirt]);
  kernel.h(qubits[bVirt]);

  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    kernel.template x<cudaq::ctrl>(qubits[i], qubits[i + 1]);

  kernel.rz(-0.125 * multiplier * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);
  kernel.template x<cudaq::ctrl>(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    kernel.template x<cudaq::ctrl>(qubits[i - 1], qubits[i]);

  kernel.h(qubits[bVirt]);
  kernel.rx(-M_PI_2, qubits[aVirt]);
  kernel.rx(-M_PI_2, qubits[jOcc]);
  kernel.rx(-M_PI_2, qubits[iOcc]);
}

__qpu__ void doubleExcitation(cudaq::qview<> qubits, std::size_t pOcc,
                              std::size_t qOcc, std::size_t rVirt,
                              std::size_t sVirt, double theta) {
  std::size_t iOcc = 0, jOcc = 0, aVirt = 0, bVirt = 0;
  if ((pOcc < qOcc) && (rVirt < sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = rVirt;
    bVirt = sVirt;
  } else if ((pOcc > qOcc) && (rVirt > sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = sVirt;
    bVirt = rVirt;
  } else if ((pOcc < qOcc) && (rVirt > sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = sVirt;
    bVirt = rVirt;
    theta *= -1.;
  } else if ((pOcc > qOcc) && (rVirt < sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = rVirt;
    bVirt = sVirt;
    theta *= -1.;
  }

  h(qubits[iOcc]);
  h(qubits[jOcc]);
  h(qubits[aVirt]);
  rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);

  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[aVirt]);
  h(qubits[jOcc]);

  rx(M_PI_2, qubits[jOcc]);
  h(qubits[aVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  h(qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  rx(M_PI_2, qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[jOcc]);
  h(qubits[iOcc]);

  rx(M_PI_2, qubits[iOcc]);
  h(qubits[jOcc]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  rx(-M_PI_2, qubits[aVirt]);

  h(qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[bVirt]);
  h(qubits[jOcc]);

  rx(M_PI_2, qubits[jOcc]);
  rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[bVirt]);
  rx(-M_PI_2, qubits[aVirt]);
  rx(-M_PI_2, qubits[jOcc]);
  rx(-M_PI_2, qubits[iOcc]);
}

__qpu__ void uccsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
                   std::size_t numElectrons) {

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      get_uccsd_excitations(numElectrons, qubits.size());

  std::size_t thetaCounter = 0;
  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(singlesAlpha.size())))
    singleExcitation(qubits, singlesAlpha[i][0], singlesAlpha[i][1],
                     thetas[thetaCounter++]);

  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(singlesBeta.size())))
    singleExcitation(qubits, singlesBeta[i][0], singlesBeta[i][1],
                     thetas[thetaCounter++]);

  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(doublesMixed.size())))
    doubleExcitation(qubits, doublesMixed[i][0], doublesMixed[i][1],
                     doublesMixed[i][2], doublesMixed[i][3],
                     thetas[thetaCounter++]);

  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(doublesAlpha.size())))
    doubleExcitation(qubits, doublesAlpha[i][0], doublesAlpha[i][1],
                     doublesAlpha[i][2], doublesAlpha[i][3],
                     thetas[thetaCounter++]);

  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(doublesBeta.size())))
    doubleExcitation(qubits, doublesBeta[i][0], doublesBeta[i][1],
                     doublesBeta[i][2], doublesBeta[i][3],
                     thetas[thetaCounter++]);
}

template <typename Kernel>
void uccsd(Kernel &kernel, QuakeValue &qubits, QuakeValue &thetas,
           std::size_t numElectrons, std::size_t numQubits) {

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      get_uccsd_excitations(numElectrons, numQubits);

  std::size_t thetaCounter = 0;
  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(singlesAlpha.size()))) {
    // FIXME fix const correctness on quake value
    auto theta = thetas[thetaCounter++];
    singleExcitation(kernel, qubits, singlesAlpha[i][0], singlesAlpha[i][1],
                     theta);
  }

  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(singlesBeta.size()))) {
    auto theta = thetas[thetaCounter++];
    singleExcitation(kernel, qubits, singlesBeta[i][0], singlesBeta[i][1],
                     theta);
  }

  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(doublesMixed.size()))) {
    auto theta = thetas[thetaCounter++];
    doubleExcitation(kernel, qubits, doublesMixed[i][0], doublesMixed[i][1],
                     doublesMixed[i][2], doublesMixed[i][3], theta);
  }

  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(doublesAlpha.size()))) {
    auto theta = thetas[thetaCounter++];
    doubleExcitation(kernel, qubits, doublesAlpha[i][0], doublesAlpha[i][1],
                     doublesAlpha[i][2], doublesAlpha[i][3], theta);
  }

  for (auto i : cudaq::range(
           static_cast<std::make_signed_t<std::size_t>>(doublesBeta.size()))) {
    auto theta = thetas[thetaCounter++];
    doubleExcitation(kernel, qubits, doublesBeta[i][0], doublesBeta[i][1],
                     doublesBeta[i][2], doublesBeta[i][3], theta);
  }
}

} // namespace cudaq
