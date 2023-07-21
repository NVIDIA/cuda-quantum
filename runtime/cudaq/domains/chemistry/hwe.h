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

/// @brief Given the number of qubits and number of layers,
/// return the number of variational parameters for the HWE ansatz.
std::size_t num_hwe_parameters(std::size_t numQubits, std::size_t numLayers) {
  return 2 * numQubits * (1 + numLayers);
}

/// @brief Utility type describing a CNOT coupler for the HWE ansatz.
struct cnot_coupling {
  std::size_t source;
  std::size_t target;
  cnot_coupling(std::size_t s, std::size_t t) : source(s), target(t) {}
};

/// @brief Utility function generation default coupling for the HWE ansatz.
inline std::vector<cnot_coupling> default_cnot_coupling(std::size_t numQubits) {
  std::vector<cnot_coupling> cnotCoupling;
  cnotCoupling.reserve(numQubits - 1);
  for (std::size_t q = 0; q < numQubits - 1; q++)
    cnotCoupling.push_back({q, q + 1});

  return cnotCoupling;
}

/// @brief This CUDA Quantum kernel implements the hardware-efficient ansatz
/// from Kandala et. al [https://arxiv.org/abs/1704.05018]. It takes the
/// qubits the state is on, the number of layers, the vector of variational
/// parameters and the CNOT couplers.
__qpu__ void hwe(cudaq::qview<> qubits, std::size_t numLayers,
                 std::vector<double> parameters,
                 const std::vector<cnot_coupling> &cnotCoupling) {
  std::size_t numQubits = qubits.size();

  for (std::size_t thetaCounter = 0, i = 0; i < numQubits; i++) {
    ry(parameters[thetaCounter], qubits[i]);
    rz(parameters[thetaCounter + 1], qubits[i]);
    thetaCounter += 2;
  }

  for (std::size_t thetaCounter = numQubits * 2, layer = 0; layer < numLayers;
       layer++) {
    for (auto &cnot : cnotCoupling)
      x<cudaq::ctrl>(qubits[cnot.source], qubits[cnot.target]);

    for (std::size_t q = 0; q < numQubits; q++) {
      ry(parameters[thetaCounter], qubits[q]);
      rz(parameters[thetaCounter + 1], qubits[q]);
      thetaCounter += 2;
    }
  }
}

/// @brief This CUDA Quantum kernel implements the hardware-efficient ansatz
/// from Kandala et. al [https://arxiv.org/abs/1704.05018]. It takes the
/// qubits the state is on, the number of layers, and the vector of
/// variational parameters. Uses default CNOT couplers.
__qpu__ void hwe(cudaq::qview<> qubits, std::size_t numLayers,
                 std::vector<double> parameters) {
  std::size_t numQubits = qubits.size();

  // generate default cnotCoupling and forward the call
  std::vector<cnot_coupling> cnotCoupling = default_cnot_coupling(numQubits);
  hwe(qubits, numLayers, parameters, cnotCoupling);
}

/// @brief This function creates the hardware-efficient ansatz from Kandala et.
/// al [https://arxiv.org/abs/1704.05018] on an existing kernel_builder
/// instance. It takes the qubits the state is on, the number of qubits and
/// layers, the vector of variational parameters and the CNOT couplers as input
/// as a QuakeValue.
template <typename KernelBuilder>
void hwe(KernelBuilder &kernel, QuakeValue &qubits, std::size_t numQubits,
         std::size_t numLayers, QuakeValue &parameters,
         const std::vector<cnot_coupling> &cnotCoupling) {

  for (std::size_t thetaCounter = 0, i = 0; i < numQubits; i++) {
    kernel.ry(parameters[thetaCounter], qubits[i]);
    kernel.rz(parameters[thetaCounter + 1], qubits[i]);
    thetaCounter += 2;
  }

  for (std::size_t thetaCounter = numQubits * 2, layer = 0; layer < numLayers;
       layer++) {
    for (auto &cnot : cnotCoupling)
      kernel.template x<cudaq::ctrl>(qubits[cnot.source], qubits[cnot.target]);

    for (std::size_t q = 0; q < numQubits; q++) {
      kernel.ry(parameters[thetaCounter], qubits[q]);
      kernel.rz(parameters[thetaCounter + 1], qubits[q]);
      thetaCounter += 2;
    }
  }
}

/// @brief This function creates the hardware-efficient ansatz from Kandala et.
/// al [https://arxiv.org/abs/1704.05018] on an existing kernel_builder
/// instance. It takes the qubits the state is on, the number of layers, and the
/// vector of variational parameters as input as a QuakeValue.
template <typename KernelBuilder>
void hwe(KernelBuilder &kernel, QuakeValue &qubits, std::size_t numQubits,
         std::size_t numLayers, QuakeValue &parameters) {

  // generate default cnotCoupling and forward the call
  std::vector<cnot_coupling> cnotCoupling = default_cnot_coupling(numQubits);
  hwe(kernel, qubits, numQubits, numLayers, parameters, cnotCoupling);
}

} // namespace cudaq