/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/stl.h>

#include "cudaq/domains/chemistry/hwe.h"
#include "cudaq/domains/chemistry/uccsd.h"
#include "py_chemistry.h"

namespace cudaq {

void bindChemistry(py::module &mod) {
  mod.def("uccsd_num_parameters", &cudaq::uccsd_num_parameters,
          "For the given number of electrons and qubits, return the required "
          "number of uccsd parameters.");
  mod.def("uccsd", &cudaq::uccsd<kernel_builder<>>,
          "Generate the unitary coupled cluster singlet doublet CUDA Quantum "
          "kernel. This function takes as input the existing `cudaq.Kernel` to "
          "append to, pre-allocated qubits, list of parameters, number of "
          "electrons, and number of qubits as input, in that order.");

  mod.def("num_hwe_parameters", &cudaq::num_hwe_parameters,
          "For the given number of qubits and layers, return the required "
          "number of hwe parameters.");
  // use explicit overload as we have two overloaded signatures, i.e. with and
  // without CNOT couplers
  mod.def(
      "hwe",
      [](kernel_builder<> &kernel, QuakeValue &qubits, std::size_t numQubits,
         std::size_t numLayers, QuakeValue &parameters,
         const std::vector<cnot_coupling> &cnotCoupling) {
        return cudaq::hwe(kernel, qubits, numQubits, numLayers, parameters,
                          cnotCoupling);
      },
      "Generate the hardware-efficient CUDA Quantum Kernel. This function "
      "takes as input the existing `cudaq.Kernel` to append to, "
      "pre-allocated qubits, number of qubits, number of layers, the list "
      "of parameters, and the CNOT couplers as input, in that order.");

  mod.def(
      "hwe",
      [](kernel_builder<> &kernel, QuakeValue &qubits, std::size_t numQubits,
         std::size_t numLayers, QuakeValue &parameters) {
        return cudaq::hwe(kernel, qubits, numQubits, numLayers, parameters);
      },
      "Generate the hardware-efficient CUDA Quantum Kernel. This function "
      "takes as input the existing `cudaq.Kernel` to append to, "
      "pre-allocated qubits, number of qubits, number of layers, and the list "
      "of parameters as input, in that order. Using default CNOT couplers.");
}

} // namespace cudaq
