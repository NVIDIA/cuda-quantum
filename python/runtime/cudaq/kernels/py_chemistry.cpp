/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
#include <pybind11/stl.h>

#include "cudaq/domains/chemistry/uccsd.h"
#include "py_chemistry.h"

namespace cudaq {

void bindChemistry(py::module &mod) {
  mod.def("uccsd", &cudaq::uccsd<kernel_builder<>>,
          "Generate the unitary coupled cluster singlet doublet CUDA Quantum "
          "kernel. This function takes as input the existing `cudaq.Kernel` to "
          "append to, pre-allocated qubits, list of parameters, number of "
          "electrons, and number of qubits as input, in that order.");
}

} // namespace cudaq
