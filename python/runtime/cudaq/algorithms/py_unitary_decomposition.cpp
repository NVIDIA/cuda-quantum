/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/UnitaryDecomposition.h"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {

void bindPyUnitaryDecomposition(py::module &mod) {

  auto unitarySynthesis = mod.def_submodule("unitary_synthesis");

  // Register EulerAngles structure
  py::class_<cudaq::detail::EulerAngles>(unitarySynthesis, "EulerAngles")
      .def_readonly("alpha", &cudaq::detail::EulerAngles::alpha)
      .def_readonly("beta", &cudaq::detail::EulerAngles::beta)
      .def_readonly("gamma", &cudaq::detail::EulerAngles::gamma)
      .def_readonly("phase", &cudaq::detail::EulerAngles::phase);

  // Register KAKComponents structure
  py::class_<cudaq::detail::KAKComponents>(unitarySynthesis, "KAKComponents")
      .def_readonly("a0", &cudaq::detail::KAKComponents::a0)
      .def_readonly("a1", &cudaq::detail::KAKComponents::a1)
      .def_readonly("b0", &cudaq::detail::KAKComponents::b0)
      .def_readonly("b1", &cudaq::detail::KAKComponents::b1)
      .def_readonly("x", &cudaq::detail::KAKComponents::x)
      .def_readonly("y", &cudaq::detail::KAKComponents::y)
      .def_readonly("z", &cudaq::detail::KAKComponents::z)
      .def_readonly("phase", &cudaq::detail::KAKComponents::phase);

  // Register the core decomposition functions
  unitarySynthesis.def("zyz_decompose", &cudaq::detail::decomposeZYZ,
                       "Decompose single-qubit unitary into ZYZ Euler angles",
                       py::arg("matrix"));

  unitarySynthesis.def("kak_decompose", &cudaq::detail::decomposeKAK,
                       "Decompose two-qubit unitary using KAK decomposition",
                       py::arg("matrix"));
}

} // namespace cudaq
