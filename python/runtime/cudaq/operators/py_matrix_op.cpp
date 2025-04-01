/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_matrix_op.h"

namespace cudaq {

void bindOperatorsModule(py::module &mod) {
  // Binding the functions in `cudaq::operators` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto operators_submodule = mod.def_submodule("operators");
  operators_submodule.def(
      "number", &cudaq::matrix_op::number<cudaq::matrix_handler>, py::arg("target"),
      "Returns a number operator on the given target index.");
  operators_submodule.def(
      "parity", &cudaq::matrix_op::parity<cudaq::matrix_handler>, py::arg("target"),
      "Returns a parity operator on the given target index.");
  operators_submodule.def(
      "position", &cudaq::matrix_op::position<cudaq::matrix_handler>, py::arg("target"),
      "Returns a position operator on the given target index.");
  operators_submodule.def(
      "momentum", &cudaq::matrix_op::momentum<cudaq::matrix_handler>, py::arg("target"),
      "Returns a momentum operator on the given target index.");
  operators_submodule.def(
      "squeeze", &cudaq::matrix_op::squeeze<cudaq::matrix_handler>, py::arg("target"),
      "Returns a squeezing operator on the given target index.");
  operators_submodule.def(
      "displace", &cudaq::matrix_op::displace<cudaq::matrix_handler>, py::arg("target"),
      "Returns a displacement operator on the given target index.");
}

void bindOperatorsWrapper(py::module &mod) {
  bindOperatorsModule(mod);
  //bindMatrixOperator(mod);
  py::implicitly_convertible<matrix_op_term, matrix_op>();
}

} // namespace cudaq