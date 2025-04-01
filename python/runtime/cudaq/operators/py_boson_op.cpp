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
#include "py_boson_op.h"

namespace cudaq {

void bindBosonModule(py::module &mod) {
  // Binding the functions in `cudaq::boson` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto boson_submodule = mod.def_submodule("boson");
  boson_submodule.def(
      "create", &cudaq::boson_op::create<cudaq::boson_handler>, py::arg("target"),
      "Returns a bosonic creation operator on the given target index.");
  boson_submodule.def(
      "annihilate", &cudaq::boson_op::annihilate<cudaq::boson_handler>, py::arg("target"),
      "Returns a bosonic annihilation operator on the given target index.");
  boson_submodule.def(
      "number", &cudaq::boson_op::number<cudaq::boson_handler>, py::arg("target"),
      "Returns a bosonic number operator on the given target index.");
  boson_submodule.def(
      "position", &cudaq::boson_op::position<cudaq::boson_handler>, py::arg("target"),
      "Returns a bosonic position operator on the given target index.");
  boson_submodule.def(
      "momentum", &cudaq::boson_op::momentum<cudaq::boson_handler>, py::arg("target"),
      "Returns a bosonic momentum operator on the given target index.");
}

void bindBosonWrapper(py::module &mod) {
  bindBosonModule(mod);
  //bindBosonOperator(mod);
  py::implicitly_convertible<boson_op_term, boson_op>();
}

} // namespace cudaq