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
#include "py_fermion_op.h"

namespace cudaq {

void bindFermionModule(py::module &mod) {
  // Binding the functions in `cudaq::fermion` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto fermion_submodule = mod.def_submodule("fermion");
  fermion_submodule.def(
      "create", &cudaq::fermion_op::create<cudaq::fermion_handler>, py::arg("target"),
      "Returns a fermionic creation operator on the given target index.");
  fermion_submodule.def(
      "annihilate", &cudaq::fermion_op::annihilate<cudaq::fermion_handler>, py::arg("target"),
      "Returns a fermionic annihilation operator on the given target index.");
  fermion_submodule.def(
      "number", &cudaq::fermion_op::number<cudaq::fermion_handler>, py::arg("target"),
      "Returns a fermionic number operator on the given target index.");
}

void bindFermionWrapper(py::module &mod) {
  bindFermionModule(mod);
  //bindFermionOperator(mod);
  py::implicitly_convertible<fermion_op_term, fermion_op>();
}

} // namespace cudaq