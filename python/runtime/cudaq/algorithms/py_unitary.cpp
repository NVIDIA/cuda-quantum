/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_unitary.h"
#include "cudaq/algorithms/unitary.h"
#include "runtime/cudaq/operators/py_helpers.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

using namespace cudaq;

/// Compute the unitary of this kernel module.
static py::array get_unitary_impl(const std::string &shortName,
                                  MlirModule module, MlirType returnTy,
                                  py::args args) {
  // Uses the same pattern as py_draw.
  auto f = [=]() {
    return cudaq::marshal_and_launch_module(shortName, module, returnTy, args);
  };

  // Return as numpy array (dim, dim), complex128
  auto temp = contrib::get_unitary_cmat(std::move(f));
  return details::cmat_to_numpy(temp);
}

/// Bind the get_unitary cudaq function
void cudaq::bindPyUnitary(py::module &mod) {
  mod.def("get_unitary_impl", get_unitary_impl,
          "See python documentation for get_unitary().");
}
