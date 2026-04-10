/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <nanobind/nanobind.h>
#include <pybind11/pybind11.h>

namespace cudaq::python {

/// Return a nanobind borrowed-reference view of a pybind11 module.
/// The pybind11 module retains ownership. The returned nb::module_ must not
/// outlive it.
inline nanobind::module_ nb_from_pb(pybind11::module_ &m) {
  return nanobind::borrow<nanobind::module_>(m.ptr());
}

} // namespace cudaq::python
