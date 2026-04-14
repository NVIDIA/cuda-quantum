/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace py = nanobind;

namespace cudaq::details {
cudaq::parameter_map kwargs_to_param_map(const py::kwargs &kwargs);
/// Extracts parameter map from kwargs, also extracting an optional
/// "invert_order" boolean (defaults to false if not present).
cudaq::parameter_map kwargs_to_param_map(py::kwargs &kwargs,
                                         bool &invert_order);
std::unordered_map<std::string, std::string>
kwargs_to_param_description(const py::kwargs &kwargs);
py::object cmat_to_numpy(complex_matrix &cmat);
} // namespace cudaq::details
