/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cudaq::details {
cudaq::parameter_map kwargs_to_param_map(const py::kwargs &kwargs);
std::unordered_map<std::string, std::string>
kwargs_to_param_description(const py::kwargs &kwargs);
py::array_t<std::complex<double>>
cmat_to_numpy(std::size_t rows, std::size_t cols,
              complex_matrix::value_type const *const data);
} // namespace cudaq::details
