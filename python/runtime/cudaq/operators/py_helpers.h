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

namespace nb = nanobind;

namespace cudaq::details {
cudaq::parameter_map kwargs_to_param_map(const nb::kwargs &kwargs);
std::unordered_map<std::string, std::string>
kwargs_to_param_description(const nb::kwargs &kwargs);
nb::ndarray<nb::numpy, std::complex<double>>
cmat_to_numpy(complex_matrix &cmat);
} // namespace cudaq::details
