/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_helpers.h"
#include "cudaq/operators.h"
#include <complex>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

namespace cudaq::details {

cudaq::parameter_map kwargs_to_param_map(const nb::kwargs &kwargs) {
  cudaq::parameter_map params;
  for (auto [keyPy, valuePy] : kwargs) {
    std::string key = nb::str(keyPy).c_str();
    std::complex<double> value = nb::cast<std::complex<double>>(valuePy);
    params.insert(params.end(),
                  std::pair<std::string, std::complex<double>>(key, value));
  }
  return params;
}

std::unordered_map<std::string, std::string>
kwargs_to_param_description(const nb::kwargs &kwargs) {
  std::unordered_map<std::string, std::string> param_desc;
  for (auto [keyPy, valuePy] : kwargs) {
    std::string key = nb::str(keyPy).c_str();
    std::string value = nb::str(valuePy).c_str();
    param_desc.insert(param_desc.end(),
                      std::pair<std::string, std::string>(key, value));
  }
  return param_desc;
}

nb::ndarray<nb::numpy, std::complex<double>>
cmat_to_numpy(complex_matrix &cmat) {
  auto rows = cmat.rows();
  auto cols = cmat.cols();
  auto data = cmat.get_data(complex_matrix::order::row_major);
  size_t shape[2] = {rows, cols};
  return nb::ndarray<nb::numpy, std::complex<double>>(data, 2, shape,
                                                      nb::handle());
};

} // namespace cudaq::details
