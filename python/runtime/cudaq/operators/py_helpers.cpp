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
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace cudaq::details {

cudaq::parameter_map kwargs_to_param_map(const py::kwargs &kwargs) {
  cudaq::parameter_map params;
  for (auto &[keyPy, valuePy] : kwargs) {
    std::string key = py::str(keyPy);
    std::complex<double> value = valuePy.cast<std::complex<double>>();
    params.insert(params.end(),
                  std::pair<std::string, std::complex<double>>(key, value));
  }
  return params;
}

std::unordered_map<std::string, std::string>
kwargs_to_param_description(const py::kwargs &kwargs) {
  std::unordered_map<std::string, std::string> param_desc;
  for (auto &[keyPy, valuePy] : kwargs) {
    std::string key = py::str(keyPy);
    std::string value = py::str(valuePy);
    param_desc.insert(param_desc.end(),
                      std::pair<std::string, std::string>(key, value));
  }
  return param_desc;
}

py::array_t<std::complex<double>> cmat_to_numpy(complex_matrix &cmat) {
  auto rows = cmat.rows();
  auto cols = cmat.cols();
  auto data = cmat.get_data(complex_matrix::order::row_major);
  std::vector<ssize_t> shape = {static_cast<ssize_t>(rows),
                                static_cast<ssize_t>(cols)};
  std::vector<ssize_t> strides = {
      static_cast<ssize_t>(sizeof(std::complex<double>) * cols),
      static_cast<ssize_t>(sizeof(std::complex<double>))};

  // Return a numpy array without copying data
  return py::array_t<std::complex<double>>(shape, strides, data);
};

} // namespace cudaq::details
