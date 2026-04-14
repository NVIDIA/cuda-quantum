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

cudaq::parameter_map kwargs_to_param_map(const py::kwargs &kwargs) {
  cudaq::parameter_map params;
  for (auto [keyPy, valuePy] : kwargs) {
    std::string key = py::str(keyPy).c_str();
    std::complex<double> value = py::cast<std::complex<double>>(valuePy);
    params.insert(params.end(),
                  std::pair<std::string, std::complex<double>>(key, value));
  }
  return params;
}

cudaq::parameter_map kwargs_to_param_map(py::kwargs &kwargs,
                                         bool &invert_order) {
  py::str invert_key("invert_order");
  py::object inv = kwargs.attr("pop")(invert_key, py::bool_(false));
  invert_order = py::cast<bool>(inv);
  return kwargs_to_param_map(static_cast<const py::kwargs &>(kwargs));
}

std::unordered_map<std::string, std::string>
kwargs_to_param_description(const py::kwargs &kwargs) {
  std::unordered_map<std::string, std::string> param_desc;
  for (auto [keyPy, valuePy] : kwargs) {
    std::string key = py::str(keyPy).c_str();
    std::string value = py::str(valuePy).c_str();
    param_desc.insert(param_desc.end(),
                      std::pair<std::string, std::string>(key, value));
  }
  return param_desc;
}

py::object cmat_to_numpy(complex_matrix &cmat) {
  auto rows = cmat.rows();
  auto cols = cmat.cols();
  auto *data = cmat.get_data(complex_matrix::order::row_major);

  // Use .cast() to force immediate creation of the numpy array.
  // Since no owner is specified, rv_policy::automatic will copy the data,
  // making this safe even when cmat is a temporary (e.g. in get_unitary).
  return py::ndarray<py::numpy, std::complex<double>, py::shape<-1, -1>>(
             data, {rows, cols}, py::handle())
      .cast();
};

} // namespace cudaq::details
