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

cudaq::parameter_map kwargs_to_param_map(const nanobind::kwargs &kwargs) {
  cudaq::parameter_map params;
  for (auto [keyPy, valuePy] : kwargs) {
    std::string key = nanobind::str(keyPy).c_str();
    std::complex<double> value = nanobind::cast<std::complex<double>>(valuePy);
    params.insert(params.end(),
                  std::pair<std::string, std::complex<double>>(key, value));
  }
  return params;
}

cudaq::parameter_map kwargs_to_param_map(nanobind::kwargs &kwargs,
                                         bool &invert_order) {
  nanobind::str invert_key("invert_order");
  nanobind::object inv = kwargs.attr("pop")(invert_key, nanobind::bool_(false));
  invert_order = nanobind::cast<bool>(inv);
  return kwargs_to_param_map(static_cast<const nanobind::kwargs &>(kwargs));
}

std::unordered_map<std::string, std::string>
kwargs_to_param_description(const nanobind::kwargs &kwargs) {
  std::unordered_map<std::string, std::string> param_desc;
  for (auto [keyPy, valuePy] : kwargs) {
    std::string key = nanobind::str(keyPy).c_str();
    std::string value = nanobind::str(valuePy).c_str();
    param_desc.insert(param_desc.end(),
                      std::pair<std::string, std::string>(key, value));
  }
  return param_desc;
}

nanobind::object cmat_to_numpy(complex_matrix &cmat) {
  auto rows = cmat.rows();
  auto cols = cmat.cols();
  auto *data = cmat.get_data(complex_matrix::order::row_major);

  // Use .cast() to force immediate creation of the numpy array.
  // Since no owner is specified, rv_policy::automatic will copy the data,
  // making this safe even when cmat is a temporary (e.g. in get_unitary).
  return nanobind::ndarray<nanobind::numpy, std::complex<double>,
                           nanobind::shape<-1, -1>>(data, {rows, cols},
                                                    nanobind::handle())
      .cast();
};

} // namespace cudaq::details
