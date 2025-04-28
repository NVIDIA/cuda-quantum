/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <functional>
#include <unordered_map>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_scalar_op.h"

namespace cudaq {

void bindScalarOperator(py::module &mod) {
  using scalar_callback =
      std::function<std::complex<double>(const parameter_map &)>;

  auto kwargs_to_param_map = [](const py::kwargs &kwargs) {
    parameter_map params;
    for (auto &[keyPy, valuePy] : kwargs) {
      std::string key = py::str(keyPy);
      std::complex<double> value = valuePy.cast<std::complex<double>>();
      params.insert(params.end(),
                    std::pair<std::string, std::complex<double>>(key, value));
    }
    return params;
  };

  auto kwargs_to_param_description = [](const py::kwargs &kwargs) {
    std::unordered_map<std::string, std::string> param_desc;
    for (auto &[keyPy, valuePy] : kwargs) {
      std::string key = py::str(keyPy);
      std::string value = py::str(valuePy);
      param_desc.insert(param_desc.end(),
                        std::pair<std::string, std::string>(key, value));
    }
    return param_desc;
  };

  py::class_<scalar_operator>(mod, "ScalarOperator")

      // properties

      .def_property_readonly("parameters",
                             &scalar_operator::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")

      // constructors

      .def(py::init<>(), "Creates a scalar operator with constant value 1.")
      .def(py::init<double>(),
           "Creates a scalar operator with the given constant value.")
      .def(py::init<std::complex<double>>(),
           "Creates a scalar operator with the given constant value.")
      .def(py::init([&kwargs_to_param_description](const scalar_callback &func,
                                                   const py::kwargs &kwargs) {
             return scalar_operator(func, kwargs_to_param_description(kwargs));
           }),
           py::arg("callback"),
           "Creates a scalar operator where the given callback function is "
           "invoked during evaluation.")
      .def(py::init<const scalar_operator &>(), "Copy constructor.")

      // evaluations

      .def(
          "evaluate",
          [&kwargs_to_param_map](const scalar_operator &self,
                                 const py::kwargs &kwargs) {
            return self.evaluate(kwargs_to_param_map(kwargs));
          },
          "Evaluated value of the operator.")

      // comparisons

      .def("__eq__", &scalar_operator::operator==)

      // general utility functions

      .def("is_constant", &scalar_operator::is_constant,
           "Returns true if the scalar is a constant value.")
      .def("__str__", &scalar_operator::to_string,
           "Returns the string representation of the operator.");
}

void bindScalarWrapper(py::module &mod) {
  bindScalarOperator(mod);
  py::implicitly_convertible<double, scalar_operator>();
  py::implicitly_convertible<std::complex<double>, scalar_operator>();
}

} // namespace cudaq
