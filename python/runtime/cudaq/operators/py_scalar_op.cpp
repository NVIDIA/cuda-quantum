/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <functional>
#include <unordered_map>

#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_helpers.h"
#include "py_scalar_op.h"

namespace cudaq {

void bindScalarOperator(nanobind::module_ &mod) {
  using scalar_callback =
      std::function<std::complex<double>(const parameter_map &)>;

  nanobind::class_<scalar_operator>(mod, "ScalarOperator")

      // properties

      .def_prop_ro("parameters", &scalar_operator::get_parameter_descriptions,
                   "Returns a dictionary that maps each parameter "
                   "name to its description.")

      // constructors

      .def(nanobind::init<>(),
           "Creates a scalar operator with constant value 1.")
      .def(nanobind::init<double>(),
           "Creates a scalar operator with the given constant value.")
      .def(nanobind::init<std::complex<double>>(),
           "Creates a scalar operator with the given constant value.")
      .def(
          "__init__",
          [](scalar_operator *self, const scalar_callback &func,
             const nanobind::kwargs &kwargs) {
            new (self) scalar_operator(
                func, details::kwargs_to_param_description(kwargs));
          },
          nanobind::arg("callback"), nanobind::arg("kwargs"),
          "Creates a scalar operator where the given callback function is "
          "invoked during evaluation.")
      .def(nanobind::init<const scalar_operator &>(), "Copy constructor.")

      // evaluations

      .def(
          "evaluate",
          [](const scalar_operator &self, const nanobind::kwargs &kwargs) {
            return self.evaluate(details::kwargs_to_param_map(kwargs));
          },
          "Evaluated value of the operator.")

      // comparisons

      .def("__eq__", &scalar_operator::operator==, nanobind::is_operator())

      // general utility functions

      .def("is_constant", &scalar_operator::is_constant,
           "Returns true if the scalar is a constant value.")
      .def("__str__", &scalar_operator::to_string,
           "Returns the string representation of the operator.");
}

void bindScalarWrapper(nanobind::module_ &mod) {
  bindScalarOperator(mod);
  nanobind::implicitly_convertible<double, scalar_operator>();
  nanobind::implicitly_convertible<std::complex<double>, scalar_operator>();
}

} // namespace cudaq
