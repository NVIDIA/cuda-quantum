/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "cudaq/operators.h"
#include "py_handlers.h"
#include "py_helpers.h"

namespace cudaq {

void bindPauli(py::module_ mod) {
  py::enum_<pauli>(mod, "Pauli",
                   "An enumeration representing the types of Pauli matrices.")
      .value("X", pauli::X)
      .value("Y", pauli::Y)
      .value("Z", pauli::Z)
      .value("I", pauli::I);
}

void bindOperatorHandlers(py::module_ &mod) {
  using matrix_callback = std::function<complex_matrix(
      const std::vector<int64_t> &, const parameter_map &)>;

  py::class_<matrix_handler>(mod, "MatrixOperatorElement")
      .def_prop_ro(
          "id",
          [](const matrix_handler &self) { return self.to_string(false); },
          "Returns the id used to define and instantiate the operator.")
      .def_prop_ro("degrees", &matrix_handler::degrees,
                   "Returns a vector that lists all degrees of "
                   "freedom that the operator targets.")
      .def_prop_ro("parameters", &matrix_handler::get_parameter_descriptions,
                   "Returns a dictionary that maps each parameter "
                   "name to its description.")
      .def_prop_ro("expected_dimensions",
                   &matrix_handler::get_expected_dimensions,
                   "The number of levels, that is the dimension, for "
                   "each degree of freedom "
                   "in canonical order that the operator acts on. A "
                   "value of zero or less "
                   "indicates that the operator is defined for any "
                   "dimension of that degree.")
      .def(py::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(
          "__init__",
          [](matrix_handler *self, std::string operator_id,
             std::vector<std::size_t> degrees) {
            new (self)
                matrix_handler(std::move(operator_id), std::move(degrees));
          },
          py::arg("id"), py::arg("degrees"),
          "Creates the matrix operator with the given id acting on the given "
          "degrees of "
          "freedom. Throws a runtime exception if no operator with that id "
          "has been defined.")
      .def(py::init<const matrix_handler &>(), "Copy constructor.")
      .def("__eq__", &matrix_handler::operator==, py::is_operator())
      .def("to_string", &matrix_handler::to_string, py::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [](const matrix_handler &self,
             std::optional<dimension_map> dimensions,
             std::optional<parameter_map> params) {
            dimension_map dims = dimensions.value_or(dimension_map());
            parameter_map pm = params.value_or(parameter_map());
            auto cmat = self.to_matrix(dims, pm);
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions").none() = py::none(),
          py::arg("parameters").none() = py::none(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [](const matrix_handler &self,
             std::optional<dimension_map> dimensions, py::kwargs kwargs) {
            dimension_map dims = dimensions.value_or(dimension_map());
            auto cmat =
                self.to_matrix(dims, details::kwargs_to_param_map(kwargs));
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions").none() = py::none(),
          "Returns the matrix representation of the operator.")

      // tools for custom operators
      .def_static(
          "_define",
          [](std::string operator_id, std::vector<int64_t> expected_dimensions,
             const matrix_callback &func, bool overwrite = false,
             py::kwargs kwargs) {
            // we need to make sure the python function that is stored in
            // the static dictionary containing the operator definitions
            // is properly cleaned up - otherwise python will hang on exit...
            auto atexit = py::module_::import_("atexit");
            atexit.attr("register")(py::cpp_function([operator_id]() {
              matrix_handler::remove_definition(operator_id);
            }));
            if (overwrite)
              matrix_handler::remove_definition(operator_id);
            matrix_handler::define(
                std::move(operator_id), std::move(expected_dimensions), func,
                details::kwargs_to_param_description(kwargs));
          },
          "Defines a matrix operator with the given name and dimensions whose"
          "matrix representation can be obtained by invoking the given "
          "callback function.");

  py::class_<boson_handler>(mod, "BosonOperatorElement")
      .def_prop_ro("target", &boson_handler::target,
                   "Returns the degree of freedom that the operator targets.")
      .def_prop_ro("degrees", &boson_handler::degrees,
                   "Returns a vector that lists all degrees of "
                   "freedom that the operator targets.")
      .def(py::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(py::init<const boson_handler &>(), "Copy constructor.")
      .def("__eq__", &boson_handler::operator==, py::is_operator())
      .def("to_string", &boson_handler::to_string, py::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [](const boson_handler &self, std::optional<dimension_map> dimensions,
             std::optional<parameter_map> params) {
            dimension_map dims = dimensions.value_or(dimension_map());
            parameter_map pm = params.value_or(parameter_map());
            auto cmat = self.to_matrix(dims, pm);
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions").none() = py::none(),
          py::arg("parameters").none() = py::none(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [](const boson_handler &self, std::optional<dimension_map> dimensions,
             py::kwargs kwargs) {
            dimension_map dims = dimensions.value_or(dimension_map());
            auto cmat =
                self.to_matrix(dims, details::kwargs_to_param_map(kwargs));
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions").none() = py::none(),
          "Returns the matrix representation of the operator.");

  py::class_<fermion_handler>(mod, "FermionOperatorElement")
      .def_prop_ro("target", &fermion_handler::target,
                   "Returns the degree of freedom that the operator targets.")
      .def_prop_ro("degrees", &fermion_handler::degrees,
                   "Returns a vector that lists all degrees of "
                   "freedom that the operator targets.")
      .def(py::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(py::init<const fermion_handler &>(), "Copy constructor.")
      .def("__eq__", &fermion_handler::operator==, py::is_operator())
      .def("to_string", &fermion_handler::to_string, py::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [](const fermion_handler &self,
             std::optional<dimension_map> dimensions,
             std::optional<parameter_map> params) {
            dimension_map dims = dimensions.value_or(dimension_map());
            parameter_map pm = params.value_or(parameter_map());
            auto cmat = self.to_matrix(dims, pm);
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions").none() = py::none(),
          py::arg("parameters").none() = py::none(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [](const fermion_handler &self,
             std::optional<dimension_map> dimensions, py::kwargs kwargs) {
            dimension_map dims = dimensions.value_or(dimension_map());
            auto cmat =
                self.to_matrix(dims, details::kwargs_to_param_map(kwargs));
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions").none() = py::none(),
          "Returns the matrix representation of the operator.");

  py::class_<spin_handler>(mod, "SpinOperatorElement")
      .def_prop_ro("target", &spin_handler::target,
                   "Returns the degree of freedom that the operator targets.")
      .def_prop_ro("degrees", &spin_handler::degrees,
                   "Returns a vector that lists all degrees of "
                   "freedom that the operator targets.")
      .def(py::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(py::init<const spin_handler &>(), "Copy constructor.")
      .def("__eq__", &spin_handler::operator==, py::is_operator())
      .def("as_pauli", &spin_handler::as_pauli,
           "Returns the Pauli representation of the operator.")
      .def("to_string", &spin_handler::to_string, py::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [](const spin_handler &self, std::optional<dimension_map> dimensions,
             std::optional<parameter_map> params) {
            dimension_map dims = dimensions.value_or(dimension_map());
            parameter_map pm = params.value_or(parameter_map());
            auto cmat = self.to_matrix(dims, pm);
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions").none() = py::none(),
          py::arg("parameters").none() = py::none(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [](const spin_handler &self, std::optional<dimension_map> dimensions,
             py::kwargs kwargs) {
            dimension_map dims = dimensions.value_or(dimension_map());
            auto cmat =
                self.to_matrix(dims, details::kwargs_to_param_map(kwargs));
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions").none() = py::none(),
          "Returns the matrix representation of the operator.");
}

void bindHandlersWrapper(py::module_ &mod) {
  bindPauli(mod);
  bindOperatorHandlers(mod);
}

} // namespace cudaq
