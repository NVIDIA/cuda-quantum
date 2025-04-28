/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/operators.h"
#include "py_handlers.h"

namespace cudaq {

void bindPauli(py::module mod) {
  py::enum_<pauli>(mod, "Pauli",
                   "An enumeration representing the types of Pauli matrices.")
      .value("X", pauli::X)
      .value("Y", pauli::Y)
      .value("Z", pauli::Z)
      .value("I", pauli::I);
}

void bindOperatorHandlers(py::module &mod) {
  using matrix_callback = std::function<complex_matrix(
      const std::vector<int64_t> &, const parameter_map &)>;

  auto cmat_to_numpy = [](const complex_matrix &m) {
    std::vector<ssize_t> shape = {static_cast<ssize_t>(m.rows()),
                                  static_cast<ssize_t>(m.cols())};
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(std::complex<double>) * m.cols()),
        static_cast<ssize_t>(sizeof(std::complex<double>))};

    // Return a numpy array without copying data
    return py::array_t<std::complex<double>>(shape, strides, m.data);
  };

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

  py::class_<matrix_handler>(mod, "MatrixOperatorElement")
      .def_property_readonly(
          "id",
          [](const matrix_handler &self) { return self.to_string(false); },
          "Returns the id used to define and instantiate the operator.")
      .def_property_readonly("degrees", &matrix_handler::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("parameters",
                             &matrix_handler::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")
      .def_property_readonly("expected_dimensions",
                             &matrix_handler::get_expected_dimensions,
                             "The number of levels, that is the dimension, for "
                             "each degree of freedom "
                             "in canonical order that the operator acts on. A "
                             "value of zero or less "
                             "indicates that the operator is defined for any "
                             "dimension of that degree.")
      .def(py::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(py::init([](std::string operator_id,
                       std::vector<std::size_t> degrees) {
             return matrix_handler(
                 std::move(operator_id),
                 std::move(
                     degrees)); // FIXME: BIND AND SUPPORT COMMUTATION BEHAVIOR
           }),
           py::arg("id"), py::arg("degrees"),
           "Creates the matrix operator with the given id acting on the given "
           "degrees of "
           "freedom. Throws a runtime exception if no operator with that id "
           "has been defined.")
      .def(py::init<const matrix_handler &>(), "Copy constructor.")
      .def("__eq__", &matrix_handler::operator==)
      .def("to_string", &matrix_handler::to_string, py::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [&cmat_to_numpy](const matrix_handler &self,
                           dimension_map &dimensions,
                           const parameter_map &params) {
            return cmat_to_numpy(self.to_matrix(dimensions, params));
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("parameters") = parameter_map(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [&cmat_to_numpy, &kwargs_to_param_map](const matrix_handler &self,
                                                 dimension_map &dimensions,
                                                 const py::kwargs &kwargs) {
            return cmat_to_numpy(
                self.to_matrix(dimensions, kwargs_to_param_map(kwargs)));
          },
          py::arg("dimensions") = dimension_map(),
          "Returns the matrix representation of the operator.")

      // tools for custom operators
      .def_static(
          "_define",
          [&kwargs_to_param_description](
              std::string operator_id, std::vector<int64_t> expected_dimensions,
              const matrix_callback &func, bool overwrite,
              const py::kwargs &kwargs) {
            // we need to make sure the python function that is stored in
            // the static dictionary containing the operator definitions
            // is properly cleaned up - otherwise python will hang on exit...
            auto atexit = py::module_::import("atexit");
            atexit.attr("register")(py::cpp_function([operator_id]() {
              matrix_handler::remove_definition(operator_id);
            }));
            if (overwrite)
              matrix_handler::remove_definition(operator_id);
            matrix_handler::define(std::move(operator_id),
                                   std::move(expected_dimensions), func,
                                   kwargs_to_param_description(kwargs));
          },
          py::arg("operator_id"), py::arg("expected_dimensions"),
          py::arg("callback"), py::arg("overwrite") = false,
          "Defines a matrix operator with the given name and dimensions whose"
          "matrix representation can be obtained by invoking the given "
          "callback function.");

  py::class_<boson_handler>(mod, "BosonOperatorElement")
      .def_property_readonly(
          "target", &boson_handler::target,
          "Returns the degree of freedom that the operator targets.")
      .def_property_readonly("degrees", &boson_handler::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets.")
      .def(py::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(py::init<const boson_handler &>(), "Copy constructor.")
      .def("__eq__", &boson_handler::operator==)
      .def("to_string", &boson_handler::to_string, py::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [&cmat_to_numpy](const boson_handler &self, dimension_map &dimensions,
                           const parameter_map &params) {
            return cmat_to_numpy(self.to_matrix(dimensions, params));
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("parameters") = parameter_map(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [&cmat_to_numpy, &kwargs_to_param_map](const boson_handler &self,
                                                 dimension_map &dimensions,
                                                 const py::kwargs &kwargs) {
            return cmat_to_numpy(
                self.to_matrix(dimensions, kwargs_to_param_map(kwargs)));
          },
          py::arg("dimensions") = dimension_map(),
          "Returns the matrix representation of the operator.");

  py::class_<fermion_handler>(mod, "FermionOperatorElement")
      .def_property_readonly(
          "target", &fermion_handler::target,
          "Returns the degree of freedom that the operator targets.")
      .def_property_readonly("degrees", &fermion_handler::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets.")
      .def(py::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(py::init<const fermion_handler &>(), "Copy constructor.")
      .def("__eq__", &fermion_handler::operator==)
      .def("to_string", &fermion_handler::to_string, py::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [&cmat_to_numpy](const fermion_handler &self,
                           dimension_map &dimensions,
                           const parameter_map &params) {
            return cmat_to_numpy(self.to_matrix(dimensions, params));
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("parameters") = parameter_map(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [&cmat_to_numpy, &kwargs_to_param_map](const fermion_handler &self,
                                                 dimension_map &dimensions,
                                                 const py::kwargs &kwargs) {
            return cmat_to_numpy(
                self.to_matrix(dimensions, kwargs_to_param_map(kwargs)));
          },
          py::arg("dimensions") = dimension_map(),
          "Returns the matrix representation of the operator.");

  py::class_<spin_handler>(mod, "SpinOperatorElement")
      .def_property_readonly(
          "target", &spin_handler::target,
          "Returns the degree of freedom that the operator targets.")
      .def_property_readonly("degrees", &spin_handler::degrees,
                             "Returns a vector that lists all degrees of "
                             "freedom that the operator targets.")
      .def(py::init<std::size_t>(),
           "Creates an identity operator on the given target.")
      .def(py::init<const spin_handler &>(), "Copy constructor.")
      .def("__eq__", &spin_handler::operator==)
      .def("as_pauli", &spin_handler::as_pauli)
      .def("to_string", &spin_handler::to_string, py::arg("include_degrees"),
           "Returns the string representation of the operator.")
      .def(
          "to_matrix",
          [&cmat_to_numpy](const spin_handler &self, dimension_map &dimensions,
                           const parameter_map &params) {
            return cmat_to_numpy(self.to_matrix(dimensions, params));
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("parameters") = parameter_map(),
          "Returns the matrix representation of the operator.")
      .def(
          "to_matrix",
          [&cmat_to_numpy, &kwargs_to_param_map](const spin_handler &self,
                                                 dimension_map &dimensions,
                                                 const py::kwargs &kwargs) {
            return cmat_to_numpy(
                self.to_matrix(dimensions, kwargs_to_param_map(kwargs)));
          },
          py::arg("dimensions") = dimension_map(),
          "Returns the matrix representation of the operator.");
}

void bindHandlersWrapper(py::module &mod) {
  bindPauli(mod);
  bindOperatorHandlers(mod);
}

} // namespace cudaq
