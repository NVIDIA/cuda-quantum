/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/operators.h"
#include "py_handlers.h"

namespace cudaq {

void bindOperatorHandlers(py::module &mod) {
  py::class_<matrix_handler>(mod, "ElementaryMatrix")
  .def(py::init<std::size_t>(), "Creates and identity operator on the given target.")
  .def(py::init<const matrix_handler &>(),
    "Copy constructor.")
  .def("__eq__", &matrix_handler::operator==)
  .def("targets", &matrix_handler::degrees,
    "Returns a vector that lists all degrees of freedom that the operator targets.")
  .def("to_string", &matrix_handler::to_string,
    py::arg("include_degrees"),
    "Returns the string representation of the operator.")
  .def("to_matrix", &matrix_handler::to_matrix,
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(),
    "Returns the matrix representation of the operator.")
  ;

  py::class_<boson_handler>(mod, "ElementaryBoson")
  .def(py::init<std::size_t>(), "Creates and identity operator on the given target.")
  .def(py::init<const boson_handler &>(),
    "Copy constructor.")
  .def("__eq__", &boson_handler::operator==)
  .def("target", &boson_handler::target,
    "Returns the degrees of freedom that the operator targets.")
  .def("to_string", &boson_handler::to_string,
    py::arg("include_degrees"),
    "Returns the string representation of the operator.")
  .def("to_matrix", &boson_handler::to_matrix,
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(),
    "Returns the matrix representation of the operator.")
  ;

  py::class_<fermion_handler>(mod, "ElementaryFermion")
  .def(py::init<std::size_t>(), "Creates and identity operator on the given target.")
  .def(py::init<const fermion_handler &>(),
    "Copy constructor.")
  .def("__eq__", &fermion_handler::operator==)
  .def("target", &fermion_handler::target,
    "Returns the degrees of freedom that the operator targets.")
  .def("to_string", &fermion_handler::to_string,
    py::arg("include_degrees"),
    "Returns the string representation of the operator.")
  .def("to_matrix", &fermion_handler::to_matrix,
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(),
    "Returns the matrix representation of the operator.")
  ;

  py::class_<spin_handler>(mod, "ElementarySpin")
  .def(py::init<std::size_t>(), "Creates and identity operator on the given target.")
  .def(py::init<const spin_handler &>(),
    "Copy constructor.")
  .def("__eq__", &spin_handler::operator==)
  .def("as_pauli", &spin_handler::as_pauli)
  .def("target", &spin_handler::target,
    "Returns the degrees of freedom that the operator targets.")
  .def("to_string", &spin_handler::to_string,
    py::arg("include_degrees"),
    "Returns the string representation of the operator.")
  .def("to_matrix", [](const spin_handler &self, dimension_map &dimensions, const parameter_map &parameters) { 
      return self.to_matrix(dimensions, parameters); },
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(),
    "Returns the matrix representation of the operator.")
  ;
}

void bindHandlersWrapper(py::module &mod) {
  bindOperatorHandlers(mod);
  //py::implicitly_convertible<matrix_op_term, matrix_op>();
}

} // namespace cudaq