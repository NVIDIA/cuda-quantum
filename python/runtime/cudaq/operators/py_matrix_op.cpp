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
#include "cudaq/operators/serialization.h"
#include "py_matrix_op.h"

namespace cudaq {

void bindOperatorsModule(py::module &mod) {
  // Binding the functions in `cudaq::operators` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto operators_submodule = mod.def_submodule("operators");
  operators_submodule.def(
      "number", &matrix_op::number<matrix_handler>, py::arg("target"),
      "Returns a number operator on the given target index.");
  operators_submodule.def(
      "parity", &matrix_op::parity<matrix_handler>, py::arg("target"),
      "Returns a parity operator on the given target index.");
  operators_submodule.def(
      "position", &matrix_op::position<matrix_handler>, py::arg("target"),
      "Returns a position operator on the given target index.");
  operators_submodule.def(
      "momentum", &matrix_op::momentum<matrix_handler>, py::arg("target"),
      "Returns a momentum operator on the given target index.");
  operators_submodule.def(
      "squeeze", &matrix_op::squeeze<matrix_handler>, py::arg("target"),
      "Returns a squeezing operator on the given target index.");
  operators_submodule.def(
      "displace", &matrix_op::displace<matrix_handler>, py::arg("target"),
      "Returns a displacement operator on the given target index.");
}

void bindMatrixOperator(py::module &mod) {
  py::class_<matrix_op>(mod, "MatrixOperator")
  .def(
    "__iter__",
    [](matrix_op &self) {
      return py::make_iterator(self.begin(), self.end());
    },
    py::keep_alive<0, 1>(),
    "Loop through each term of the operator.")
  // properties
  .def("degrees", &matrix_op::degrees,
    "Returns a vector that lists all degrees of freedom that the operator targets.")
  .def("min_degree", &matrix_op::min_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("max_degree", &matrix_op::max_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("num_terms", &matrix_op::num_terms,
    "Returns the number of terms in the operator.")
  // constructors
  .def(py::init<>(), "Creates a default instantiated sum. A default instantiated "
    "sum has no value; it will take a value the first time an arithmetic operation "
    "is applied to it. In that sense, it acts as both the additive and multiplicative "
    "identity. To construct a `0` value in the mathematical sense (neutral element "
    "for addition), use `empty()` instead.")
  .def(py::init<std::size_t>(), "Creates a sum operator with no terms, reserving "
    "space for the given number of terms.")
  .def(py::init<const matrix_op_term &>(),
    "Creates a sum operator with the given term.")
  .def(py::init<const matrix_op &>(),
    "Copy constructor.")
  // evaluations
  // todo: add to_sparse_matrix
  .def("to_matrix", &matrix_op::to_matrix,
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(), py::arg("invert_order") = false,
    "Returns the matrix representation of the operator."
    "The matrix is ordered according to the convention (endianness) "
    "used in CUDA-Q, and the ordering returned by `degrees`. This order "
    "can be inverted by setting the optional `invert_order` argument to `True`. "
    "See also the documentation for `degrees` for more detail.")
  // comparisons
  .def("__eq__", &matrix_op::operator==,
    "Return true if the two operators are equivalent. The equivalence check takes "
    "into account that addition is commutative and so is multiplication on operators "
    "that act on different degrees of freedom. Operators acting on different degrees of "
    "freedom are never equivalent, even if they only differ by an identity operator.")
  // unary operators
  .def("__neg__", [](const matrix_op &self) { return -self; })
  .def("__pos__", [](const matrix_op &self) { return +self; })
  // right-hand arithmetics
  .def("__mul__", [](const matrix_op &self, const matrix_op &other) { return self * other; })
  .def("__add__", [](const matrix_op &self, const matrix_op &other) { return self + other; })
  .def("__sub__", [](const matrix_op &self, const matrix_op &other) { return self - other; })
  .def("__mul__", [](const matrix_op &self, const matrix_op_term &other) { return self * other; })
  .def("__add__", [](const matrix_op &self, const matrix_op_term &other) { return self + other; })
  .def("__sub__", [](const matrix_op &self, const matrix_op_term &other) { return self - other; })
  .def("__imul__", [](matrix_op &self, const matrix_op &other) { return self *= other; })
  .def("__iadd__", [](matrix_op &self, const matrix_op &other) { return self += other; })
  .def("__isub__", [](matrix_op &self, const matrix_op &other) { return self -= other; })
  .def("__imul__", [](matrix_op &self, const matrix_op_term &other) { return self *= other; })
  .def("__iadd__", [](matrix_op &self, const matrix_op_term &other) { return self += other; })
  .def("__isub__", [](matrix_op &self, const matrix_op_term &other) { return self -= other; })
  // common operators
  .def_static("empty", &matrix_op::empty,
    "Creates a sum operator with no terms. And empty sum is the neutral element for addition; "
    "multiplying an empty sum with anything will still result in an empty sum.")
  .def_static("identity", []() { return matrix_op::identity(); },
    "Creates a product operator with constant value 1. The identity operator is the neutral "
    "element for multiplication.")
  .def_static("identity", [](std::size_t target) { return matrix_op::identity(target); },
    "Creates a product operator that applies the identity to the given target index.")
  // general utility functions
  .def("to_string", [](const matrix_op &self) { return self.to_string(); },
    "Returns the string representation of the operator.")
  .def("dump", &matrix_op::dump,
    "Prints the string representation of the operator to the standard output.")
  .def("trim", &matrix_op::trim,
    py::arg("tol") = 0.0, py::arg("parameters") = parameter_map(),
    "Removes all terms from the sum for which the absolute value of the coefficient is below "
    "the given tolerance.")
  .def("canonicalize", [](matrix_op &self) { return self.canonicalize(); }, // FIXME: check if this works as expected...
    "Removes all identity operators from the operator.")
  .def_static("canonicalize", [](const matrix_op &orig) { return matrix_op::canonicalize(orig); },
    "Removes all identity operators from the operator.")
  .def("canonicalize", [](matrix_op &self, const std::set<std::size_t> &degrees) { return self.canonicalize(degrees); }, // FIXME: check if this works as expected...
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "If an empty set is passed, canonicalizes all terms in the sum to act on the same "
    "degrees of freedom.")
  .def_static("canonicalize", [](const matrix_op &orig, const std::set<std::size_t> &degrees) { return matrix_op::canonicalize(orig, degrees); },
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "If an empty set is passed, canonicalizes all terms in the sum to act on the same "
    "degrees of freedom.")
  .def("distribute_terms", &matrix_op::distribute_terms,
    "Partitions the terms of the sums into the given number of separate sums.")
  ;

  py::class_<matrix_op_term>(mod, "MatrixOperatorTerm")
  /* todo: bind iterator 
  .def(
    "__iter__",
    [](matrix_op_term &self) {
      return py::make_iterator(self.begin(), self.end());
    },
    py::keep_alive<0, 1>(),
    "Loop through each term of the operator.")
  */
  // properties
  .def("degrees", &matrix_op_term::degrees,
    "Returns a vector that lists all degrees of freedom that the operator targets. "
    "The order of degrees is from smallest to largest and reflects the ordering of "
    "the matrix returned by `to_matrix`. Specifically, the indices of a statevector "
    "with two qubits are {00, 01, 10, 11}. An ordering of degrees {0, 1} then indicates "
    "that a state where the qubit with index 0 equals 1 with probability 1 is given by "
    "the vector {0., 1., 0., 0.}.")
  .def("min_degree", &matrix_op_term::min_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("max_degree", &matrix_op_term::max_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("num_ops", &matrix_op_term::num_ops,
    "Returns the number of operators in the product.")
  .def("get_term_id", &matrix_op_term::get_term_id,
    "The term id uniquely identifies the operators and targets (degrees) that they act on, "
    "but does not include information about the coefficient.")
  // todo: get_coefficient?
  // constructors
  .def(py::init<>(), "Creates a product operator with constant value 1. The returned "
    "operator does not target any degrees of freedom but merely represents a constant.")
  .def(py::init<std::size_t, std::size_t>(), 
    py::arg("first_degree"), py::arg("last_degree"),
    "Creates a product operator that applies an identity operation to all degrees of "
    "freedom in the range [first_degree, last_degree).")
  .def(py::init<double>(), "Creates a product operator with the given constant value. "
    "The returned operator does not target any degrees of freedom.")
  .def(py::init<std::complex<double>>(), "Creates a product operator with the given "
    "constant value. The returned operator does not target any degrees of freedom.")
  .def(py::init<const matrix_op_term &, std::size_t>(),
    py::arg("operator"), py::arg("size") = 0,
    "Creates a copy of the given operator and reserves space for storing the given "
    "number of product terms (if a size is provided).")
  // evaluations
  .def("evaluate_coefficient", &matrix_op_term::evaluate_coefficient,
    py::arg("parameters") = parameter_map(),
    "Returns the evaluated coefficient of the product operator.")
  // todo: add to_sparse_matrix
  .def("to_matrix", &matrix_op_term::to_matrix,
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(), py::arg("invert_order") = false,
    "Returns the matrix representation of the operator."
    "The matrix is ordered according to the convention (endianness) "
    "used in CUDA-Q, and the ordering returned by `degrees`. This order "
    "can be inverted by setting the optional `invert_order` argument to `True`. "
    "See also the documentation for `degrees` for more detail.")
  // comparisons
  .def("__eq__", &matrix_op_term::operator==,
    "Return true if the two operators are equivalent. The equivalence check takes "
    "into account that multiplication of operators that act on different degrees of "
    "is commutative. Operators acting on different degrees of freedom are never "
    "equivalent, even if they only differ by an identity operator.")
  // unary operators
  .def("__neg__", [](const matrix_op_term &self) { return -self; })
  .def("__pos__", [](const matrix_op_term &self) { return +self; })
  // right-hand arithmetics
  .def("__mul__", [](const matrix_op_term &self, const matrix_op_term &other) { return self * other; })
  .def("__add__", [](const matrix_op_term &self, const matrix_op_term &other) { return self + other; })
  .def("__sub__", [](const matrix_op_term &self, const matrix_op_term &other) { return self - other; })
  .def("__mul__", [](const matrix_op_term &self, const matrix_op &other) { return self * other; })
  .def("__add__", [](const matrix_op_term &self, const matrix_op &other) { return self + other; })
  .def("__sub__", [](const matrix_op_term &self, const matrix_op &other) { return self - other; })
  .def("__imul__", [](matrix_op_term &self, const matrix_op_term &other) { return self *= other; })
  // general utility functions
  .def("is_identity", &matrix_op_term::is_identity,
    "Checks if all operators in the product are the identity. "
    "Note: this function returns true regardless of the value of the coefficient.")
  .def("to_string", [](const matrix_op_term &self) { return self.to_string(); },
    "Returns the string representation of the operator.")
  .def("dump", &matrix_op_term::dump,
    "Prints the string representation of the operator to the standard output.")
  .def("canonicalize", [](matrix_op_term &self) { return self.canonicalize(); }, // FIXME: check if this works as expected...
    "Removes all identity operators from the operator.")
  .def_static("canonicalize", [](const matrix_op_term &orig) { return matrix_op_term::canonicalize(orig); },
    "Removes all identity operators from the operator.")
  .def("canonicalize", [](matrix_op_term &self, const std::set<std::size_t> &degrees) { return self.canonicalize(degrees); }, // FIXME: check if this works as expected...
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "The canonicalization will throw a runtime exception if the operator acts on any degrees "
    "of freedom that are not included in the given set.")
  .def_static("canonicalize", [](const matrix_op_term &orig, const std::set<std::size_t> &degrees) { return matrix_op_term::canonicalize(orig, degrees); },
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "The canonicalization will throw a runtime exception if the operator acts on any degrees "
    "of freedom that are not included in the given set.")
  ;
}

void bindOperatorsWrapper(py::module &mod) {
  bindOperatorsModule(mod);
  bindMatrixOperator(mod);
  py::implicitly_convertible<matrix_op_term, matrix_op>();
}

} // namespace cudaq