/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_fermion_op.h"

namespace cudaq {

void bindFermionModule(py::module &mod) {
  // Binding the functions in `cudaq::fermion` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto fermion_submodule = mod.def_submodule("fermion");
  fermion_submodule.def(
      "empty", &fermion_op::empty,
      "Returns sum operator with no terms. Note that a sum with no terms "
      "multiplied by anything still is a sum with no terms.");
  fermion_submodule.def(
      "identity", []() { return fermion_op::identity(); },
      "Returns product operator with constant value 1.");
  fermion_submodule.def(
      "identity", [](std::size_t target) { return fermion_op::identity(target); }, py::arg("target"),
      "Returns an identity operator on the given target index.");
  fermion_submodule.def(
      "create", &fermion_op::create<fermion_handler>, py::arg("target"),
      "Returns a fermionic creation operator on the given target index.");
  fermion_submodule.def(
      "annihilate", &fermion_op::annihilate<fermion_handler>, py::arg("target"),
      "Returns a fermionic annihilation operator on the given target index.");
  fermion_submodule.def(
      "number", &fermion_op::number<fermion_handler>, py::arg("target"),
      "Returns a fermionic number operator on the given target index.");
  fermion_submodule.def("canonicalized", [](const fermion_op_term &orig) { return fermion_op_term::canonicalize(orig); },
    "Removes all identity operators from the operator.");
  fermion_submodule.def("canonicalized", [](const fermion_op_term &orig, const std::set<std::size_t> &degrees) { return fermion_op_term::canonicalize(orig, degrees); },
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "The canonicalization will throw a runtime exception if the operator acts on any degrees "
    "of freedom that are not included in the given set.");
  fermion_submodule.def("canonicalized", [](const fermion_op &orig) { return fermion_op::canonicalize(orig); },
    "Removes all identity operators from the operator.");
  fermion_submodule.def("canonicalized", [](const fermion_op &orig, const std::set<std::size_t> &degrees) { return fermion_op::canonicalize(orig, degrees); },
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "If an empty set is passed, canonicalizes all terms in the sum to act on the same "
    "degrees of freedom.");
}

void bindFermionOperator(py::module &mod) {
  auto cmat_to_numpy = [](const complex_matrix &m) {
    std::vector<ssize_t> shape = {static_cast<ssize_t>(m.rows()),
                                  static_cast<ssize_t>(m.cols())};
    std::vector<ssize_t> strides = {
      static_cast<ssize_t>(sizeof(std::complex<double>) * m.cols()),
      static_cast<ssize_t>(sizeof(std::complex<double>))};

    // Return a numpy array without copying data
    return py::array_t<std::complex<double>>(shape, strides, m.data);
  };

  py::class_<fermion_op>(mod, "FermionOperator")
  .def(
    "__iter__",
    [](fermion_op &self) {
    return py::make_iterator(self.begin(), self.end());
    },
    py::keep_alive<0, 1>(),
    "Loop through each term of the operator.")
  // properties
  // todo: add a target property?      
  .def("degrees", &fermion_op::degrees,
    "Returns a vector that lists all degrees of freedom that the operator targets. "
    "The order of degrees is from smallest to largest and reflects the ordering of "
    "the matrix returned by `to_matrix`. Specifically, the indices of a statevector "
    "with two qubits are {00, 01, 10, 11}. An ordering of degrees {0, 1} then indicates "
    "that a state where the qubit with index 0 equals 1 with probability 1 is given by "
    "the vector {0., 1., 0., 0.}.")
  .def("min_degree", &fermion_op::min_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("max_degree", &fermion_op::max_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("num_terms", &fermion_op::num_terms,
    "Returns the number of terms in the operator.")
  // constructors
  .def(py::init<>(), "Creates a default instantiated sum. A default instantiated "
    "sum has no value; it will take a value the first time an arithmetic operation "
    "is applied to it. In that sense, it acts as both the additive and multiplicative "
    "identity. To construct a `0` value in the mathematical sense (neutral element "
    "for addition), use `empty()` instead.")
  .def(py::init<std::size_t>(), "Creates a sum operator with no terms, reserving "
    "space for the given number of terms.")
  .def(py::init<const fermion_op_term &>(),
    "Creates a sum operator with the given term.")
  .def(py::init<const fermion_op &>(),
    "Copy constructor.")
  .def("copy", [](const fermion_op &self) { return fermion_op(self); },
    "Creates a copy of the operator.")
  // evaluations
  // todo: add to_sparse_matrix
  .def("to_matrix", [&cmat_to_numpy](const fermion_op &self,
                                     dimension_map &dimensions,
                                     const parameter_map &params,
                                     bool invert_order) {
      return cmat_to_numpy(self.to_matrix(dimensions, params, invert_order));
    },
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(), py::arg("invert_order") = false,
    "Returns the matrix representation of the operator."
    "The matrix is ordered according to the convention (endianness) "
    "used in CUDA-Q, and the ordering returned by `degrees`. This order "
    "can be inverted by setting the optional `invert_order` argument to `True`. "
    "See also the documentation for `degrees` for more detail.")
  // comparisons
  .def("__eq__", &fermion_op::operator==,
    "Return true if the two operators are equivalent. The equivalence check takes "
    "commutation relations into account. Operators acting on different degrees of "
    "freedom are never equivalent, even if they only differ by an identity operator.")
  // unary operators
  .def("__neg__", [](const fermion_op &self) { return -self; })
  .def("__pos__", [](const fermion_op &self) { return +self; })
  // right-hand arithmetics
  .def("__mul__", [](const fermion_op &self, const fermion_op &other) { return self * other; })
  .def("__add__", [](const fermion_op &self, const fermion_op &other) { return self + other; })
  .def("__sub__", [](const fermion_op &self, const fermion_op &other) { return self - other; })
  .def("__mul__", [](const fermion_op &self, const fermion_op_term &other) { return self * other; })
  .def("__add__", [](const fermion_op &self, const fermion_op_term &other) { return self + other; })
  .def("__sub__", [](const fermion_op &self, const fermion_op_term &other) { return self - other; })
  .def("__imul__", [](fermion_op &self, const fermion_op &other) { return self *= other; })
  .def("__iadd__", [](fermion_op &self, const fermion_op &other) { return self += other; })
  .def("__isub__", [](fermion_op &self, const fermion_op &other) { return self -= other; })
  .def("__imul__", [](fermion_op &self, const fermion_op_term &other) { return self *= other; })
  .def("__iadd__", [](fermion_op &self, const fermion_op_term &other) { return self += other; })
  .def("__isub__", [](fermion_op &self, const fermion_op_term &other) { return self -= other; })
  // common operators
  .def_static("empty", &fermion_op::empty,
    "Creates a sum operator with no terms. And empty sum is the neutral element for addition; "
    "multiplying an empty sum with anything will still result in an empty sum.")
  .def_static("identity", []() { return fermion_op::identity(); },
    "Creates a product operator with constant value 1. The identity operator is the neutral "
    "element for multiplication.")
  .def_static("identity", [](std::size_t target) { return fermion_op::identity(target); },
    "Creates a product operator that applies the identity to the given target index.")
  // general utility functions
  .def("to_string", [](const fermion_op &self) { return self.to_string(); },
    "Returns the string representation of the operator.")
  .def("dump", &fermion_op::dump,
    "Prints the string representation of the operator to the standard output.")
  .def("trim", &fermion_op::trim,
    py::arg("tol") = 0.0, py::arg("parameters") = parameter_map(),
    "Removes all terms from the sum for which the absolute value of the coefficient is below "
    "the given tolerance.")
  .def("canonicalize", [](fermion_op &self) { return self.canonicalize(); }, // FIXME: check if this works as expected...
    "Removes all identity operators from the operator.")
  .def("canonicalize", [](fermion_op &self, const std::set<std::size_t> &degrees) { return self.canonicalize(degrees); }, // FIXME: check if this works as expected...
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "If an empty set is passed, canonicalizes all terms in the sum to act on the same "
    "degrees of freedom.")
  .def("distribute_terms", &fermion_op::distribute_terms,
    "Partitions the terms of the sums into the given number of separate sums.")
  ;

  py::class_<fermion_op_term>(mod, "FermionOperatorTerm")
  .def(
    "__iter__",
    [](fermion_op_term &self) {
      return py::make_iterator(self.begin(), self.end());
    },
    py::keep_alive<0, 1>(),
    "Loop through each term of the operator.")
  // properties
  .def("degrees", &fermion_op_term::degrees,
    "Returns a vector that lists all degrees of freedom that the operator targets. "
    "The order of degrees is from smallest to largest and reflects the ordering of "
    "the matrix returned by `to_matrix`. Specifically, the indices of a statevector "
    "with two qubits are {00, 01, 10, 11}. An ordering of degrees {0, 1} then indicates "
    "that a state where the qubit with index 0 equals 1 with probability 1 is given by "
    "the vector {0., 1., 0., 0.}.")
  .def("min_degree", &fermion_op_term::min_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("max_degree", &fermion_op_term::max_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("num_ops", &fermion_op_term::num_ops,
    "Returns the number of operators in the product.")
  .def("get_term_id", &fermion_op_term::get_term_id,
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
  .def(py::init<const fermion_op_term &, std::size_t>(),
    py::arg("operator"), py::arg("size") = 0,
    "Creates a copy of the given operator and reserves space for storing the given "
    "number of product terms (if a size is provided).")
  .def("copy", [](const fermion_op_term &self) { return fermion_op_term(self); },
    "Creates a copy of the operator.")
  // evaluations
  .def("evaluate_coefficient", &fermion_op_term::evaluate_coefficient,
    py::arg("parameters") = parameter_map(),
    "Returns the evaluated coefficient of the product operator.")
  // todo: add to_sparse_matrix
  .def("to_matrix", [&cmat_to_numpy](const fermion_op_term &self,
                                     dimension_map &dimensions,
                                     const parameter_map &params,
                                     bool invert_order) {
      return cmat_to_numpy(self.to_matrix(dimensions, params, invert_order));
    },
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(), py::arg("invert_order") = false,
    "Returns the matrix representation of the operator."
    "The matrix is ordered according to the convention (endianness) "
    "used in CUDA-Q, and the ordering returned by `degrees`. This order "
    "can be inverted by setting the optional `invert_order` argument to `True`. "
    "See also the documentation for `degrees` for more detail.")
  // comparisons
  .def("__eq__", &fermion_op_term::operator==,
    "Return true if the two operators are equivalent. The equivalence check takes "
    "commutation relations into account. Operators acting on different degrees of "
    "freedom are never equivalent, even if they only differ by an identity operator.")
  // unary operators
  .def("__neg__", [](const fermion_op_term &self) { return -self; })
  .def("__pos__", [](const fermion_op_term &self) { return +self; })
  // right-hand arithmetics
  .def("__mul__", [](const fermion_op_term &self, const fermion_op_term &other) { return self * other; })
  .def("__add__", [](const fermion_op_term &self, const fermion_op_term &other) { return self + other; })
  .def("__sub__", [](const fermion_op_term &self, const fermion_op_term &other) { return self - other; })
  .def("__mul__", [](const fermion_op_term &self, const fermion_op &other) { return self * other; })
  .def("__add__", [](const fermion_op_term &self, const fermion_op &other) { return self + other; })
  .def("__sub__", [](const fermion_op_term &self, const fermion_op &other) { return self - other; })
  .def("__imul__", [](fermion_op_term &self, const fermion_op_term &other) { return self *= other; })
  // general utility functions
  .def("is_identity", &fermion_op_term::is_identity,
    "Checks if all operators in the product are the identity. "
    "Note: this function returns true regardless of the value of the coefficient.")
  .def("to_string", [](const fermion_op_term &self) { return self.to_string(); },
    "Returns the string representation of the operator.")
  .def("dump", &fermion_op_term::dump,
    "Prints the string representation of the operator to the standard output.")
  .def("canonicalize", [](fermion_op_term &self) { return self.canonicalize(); }, // FIXME: check if this works as expected...
    "Removes all identity operators from the operator.")
  .def("canonicalize", [](fermion_op_term &self, const std::set<std::size_t> &degrees) { return self.canonicalize(degrees); }, // FIXME: check if this works as expected...
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "The canonicalization will throw a runtime exception if the operator acts on any degrees "
    "of freedom that are not included in the given set.")
  ;
}
  
void bindFermionWrapper(py::module &mod) {
  bindFermionModule(mod);
  bindFermionOperator(mod);
  py::implicitly_convertible<fermion_op_term, fermion_op>();
  py::implicitly_convertible<double, fermion_op_term>();
}

} // namespace cudaq