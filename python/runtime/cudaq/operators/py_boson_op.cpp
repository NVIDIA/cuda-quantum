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
#include "py_boson_op.h"

namespace cudaq {

void bindBosonModule(py::module &mod) {
  // Binding the functions in `cudaq::boson` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto boson_submodule = mod.def_submodule("boson");
  boson_submodule.def(
      "create", &boson_op::create<boson_handler>, py::arg("target"),
      "Returns a bosonic creation operator on the given target index.");
  boson_submodule.def(
      "annihilate", &boson_op::annihilate<boson_handler>, py::arg("target"),
      "Returns a bosonic annihilation operator on the given target index.");
  boson_submodule.def(
      "number", &boson_op::number<boson_handler>, py::arg("target"),
      "Returns a bosonic number operator on the given target index.");
  boson_submodule.def(
      "position", &boson_op::position<boson_handler>, py::arg("target"),
      "Returns a bosonic position operator on the given target index.");
  boson_submodule.def(
      "momentum", &boson_op::momentum<boson_handler>, py::arg("target"),
      "Returns a bosonic momentum operator on the given target index.");
}

void bindBosonOperator(py::module &mod) {
  py::class_<boson_op>(mod, "BosonOperator")
  .def(
    "__iter__",
    [](boson_op &self) {
      return py::make_iterator(self.begin(), self.end());
    },
    py::keep_alive<0, 1>(),
    "Loop through each term of the operator.")
  // properties
  .def("degrees", &boson_op::degrees,
    "Returns a vector that lists all degrees of freedom that the operator targets.")
  .def("min_degree", &boson_op::min_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("max_degree", &boson_op::max_degree,
    "Returns the smallest index of the degrees of freedom that the operator targets.")
  .def("num_terms", &boson_op::num_terms,
    "Returns the number of terms in the operator.")
  // constructors
  .def(py::init<>(), "Creates a default instantiated sum. A default instantiated "
    "sum has no value; it will take a value the first time an arithmetic operation "
    "is applied to it. In that sense, it acts as both the additive and multiplicative "
    "identity. To construct a `0` value in the mathematical sense (neutral element "
    "for addition), use `empty()` instead.")
  .def(py::init<std::size_t>(), "Creates a sum operator with no terms, reserving "
    "space for the given number of terms.")
  .def(py::init<const boson_op_term &>(),
    "Creates a sum operator with the given term.")
  .def(py::init<const boson_op &>(),
    "Copy constructor.")
  // evaluations
  // todo: add to_sparse_matrix
  .def("to_matrix", &boson_op::to_matrix,
    py::arg("dimensions") = dimension_map(), py::arg("parameters") = parameter_map(), py::arg("invert_order") = false,
    "Returns the matrix representation of the operator."
    "The matrix is ordered according to the convention (endianness) "
    "used in CUDA-Q, and the ordering returned by `degrees`. This order "
    "can be inverted by setting the optional `invert_order` argument to `True`. "
    "See also the documentation for `degrees` for more detail.")
  // comparisons
  .def("__eq__", &boson_op::operator==,
    "Return true if the two operators are equivalent. The equivalence check takes "
    "commutation relations into account. Operators acting on different degrees of "
    "freedom are never equivalent, even if they only differ by an identity operator.")
  // unary operators
  .def("__neg__", [](const boson_op &self) { return -self; })
  .def("__pos__", [](const boson_op &self) { return +self; })
  // right-hand arithmetics
  .def("__mul__", [](const boson_op &self, const boson_op &other) { return self * other; })
  .def("__add__", [](const boson_op &self, const boson_op &other) { return self + other; })
  .def("__sub__", [](const boson_op &self, const boson_op &other) { return self - other; })
  .def("__mul__", [](const boson_op &self, const boson_op_term &other) { return self * other; })
  .def("__add__", [](const boson_op &self, const boson_op_term &other) { return self + other; })
  .def("__sub__", [](const boson_op &self, const boson_op_term &other) { return self - other; })
  .def("__imul__", [](boson_op &self, const boson_op &other) { return self *= other; })
  .def("__iadd__", [](boson_op &self, const boson_op &other) { return self += other; })
  .def("__isub__", [](boson_op &self, const boson_op &other) { return self -= other; })
  .def("__imul__", [](boson_op &self, const boson_op_term &other) { return self *= other; })
  .def("__iadd__", [](boson_op &self, const boson_op_term &other) { return self += other; })
  .def("__isub__", [](boson_op &self, const boson_op_term &other) { return self -= other; })
  // left-hand arithmetics
  //.def("__rmul__", [](const boson_op &other, const boson_op &self) { return self *= other; })
  //.def("__radd__", [](const boson_op &other, const boson_op &self) { return self += other; })
  //.def("__rsub__", [](const boson_op &other, const boson_op &self) { return self -= other; })
  // common operators
  .def_static("empty", &boson_op::empty,
    "Creates a sum operator with no terms. And empty sum is the neutral element for addition; "
    "multiplying an empty sum with anything will still result in an empty sum.")
  .def_static("identity", []() { return boson_op::identity(); },
    "Creates a product operator with constant value 1. The identity operator is the neutral "
    "element for multiplication.")
  .def_static("identity", [](std::size_t target) { return boson_op::identity(target); },
    "Creates a product operator that applies the identity to the given target index.")
  // general utility functions
  .def("to_string", [](const boson_op &self) { return self.to_string(); },
    "Returns the string representation of the operator.")
  .def("dump", &boson_op::dump,
    "Prints the string representation of the operator to the standard output.")
  .def("trim", &boson_op::trim,
    py::arg("tol") = 0.0, py::arg("parameters") = parameter_map(),
    "Removes all terms from the sum for which the absolute value of the coefficient is below "
    "the given tolerance.")
  .def("canonicalize", [](boson_op &self) { return self.canonicalize(); }, // FIXME: check if this works as expected...
    "Removes all identity operators from the operator.")
  .def_static("canonicalize", [](const boson_op &orig) { return boson_op::canonicalize(orig); },
    "Removes all identity operators from the operator.")
  .def("canonicalize", [](boson_op &self, const std::set<std::size_t> &degrees) { return self.canonicalize(degrees); }, // FIXME: check if this works as expected...
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "If an empty set is passed, canonicalizes all terms in the sum to act on the same "
    "degrees of freedom.")
  .def_static("canonicalize", [](const boson_op &orig, const std::set<std::size_t> &degrees) { return boson_op::canonicalize(orig, degrees); },
    "Expands the operator to act on all given degrees, applying identities as needed. "
    "If an empty set is passed, canonicalizes all terms in the sum to act on the same "
    "degrees of freedom.")
  .def("distribute_terms", &boson_op::distribute_terms,
    "Partitions the terms of the sums into the given number of separate sums.")

  ;
}

void bindBosonWrapper(py::module &mod) {
  bindBosonModule(mod);
  bindBosonOperator(mod);
  py::implicitly_convertible<boson_op_term, boson_op>();
}

} // namespace cudaq