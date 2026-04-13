/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_helpers.h"
#include "py_matrix_op.h"

namespace cudaq {

void bindOperatorsModule(nb::module_ &mod) {
  // Binding the functions in `cudaq::operators` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto operators_submodule = mod.def_submodule("operators");
  operators_submodule.def(
      "empty", &matrix_op::empty,
      "Returns sum operator with no terms. Note that a sum with no terms "
      "multiplied by anything still is a sum with no terms.");
  operators_submodule.def(
      "identity", []() { return matrix_op::identity(); },
      "Returns product operator with constant value 1.");
  operators_submodule.def(
      "identity",
      [](std::size_t target) { return matrix_op::identity(target); },
      nb::arg("target"),
      "Returns an identity operator on the given target index.");
  operators_submodule.def(
      "identities",
      [](std::size_t first, std::size_t last) {
        return matrix_op_term(first, last);
      },
      nb::arg("first"), nb::arg("last"),
      "Creates a product operator that applies an identity operation to all "
      "degrees of "
      "freedom in the open range [first, last).");
  operators_submodule.def(
      "number", &matrix_op::number<matrix_handler>, nb::arg("target"),
      "Returns a number operator on the given target index.");
  operators_submodule.def(
      "parity", &matrix_op::parity<matrix_handler>, nb::arg("target"),
      "Returns a parity operator on the given target index.");
  operators_submodule.def(
      "position", &matrix_op::position<matrix_handler>, nb::arg("target"),
      "Returns a position operator on the given target index.");
  operators_submodule.def(
      "momentum", &matrix_op::momentum<matrix_handler>, nb::arg("target"),
      "Returns a momentum operator on the given target index.");
  operators_submodule.def(
      "squeeze", &matrix_op::squeeze<matrix_handler>, nb::arg("target"),
      "Returns a squeezing operator on the given target index.");
  operators_submodule.def(
      "displace", &matrix_op::displace<matrix_handler>, nb::arg("target"),
      "Returns a displacement operator on the given target index.");
  operators_submodule.def(
      "canonicalized",
      [](const matrix_op_term &orig) {
        return matrix_op_term::canonicalize(orig);
      },
      "Removes all identity operators from the operator.");
  operators_submodule.def(
      "canonicalized",
      [](const matrix_op_term &orig, const std::set<std::size_t> &degrees) {
        return matrix_op_term::canonicalize(orig, degrees);
      },
      "Expands the operator to act on all given degrees, applying identities "
      "as needed. "
      "The canonicalization will throw a runtime exception if the operator "
      "acts on any degrees "
      "of freedom that are not included in the given set.");
  operators_submodule.def(
      "canonicalized",
      [](const matrix_op &orig) { return matrix_op::canonicalize(orig); },
      "Removes all identity operators from the operator.");
  operators_submodule.def(
      "canonicalized",
      [](const matrix_op &orig, const std::set<std::size_t> &degrees) {
        return matrix_op::canonicalize(orig, degrees);
      },
      "Expands the operator to act on all given degrees, applying identities "
      "as needed. "
      "If an empty set is passed, canonicalizes all terms in the sum to act on "
      "the same "
      "degrees of freedom.");
}

void bindMatrixOperator(nb::module_ &mod) {

  auto matrix_op_class = nb::class_<matrix_op>(mod, "MatrixOperator");
  auto matrix_op_term_class =
      nb::class_<matrix_op_term>(mod, "MatrixOperatorTerm");

  matrix_op_class
      .def(
          "__iter__",
          [](matrix_op &self) {
            return nb::make_iterator(nb::type<matrix_op>(), "iterator",
                                     self.begin(), self.end());
          },
          nb::keep_alive<0, 1>(), "Loop through each term of the operator.")

      // properties

      .def_prop_ro("parameters", &matrix_op::get_parameter_descriptions,
                   "Returns a dictionary that maps each parameter "
                   "name to its description.")
      .def_prop_ro("degrees", &matrix_op::degrees,
                   "Returns a vector that lists all degrees of "
                   "freedom that the operator targets.")
      .def_prop_ro("min_degree", &matrix_op::min_degree,
                   "Returns the smallest index of the degrees of "
                   "freedom that the operator targets.")
      .def_prop_ro("max_degree", &matrix_op::max_degree,
                   "Returns the smallest index of the degrees of "
                   "freedom that the operator targets.")
      .def_prop_ro("term_count", &matrix_op::num_terms,
                   "Returns the number of terms in the operator.")

      // constructors

      .def(nb::init<>(),
           "Creates a default instantiated sum. A default instantiated "
           "sum has no value; it will take a value the first time an "
           "arithmetic operation "
           "is applied to it. In that sense, it acts as both the additive and "
           "multiplicative "
           "identity. To construct a `0` value in the mathematical sense "
           "(neutral element "
           "for addition), use `empty()` instead.")
      .def(nb::init<std::size_t>(),
           "Creates a sum operator with no terms, reserving "
           "space for the given number of terms.")
      .def(nb::init<spin_op>())
      .def(nb::init<fermion_op>())
      .def(nb::init<boson_op>())
      .def(nb::init<const matrix_op_term &>(),
           "Creates a sum operator with the given term.")
      .def(nb::init<const matrix_op &>(), "Copy constructor.")
      .def(
          "copy", [](const matrix_op &self) { return matrix_op(self); },
          "Creates a copy of the operator.")

      // evaluations

      .def(
          "to_matrix",
          [](const matrix_op &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            auto cmat = self.to_matrix(dimensions, params, invert_order);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          nb::arg("invert_order") = false,
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")

      .def(
          "to_matrix",
          [](const matrix_op &self, dimension_map &dimensions,
             bool invert_order, const nb::kwargs &kwargs) {
            auto cmat = self.to_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("invert_order") = false, nb::arg("kwargs"),
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")

      // comparisons

      .def("__eq__", &matrix_op::operator==, nb::is_operator(),
           "Return true if the two operators are equivalent. The equivalence "
           "check takes "
           "into account that addition is commutative and so is multiplication "
           "on operators "
           "that act on different degrees of freedom. Operators acting on "
           "different degrees of "
           "freedom are never equivalent, even if they only differ by an "
           "identity operator.")
      .def(
          "__eq__",
          [](const matrix_op &self, const matrix_op_term &other) {
            return self.num_terms() == 1 && *self.begin() == other;
          },
          nb::is_operator(), "Return true if the two operators are equivalent.")

      // unary operators

      .def(-nb::self, nb::is_operator())
      .def(+nb::self, nb::is_operator())

      // in-place arithmetics

      .def(nb::self /= int(), nb::is_operator())
      .def(nb::self *= int(), nb::is_operator())
      .def(nb::self += int(), nb::is_operator())
      .def(nb::self -= int(), nb::is_operator())
      .def(nb::self /= double(), nb::is_operator())
      .def(nb::self *= double(), nb::is_operator())
      .def(nb::self += double(), nb::is_operator())
      .def(nb::self -= double(), nb::is_operator())
      .def(nb::self /= std::complex<double>(), nb::is_operator())
      .def(nb::self *= std::complex<double>(), nb::is_operator())
      .def(nb::self += std::complex<double>(), nb::is_operator())
      .def(nb::self -= std::complex<double>(), nb::is_operator())
      .def(nb::self /= scalar_operator(), nb::is_operator())
      .def(nb::self *= scalar_operator(), nb::is_operator())
      .def(nb::self += scalar_operator(), nb::is_operator())
      .def(nb::self -= scalar_operator(), nb::is_operator())
      .def(nb::self *= matrix_op_term(), nb::is_operator())
      .def(nb::self += matrix_op_term(), nb::is_operator())
      .def(nb::self -= matrix_op_term(), nb::is_operator())
      .def(nb::self *= nb::self, nb::is_operator())
      .def(nb::self += nb::self, nb::is_operator())
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
      .def(nb::self -= nb::self, nb::is_operator())
#ifdef __clang__
#pragma clang diagnostic pop
#endif

      // right-hand arithmetics

      .def(nb::self / int(), nb::is_operator())
      .def(nb::self * int(), nb::is_operator())
      .def(nb::self + int(), nb::is_operator())
      .def(nb::self - int(), nb::is_operator())
      .def(nb::self / double(), nb::is_operator())
      .def(nb::self * double(), nb::is_operator())
      .def(nb::self + double(), nb::is_operator())
      .def(nb::self - double(), nb::is_operator())
      .def(nb::self / std::complex<double>(), nb::is_operator())
      .def(nb::self * std::complex<double>(), nb::is_operator())
      .def(nb::self + std::complex<double>(), nb::is_operator())
      .def(nb::self - std::complex<double>(), nb::is_operator())
      .def(nb::self / scalar_operator(), nb::is_operator())
      .def(nb::self * scalar_operator(), nb::is_operator())
      .def(nb::self + scalar_operator(), nb::is_operator())
      .def(nb::self - scalar_operator(), nb::is_operator())
      .def(nb::self * matrix_op_term(), nb::is_operator())
      .def(nb::self + matrix_op_term(), nb::is_operator())
      .def(nb::self - matrix_op_term(), nb::is_operator())
      .def(nb::self * nb::self, nb::is_operator())
      .def(nb::self + nb::self, nb::is_operator())
      .def(nb::self - nb::self, nb::is_operator())

      // left-hand arithmetics

      .def(int() * nb::self, nb::is_operator())
      .def(int() + nb::self, nb::is_operator())
      .def(int() - nb::self, nb::is_operator())
      .def(double() * nb::self, nb::is_operator())
      .def(double() + nb::self, nb::is_operator())
      .def(double() - nb::self, nb::is_operator())
      .def(std::complex<double>() * nb::self, nb::is_operator())
      .def(std::complex<double>() + nb::self, nb::is_operator())
      .def(std::complex<double>() - nb::self, nb::is_operator())
      .def(scalar_operator() * nb::self, nb::is_operator())
      .def(scalar_operator() + nb::self, nb::is_operator())
      .def(scalar_operator() - nb::self, nb::is_operator())

      // common operators

      .def_static("empty", &matrix_op::empty,
                  "Creates a sum operator with no terms. And empty sum is the "
                  "neutral element for addition; "
                  "multiplying an empty sum with anything will still result in "
                  "an empty sum.")
      .def_static(
          "identity", []() { return matrix_op::identity(); },
          "Creates a product operator with constant value 1. The identity "
          "operator is the neutral "
          "element for multiplication.")
      .def_static(
          "identity",
          [](std::size_t target) { return matrix_op::identity(target); },
          "Creates a product operator that applies the identity to the given "
          "target index.")

      // general utility functions

      .def(
          "__str__", [](const matrix_op &self) { return self.to_string(); },
          "Returns the string representation of the operator.")
      .def("dump", &matrix_op::dump,
           "Prints the string representation of the operator to the standard "
           "output.")
      .def("trim", &matrix_op::trim, nb::arg("tol") = 0.0,
           nb::arg("parameters") = parameter_map(),
           "Removes all terms from the sum for which the absolute value of the "
           "coefficient is below "
           "the given tolerance.")
      .def(
          "trim",
          [](matrix_op &self, double tol, const nb::kwargs &kwargs) {
            return self.trim(tol, details::kwargs_to_param_map(kwargs));
          },
          nb::arg("tol") = 0.0, nb::arg("kwargs"),
          "Removes all terms from the sum for which the absolute value of the "
          "coefficient is below "
          "the given tolerance.")
      .def(
          "canonicalize", [](matrix_op &self) { return self.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def(
          "canonicalize",
          [](matrix_op &self, const std::set<std::size_t> &degrees) {
            return self.canonicalize(degrees);
          },
          "Expands the operator to act on all given degrees, applying "
          "identities as needed. "
          "If an empty set is passed, canonicalizes all terms in the sum to "
          "act on the same "
          "degrees of freedom.")
      .def("distribute_terms", &matrix_op::distribute_terms,
           "Partitions the terms of the sums into the given number of separate "
           "sums.");

  matrix_op_term_class
      .def(
          "__iter__",
          [](matrix_op_term &self) {
            return nb::make_iterator(nb::type<matrix_op_term>(), "iterator",
                                     self.begin(), self.end());
          },
          nb::keep_alive<0, 1>(), "Loop through each term of the operator.")

      // properties

      .def_prop_ro("parameters", &matrix_op_term::get_parameter_descriptions,
                   "Returns a dictionary that maps each parameter "
                   "name to its description.")
      .def_prop_ro("degrees", &matrix_op_term::degrees,
                   "Returns a vector that lists all degrees of "
                   "freedom that the operator targets. "
                   "The order of degrees is from smallest to largest "
                   "and reflects the ordering of "
                   "the matrix returned by `to_matrix`. "
                   "Specifically, the indices of a statevector "
                   "with two qubits are {00, 01, 10, 11}. An "
                   "ordering of degrees {0, 1} then indicates "
                   "that a state where the qubit with index 0 equals "
                   "1 with probability 1 is given by "
                   "the vector {0., 1., 0., 0.}.")
      .def_prop_ro("min_degree", &matrix_op_term::min_degree,
                   "Returns the smallest index of the degrees of "
                   "freedom that the operator targets.")
      .def_prop_ro("max_degree", &matrix_op_term::max_degree,
                   "Returns the smallest index of the degrees of "
                   "freedom that the operator targets.")
      .def_prop_ro("ops_count", &matrix_op_term::num_ops,
                   "Returns the number of operators in the product.")
      .def_prop_ro(
          "term_id", &matrix_op_term::get_term_id,
          "The term id uniquely identifies the operators and targets (degrees) "
          "that they act on, "
          "but does not include information about the coefficient.")
      .def_prop_ro(
          "coefficient", &matrix_op_term::get_coefficient,
          "Returns the unevaluated coefficient of the operator. The "
          "coefficient is a "
          "callback function that can be invoked with the `evaluate` method.")

      // constructors

      .def(nb::init<>(),
           "Creates a product operator with constant value 1. The returned "
           "operator does not target any degrees of freedom but merely "
           "represents a constant.")
      .def(nb::init<std::size_t, std::size_t>(), nb::arg("first_degree"),
           nb::arg("last_degree"),
           "Creates a product operator that applies an identity operation to "
           "all degrees of "
           "freedom in the range [first_degree, last_degree).")
      .def(nb::init<double>(),
           "Creates a product operator with the given constant value. "
           "The returned operator does not target any degrees of freedom.")
      .def(nb::init<std::complex<double>>(),
           "Creates a product operator with the given "
           "constant value. The returned operator does not target any degrees "
           "of freedom.")
      .def(
          "__init__",
          [](matrix_op_term *self, const scalar_operator &scalar) {
            new (self) matrix_op_term(matrix_op_term() * scalar);
          },
          "Creates a product operator with non-constant scalar value.")
      .def(nb::init<matrix_handler>(),
           "Creates a product operator with the given elementary operator.")
      .def(nb::init<spin_op_term>())
      .def(nb::init<fermion_op_term>())
      .def(nb::init<boson_op_term>())
      .def(nb::init<const matrix_op_term &, std::size_t>(), nb::arg("operator"),
           nb::arg("size") = 0,
           "Creates a copy of the given operator and reserves space for "
           "storing the given "
           "number of product terms (if a size is provided).")
      .def(
          "copy",
          [](const matrix_op_term &self) { return matrix_op_term(self); },
          "Creates a copy of the operator.")

      // evaluations

      .def("evaluate_coefficient", &matrix_op_term::evaluate_coefficient,
           nb::arg("parameters") = parameter_map(),
           "Returns the evaluated coefficient of the product operator. The "
           "parameters is a map of parameter names to their concrete, complex "
           "values.")
      .def(
          "to_matrix",
          [](const matrix_op_term &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            auto cmat = self.to_matrix(dimensions, params, invert_order);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("parameters") = parameter_map(),
          nb::arg("invert_order") = false,
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_matrix",
          [](const matrix_op_term &self, dimension_map &dimensions,
             bool invert_order, const nb::kwargs &kwargs) {
            auto cmat = self.to_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
            return details::cmat_to_numpy(cmat);
          },
          nb::arg("dimensions") = dimension_map(),
          nb::arg("invert_order") = false, nb::arg("kwargs"),
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")

      // comparisons

      .def("__eq__", &matrix_op_term::operator==, nb::is_operator(),
           "Return true if the two operators are equivalent. The equivalence "
           "check takes "
           "into account that multiplication of operators that act on "
           "different degrees of "
           "is commutative. Operators acting on different degrees of freedom "
           "are never "
           "equivalent, even if they only differ by an identity operator.")
      .def(
          "__eq__",
          [](const matrix_op_term &self, const matrix_op &other) {
            return other.num_terms() == 1 && *other.begin() == self;
          },
          nb::is_operator(), "Return true if the two operators are equivalent.")

      // unary operators

      .def(-nb::self, nb::is_operator())
      .def(+nb::self, nb::is_operator())

      // in-place arithmetics

      .def(nb::self /= int(), nb::is_operator())
      .def(nb::self *= int(), nb::is_operator())
      .def(nb::self /= double(), nb::is_operator())
      .def(nb::self *= double(), nb::is_operator())
      .def(nb::self /= std::complex<double>(), nb::is_operator())
      .def(nb::self *= std::complex<double>(), nb::is_operator())
      .def(nb::self /= scalar_operator(), nb::is_operator())
      .def(nb::self *= scalar_operator(), nb::is_operator())
      .def(nb::self *= nb::self, nb::is_operator())

      // right-hand arithmetics

      .def(nb::self / int(), nb::is_operator())
      .def(nb::self * int(), nb::is_operator())
      .def(nb::self + int(), nb::is_operator())
      .def(nb::self - int(), nb::is_operator())
      .def(nb::self / double(), nb::is_operator())
      .def(nb::self * double(), nb::is_operator())
      .def(nb::self + double(), nb::is_operator())
      .def(nb::self - double(), nb::is_operator())
      .def(nb::self / std::complex<double>(), nb::is_operator())
      .def(nb::self * std::complex<double>(), nb::is_operator())
      .def(nb::self + std::complex<double>(), nb::is_operator())
      .def(nb::self - std::complex<double>(), nb::is_operator())
      .def(nb::self / scalar_operator(), nb::is_operator())
      .def(nb::self * scalar_operator(), nb::is_operator())
      .def(nb::self + scalar_operator(), nb::is_operator())
      .def(nb::self - scalar_operator(), nb::is_operator())
      .def(nb::self * nb::self, nb::is_operator())
      .def(nb::self + nb::self, nb::is_operator())
      .def(nb::self - nb::self, nb::is_operator())
      .def(nb::self * matrix_op(), nb::is_operator())
      .def(nb::self + matrix_op(), nb::is_operator())
      .def(nb::self - matrix_op(), nb::is_operator())

      // left-hand arithmetics

      .def(int() * nb::self, nb::is_operator())
      .def(int() + nb::self, nb::is_operator())
      .def(int() - nb::self, nb::is_operator())
      .def(double() * nb::self, nb::is_operator())
      .def(double() + nb::self, nb::is_operator())
      .def(double() - nb::self, nb::is_operator())
      .def(std::complex<double>() * nb::self, nb::is_operator())
      .def(std::complex<double>() + nb::self, nb::is_operator())
      .def(std::complex<double>() - nb::self, nb::is_operator())
      .def(scalar_operator() * nb::self, nb::is_operator())
      .def(scalar_operator() + nb::self, nb::is_operator())
      .def(scalar_operator() - nb::self, nb::is_operator())

      // general utility functions

      .def("is_identity", &matrix_op_term::is_identity,
           "Checks if all operators in the product are the identity. "
           "Note: this function returns true regardless of the value of the "
           "coefficient.")
      .def(
          "__str__",
          [](const matrix_op_term &self) { return self.to_string(); },
          "Returns the string representation of the operator.")
      .def("dump", &matrix_op_term::dump,
           "Prints the string representation of the operator to the standard "
           "output.")
      .def(
          "canonicalize",
          [](matrix_op_term &self) { return self.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def(
          "canonicalize",
          [](matrix_op_term &self, const std::set<std::size_t> &degrees) {
            return self.canonicalize(degrees);
          },
          "Expands the operator to act on all given degrees, applying "
          "identities as needed. "
          "The canonicalization will throw a runtime exception if the operator "
          "acts on any degrees "
          "of freedom that are not included in the given set.");
}

void bindOperatorsWrapper(nb::module_ &mod) {
  bindMatrixOperator(mod);
  nb::implicitly_convertible<double, matrix_op_term>();
  nb::implicitly_convertible<std::complex<double>, matrix_op_term>();
  nb::implicitly_convertible<scalar_operator, matrix_op_term>();
  nb::implicitly_convertible<spin_op_term, matrix_op_term>();
  nb::implicitly_convertible<spin_op, matrix_op>();
  nb::implicitly_convertible<boson_op_term, matrix_op_term>();
  nb::implicitly_convertible<boson_op, matrix_op>();
  nb::implicitly_convertible<fermion_op_term, matrix_op_term>();
  nb::implicitly_convertible<fermion_op, matrix_op>();
  nb::implicitly_convertible<matrix_op_term, matrix_op>();
  bindOperatorsModule(mod);
}

} // namespace cudaq
