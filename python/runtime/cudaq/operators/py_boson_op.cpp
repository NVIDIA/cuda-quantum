/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "py_boson_op.h"
#include "py_helpers.h"

namespace cudaq {

void bindBosonModule(py::module &mod) {
  // Binding the functions in `cudaq::boson` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto boson_submodule = mod.def_submodule("boson");
  boson_submodule.def(
      "empty", &boson_op::empty,
      "Returns sum operator with no terms. Note that a sum with no terms "
      "multiplied by anything still is a sum with no terms.");
  boson_submodule.def(
      "identity", []() { return boson_op::identity(); },
      "Returns product operator with constant value 1.");
  boson_submodule.def(
      "identity", [](std::size_t target) { return boson_op::identity(target); },
      py::arg("target"),
      "Returns an identity operator on the given target index.");
  boson_submodule.def(
      "identities",
      [](std::size_t first, std::size_t last) {
        return boson_op_term(first, last);
      },
      py::arg("first"), py::arg("last"),
      "Creates a product operator that applies an identity operation to all "
      "degrees of "
      "freedom in the open range [first, last).");
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
  boson_submodule.def(
      "canonicalized",
      [](const boson_op_term &orig) {
        return boson_op_term::canonicalize(orig);
      },
      "Removes all identity operators from the operator.");
  boson_submodule.def(
      "canonicalized",
      [](const boson_op_term &orig, const std::set<std::size_t> &degrees) {
        return boson_op_term::canonicalize(orig, degrees);
      },
      "Expands the operator to act on all given degrees, applying identities "
      "as needed. "
      "The canonicalization will throw a runtime exception if the operator "
      "acts on any degrees "
      "of freedom that are not included in the given set.");
  boson_submodule.def(
      "canonicalized",
      [](const boson_op &orig) { return boson_op::canonicalize(orig); },
      "Removes all identity operators from the operator.");
  boson_submodule.def(
      "canonicalized",
      [](const boson_op &orig, const std::set<std::size_t> &degrees) {
        return boson_op::canonicalize(orig, degrees);
      },
      "Expands the operator to act on all given degrees, applying identities "
      "as needed. "
      "If an empty set is passed, canonicalizes all terms in the sum to act on "
      "the same "
      "degrees of freedom.");
}

void bindBosonOperator(py::module &mod) {

  auto boson_op_class = py::class_<boson_op>(mod, "BosonOperator");
  auto boson_op_term_class =
      py::class_<boson_op_term>(mod, "BosonOperatorTerm");

  boson_op_class
      .def(
          "__iter__",
          [](boson_op &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(), "Loop through each term of the operator.")

      // properties

      .def_property_readonly("parameters",
                             &boson_op::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")
      .def_property_readonly("degrees", &boson_op::degrees,
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
      .def_property_readonly("min_degree", &boson_op::min_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("max_degree", &boson_op::max_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("term_count", &boson_op::num_terms,
                             "Returns the number of terms in the operator.")

      // constructors

      .def(py::init<>(),
           "Creates a default instantiated sum. A default instantiated "
           "sum has no value; it will take a value the first time an "
           "arithmetic operation "
           "is applied to it. In that sense, it acts as both the additive and "
           "multiplicative "
           "identity. To construct a `0` value in the mathematical sense "
           "(neutral element "
           "for addition), use `empty()` instead.")
      .def(py::init<std::size_t>(),
           "Creates a sum operator with no terms, reserving "
           "space for the given number of terms.")
      .def(py::init<const boson_op_term &>(),
           "Creates a sum operator with the given term.")
      .def(py::init<const boson_op &>(), "Copy constructor.")
      .def(
          "copy", [](const boson_op &self) { return boson_op(self); },
          "Creates a copy of the operator.")

      // evaluations

      .def(
          "to_matrix",
          [](const boson_op &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            auto cmat = self.to_matrix(dimensions, params, invert_order);
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("parameters") = parameter_map(),
          py::arg("invert_order") = false,
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_matrix",
          [](const boson_op &self, dimension_map &dimensions, bool invert_order,
             const py::kwargs &kwargs) {
            auto cmat = self.to_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("invert_order") = false,
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_sparse_matrix",
          [](const boson_op &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            return self.to_sparse_matrix(dimensions, params, invert_order);
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("parameters") = parameter_map(),
          py::arg("invert_order") = false,
          "Return the sparse matrix representation of the operator. This "
          "representation is a "
          "`Tuple[list[complex], list[int], list[int]]`, encoding the "
          "non-zero values, rows, and columns of the matrix. "
          "This format is supported by `scipy.sparse.csr_array`."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_sparse_matrix",
          [](const boson_op &self, dimension_map &dimensions, bool invert_order,
             const py::kwargs &kwargs) {
            return self.to_sparse_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("invert_order") = false,
          "Return the sparse matrix representation of the operator. This "
          "representation is a "
          "`Tuple[list[complex], list[int], list[int]]`, encoding the "
          "non-zero values, rows, and columns of the matrix. "
          "This format is supported by `scipy.sparse.csr_array`."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")

      // comparisons

      .def("__eq__", &boson_op::operator==, py::is_operator(),
           "Return true if the two operators are equivalent. The equivalence "
           "check takes "
           "commutation relations into account. Operators acting on different "
           "degrees of "
           "freedom are never equivalent, even if they only differ by an "
           "identity operator.")
      .def(
          "__eq__",
          [](const boson_op &self, const boson_op_term &other) {
            return self.num_terms() == 1 && *self.begin() == other;
          },
          py::is_operator(), "Return true if the two operators are equivalent.")

      // unary operators

      .def(-py::self, py::is_operator())
      .def(+py::self, py::is_operator())

      // in-place arithmetics

      .def(py::self /= int(), py::is_operator())
      .def(py::self *= int(), py::is_operator())
      .def(py::self += int(), py::is_operator())
      .def(py::self -= int(), py::is_operator())
      .def(py::self /= double(), py::is_operator())
      .def(py::self *= double(), py::is_operator())
      .def(py::self += double(), py::is_operator())
      .def(py::self -= double(), py::is_operator())
      .def(py::self /= std::complex<double>(), py::is_operator())
      .def(py::self *= std::complex<double>(), py::is_operator())
      .def(py::self += std::complex<double>(), py::is_operator())
      .def(py::self -= std::complex<double>(), py::is_operator())
      .def(py::self /= scalar_operator(), py::is_operator())
      .def(py::self *= scalar_operator(), py::is_operator())
      .def(py::self += scalar_operator(), py::is_operator())
      .def(py::self -= scalar_operator(), py::is_operator())
      .def(py::self *= boson_op_term(), py::is_operator())
      .def(py::self += boson_op_term(), py::is_operator())
      .def(py::self -= boson_op_term(), py::is_operator())
      .def(py::self *= py::self, py::is_operator())
      .def(py::self += py::self, py::is_operator())
// see issue https://github.com/pybind/pybind11/issues/1893
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
      .def(py::self -= py::self, py::is_operator())
#ifdef __clang__
#pragma clang diagnostic pop
#endif

      // right-hand arithmetics

      .def(py::self / int(), py::is_operator())
      .def(py::self * int(), py::is_operator())
      .def(py::self + int(), py::is_operator())
      .def(py::self - int(), py::is_operator())
      .def(py::self / double(), py::is_operator())
      .def(py::self * double(), py::is_operator())
      .def(py::self + double(), py::is_operator())
      .def(py::self - double(), py::is_operator())
      .def(py::self / std::complex<double>(), py::is_operator())
      .def(py::self * std::complex<double>(), py::is_operator())
      .def(py::self + std::complex<double>(), py::is_operator())
      .def(py::self - std::complex<double>(), py::is_operator())
      .def(py::self / scalar_operator(), py::is_operator())
      .def(py::self * scalar_operator(), py::is_operator())
      .def(py::self + scalar_operator(), py::is_operator())
      .def(py::self - scalar_operator(), py::is_operator())
      .def(py::self * boson_op_term(), py::is_operator())
      .def(py::self + boson_op_term(), py::is_operator())
      .def(py::self - boson_op_term(), py::is_operator())
      .def(py::self * py::self, py::is_operator())
      .def(py::self + py::self, py::is_operator())
      .def(py::self - py::self, py::is_operator())
      .def(py::self * matrix_op_term(), py::is_operator())
      .def(py::self + matrix_op_term(), py::is_operator())
      .def(py::self - matrix_op_term(), py::is_operator())
      .def(py::self * matrix_op(), py::is_operator())
      .def(py::self + matrix_op(), py::is_operator())
      .def(py::self - matrix_op(), py::is_operator())

      // left-hand arithmetics

      .def(int() * py::self, py::is_operator())
      .def(int() + py::self, py::is_operator())
      .def(int() - py::self, py::is_operator())
      .def(double() * py::self, py::is_operator())
      .def(double() + py::self, py::is_operator())
      .def(double() - py::self, py::is_operator())
      .def(std::complex<double>() * py::self, py::is_operator())
      .def(std::complex<double>() + py::self, py::is_operator())
      .def(std::complex<double>() - py::self, py::is_operator())
      .def(scalar_operator() * py::self, py::is_operator())
      .def(scalar_operator() + py::self, py::is_operator())
      .def(scalar_operator() - py::self, py::is_operator())

      // common operators

      .def_static("empty", &boson_op::empty,
                  "Creates a sum operator with no terms. And empty sum is the "
                  "neutral element for addition; "
                  "multiplying an empty sum with anything will still result in "
                  "an empty sum.")
      .def_static(
          "identity", []() { return boson_op::identity(); },
          "Creates a product operator with constant value 1. The identity "
          "operator is the neutral "
          "element for multiplication.")
      .def_static(
          "identity",
          [](std::size_t target) { return boson_op::identity(target); },
          "Creates a product operator that applies the identity to the given "
          "target index.")

      // general utility functions

      .def(
          "__str__", [](const boson_op &self) { return self.to_string(); },
          "Returns the string representation of the operator.")
      .def("dump", &boson_op::dump,
           "Prints the string representation of the operator to the standard "
           "output.")
      .def("trim", &boson_op::trim, py::arg("tol") = 0.0,
           py::arg("parameters") = parameter_map(),
           "Removes all terms from the sum for which the absolute value of the "
           "coefficient is below "
           "the given tolerance.")
      .def(
          "trim",
          [](boson_op &self, double tol, const py::kwargs &kwargs) {
            return self.trim(tol, details::kwargs_to_param_map(kwargs));
          },
          py::arg("tol") = 0.0,
          "Removes all terms from the sum for which the absolute value of the "
          "coefficient is below "
          "the given tolerance.")
      .def(
          "canonicalize", [](boson_op &self) { return self.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def(
          "canonicalize",
          [](boson_op &self, const std::set<std::size_t> &degrees) {
            return self.canonicalize(degrees);
          },
          "Expands the operator to act on all given degrees, applying "
          "identities as needed. "
          "If an empty set is passed, canonicalizes all terms in the sum to "
          "act on the same "
          "degrees of freedom.")
      .def("distribute_terms", &boson_op::distribute_terms,
           "Partitions the terms of the sums into the given number of separate "
           "sums.");

  boson_op_term_class
      .def(
          "__iter__",
          [](boson_op_term &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(), "Loop through each term of the operator.")

      // properties

      .def_property_readonly("parameters",
                             &boson_op_term::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")
      .def_property_readonly("degrees", &boson_op_term::degrees,
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
      .def_property_readonly("min_degree", &boson_op_term::min_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("max_degree", &boson_op_term::max_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("ops_count", &boson_op_term::num_ops,
                             "Returns the number of operators in the product.")
      .def_property_readonly(
          "term_id", &boson_op_term::get_term_id,
          "The term id uniquely identifies the operators and targets (degrees) "
          "that they act on, "
          "but does not include information about the coefficient.")
      .def_property_readonly(
          "coefficient", &boson_op_term::get_coefficient,
          "Returns the unevaluated coefficient of the operator. The "
          "coefficient is a "
          "callback function that can be invoked with the `evaluate` method.")

      // constructors

      .def(py::init<>(),
           "Creates a product operator with constant value 1. The returned "
           "operator does not target any degrees of freedom but merely "
           "represents a constant.")
      .def(py::init<std::size_t, std::size_t>(), py::arg("first_degree"),
           py::arg("last_degree"),
           "Creates a product operator that applies an identity operation to "
           "all degrees of "
           "freedom in the range [first_degree, last_degree).")
      .def(py::init<double>(),
           "Creates a product operator with the given constant value. "
           "The returned operator does not target any degrees of freedom.")
      .def(py::init<std::complex<double>>(),
           "Creates a product operator with the given "
           "constant value. The returned operator does not target any degrees "
           "of freedom.")
      .def(py::init([](const scalar_operator &scalar) {
             return boson_op_term() * scalar;
           }),
           "Creates a product operator with non-constant scalar value.")
      .def(py::init<boson_handler>(),
           "Creates a product operator with the given elementary operator.")
      .def(py::init<const boson_op_term &, std::size_t>(), py::arg("operator"),
           py::arg("size") = 0,
           "Creates a copy of the given operator and reserves space for "
           "storing the given "
           "number of product terms (if a size is provided).")
      .def(
          "copy", [](const boson_op_term &self) { return boson_op_term(self); },
          "Creates a copy of the operator.")

      // evaluations

      .def("evaluate_coefficient", &boson_op_term::evaluate_coefficient,
           py::arg("parameters") = parameter_map(),
           "Returns the evaluated coefficient of the product operator. The "
           "parameters is a map of parameter names to their concrete, complex "
           "values.")
      .def(
          "to_matrix",
          [](const boson_op_term &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            auto cmat = self.to_matrix(dimensions, params, invert_order);
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("parameters") = parameter_map(),
          py::arg("invert_order") = false,
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_matrix",
          [](const boson_op_term &self, dimension_map &dimensions,
             bool invert_order, const py::kwargs &kwargs) {
            auto cmat = self.to_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
            return details::cmat_to_numpy(cmat);
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("invert_order") = false,
          "Returns the matrix representation of the operator."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_sparse_matrix",
          [](const boson_op_term &self, dimension_map &dimensions,
             const parameter_map &params, bool invert_order) {
            return self.to_sparse_matrix(dimensions, params, invert_order);
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("parameters") = parameter_map(),
          py::arg("invert_order") = false,
          "Return the sparse matrix representation of the operator. This "
          "representation is a "
          "`Tuple[list[complex], list[int], list[int]]`, encoding the "
          "non-zero values, rows, and columns of the matrix. "
          "This format is supported by `scipy.sparse.csr_array`."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")
      .def(
          "to_sparse_matrix",
          [](const boson_op_term &self, dimension_map &dimensions,
             bool invert_order, const py::kwargs &kwargs) {
            return self.to_sparse_matrix(
                dimensions, details::kwargs_to_param_map(kwargs), invert_order);
          },
          py::arg("dimensions") = dimension_map(),
          py::arg("invert_order") = false,
          "Return the sparse matrix representation of the operator. This "
          "representation is a "
          "`Tuple[list[complex], list[int], list[int]]`, encoding the "
          "non-zero values, rows, and columns of the matrix. "
          "This format is supported by `scipy.sparse.csr_array`."
          "The matrix is ordered according to the convention (endianness) "
          "used in CUDA-Q, and the ordering returned by `degrees`. This order "
          "can be inverted by setting the optional `invert_order` argument to "
          "`True`. "
          "See also the documentation for `degrees` for more detail.")

      // comparisons

      .def("__eq__", &boson_op_term::operator==, py::is_operator(),
           "Return true if the two operators are equivalent. The equivalence "
           "check takes "
           "commutation relations into account. Operators acting on different "
           "degrees of "
           "freedom are never equivalent, even if they only differ by an "
           "identity operator.")
      .def(
          "__eq__",
          [](const boson_op_term &self, const boson_op &other) {
            return other.num_terms() == 1 && *other.begin() == self;
          },
          py::is_operator(), "Return true if the two operators are equivalent.")

      // unary operators

      .def(-py::self, py::is_operator())
      .def(+py::self, py::is_operator())

      // in-place arithmetics

      .def(py::self /= int(), py::is_operator())
      .def(py::self *= int(), py::is_operator())
      .def(py::self /= double(), py::is_operator())
      .def(py::self *= double(), py::is_operator())
      .def(py::self /= std::complex<double>(), py::is_operator())
      .def(py::self *= std::complex<double>(), py::is_operator())
      .def(py::self /= scalar_operator(), py::is_operator())
      .def(py::self *= scalar_operator(), py::is_operator())
      .def(py::self *= py::self, py::is_operator())

      // right-hand arithmetics

      .def(py::self / int(), py::is_operator())
      .def(py::self * int(), py::is_operator())
      .def(py::self + int(), py::is_operator())
      .def(py::self - int(), py::is_operator())
      .def(py::self / double(), py::is_operator())
      .def(py::self * double(), py::is_operator())
      .def(py::self + double(), py::is_operator())
      .def(py::self - double(), py::is_operator())
      .def(py::self / std::complex<double>(), py::is_operator())
      .def(py::self * std::complex<double>(), py::is_operator())
      .def(py::self + std::complex<double>(), py::is_operator())
      .def(py::self - std::complex<double>(), py::is_operator())
      .def(py::self / scalar_operator(), py::is_operator())
      .def(py::self * scalar_operator(), py::is_operator())
      .def(py::self + scalar_operator(), py::is_operator())
      .def(py::self - scalar_operator(), py::is_operator())
      .def(py::self * py::self, py::is_operator())
      .def(py::self + py::self, py::is_operator())
      .def(py::self - py::self, py::is_operator())
      .def(py::self * boson_op(), py::is_operator())
      .def(py::self + boson_op(), py::is_operator())
      .def(py::self - boson_op(), py::is_operator())
      .def(py::self * matrix_op_term(), py::is_operator())
      .def(py::self + matrix_op_term(), py::is_operator())
      .def(py::self - matrix_op_term(), py::is_operator())
      .def(py::self * matrix_op(), py::is_operator())
      .def(py::self + matrix_op(), py::is_operator())
      .def(py::self - matrix_op(), py::is_operator())

      // left-hand arithmetics

      .def(int() * py::self, py::is_operator())
      .def(int() + py::self, py::is_operator())
      .def(int() - py::self, py::is_operator())
      .def(double() * py::self, py::is_operator())
      .def(double() + py::self, py::is_operator())
      .def(double() - py::self, py::is_operator())
      .def(std::complex<double>() * py::self, py::is_operator())
      .def(std::complex<double>() + py::self, py::is_operator())
      .def(std::complex<double>() - py::self, py::is_operator())
      .def(scalar_operator() * py::self, py::is_operator())
      .def(scalar_operator() + py::self, py::is_operator())
      .def(scalar_operator() - py::self, py::is_operator())

      // general utility functions

      .def("is_identity", &boson_op_term::is_identity,
           "Checks if all operators in the product are the identity. "
           "Note: this function returns true regardless of the value of the "
           "coefficient.")
      .def(
          "__str__", [](const boson_op_term &self) { return self.to_string(); },
          "Returns the string representation of the operator.")
      .def("dump", &boson_op_term::dump,
           "Prints the string representation of the operator to the standard "
           "output.")
      .def(
          "canonicalize",
          [](boson_op_term &self) { return self.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def(
          "canonicalize",
          [](boson_op_term &self, const std::set<std::size_t> &degrees) {
            return self.canonicalize(degrees);
          },
          "Expands the operator to act on all given degrees, applying "
          "identities as needed. "
          "The canonicalization will throw a runtime exception if the operator "
          "acts on any degrees "
          "of freedom that are not included in the given set.");
}

void bindBosonWrapper(py::module &mod) {
  bindBosonOperator(mod);
  py::implicitly_convertible<double, boson_op_term>();
  py::implicitly_convertible<std::complex<double>, boson_op_term>();
  py::implicitly_convertible<scalar_operator, boson_op_term>();
  py::implicitly_convertible<boson_op_term, boson_op>();
  bindBosonModule(mod);
}

} // namespace cudaq
