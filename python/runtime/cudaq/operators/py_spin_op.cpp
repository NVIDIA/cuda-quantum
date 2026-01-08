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
#include "py_helpers.h"
#include "py_spin_op.h"

namespace cudaq {

/// @brief Map an OpenFermion operator to our own spin operator
spin_op fromOpenFermionQubitOperator(py::object &op) {
  if (!py::hasattr(op, "terms"))
    throw std::runtime_error(
        "This is not an openfermion operator, must have 'terms' attribute.");
  std::map<std::string, std::function<spin_op_term(std::size_t)>> creatorMap{
      {"X", [](std::size_t i) { return spin_op::x(i); }},
      {"Y", [](std::size_t i) { return spin_op::y(i); }},
      {"Z", [](std::size_t i) { return spin_op::z(i); }}};
  auto terms = op.attr("terms");
  auto H = spin_op::empty();
  for (auto term : terms) {
    auto termTuple = term.cast<py::tuple>();
    auto localTerm = spin_op::identity();
    for (auto &element : termTuple) {
      auto casted = element.cast<std::pair<std::size_t, std::string>>();
      localTerm *= creatorMap[casted.second](casted.first);
    }
    H += terms[term].cast<double>() * localTerm;
  }
  return H;
}

void bindSpinModule(py::module &mod) {
  // Binding the functions in `cudaq::spin` as `_pycudaq` submodule
  // so it's accessible directly in the cudaq namespace.
  auto spin_submodule = mod.def_submodule("spin");
  spin_submodule.def(
      "empty", &spin_op::empty,
      "Returns sum operator with no terms. Note that a sum with no terms "
      "multiplied by anything still is a sum with no terms.");
  spin_submodule.def(
      "identity", []() { return spin_op::identity(); },
      "Returns product operator with constant value 1.");
  // here for consistency with other operators
  spin_submodule.def(
      "identity", [](std::size_t target) { return spin_op::identity(target); },
      py::arg("target"),
      "Returns an identity operator on the given target index.");
  spin_submodule.def(
      "identities",
      [](std::size_t first, std::size_t last) {
        return spin_op_term(first, last);
      },
      py::arg("first"), py::arg("last"),
      "Creates a product operator that applies an identity operation to all "
      "degrees of "
      "freedom in the open range [first, last).");
  spin_submodule.def("i", &spin_op::i<spin_handler>, py::arg("target"),
                     "Returns a Pauli I spin operator on the given "
                     "target qubit index.");
  spin_submodule.def(
      "x", &spin_op::x<spin_handler>, py::arg("target"),
      "Returns a Pauli X spin operator on the given target qubit index.");
  spin_submodule.def(
      "y", &spin_op::y<spin_handler>, py::arg("target"),
      "Returns a Pauli Y spin operator on the given target qubit index.");
  spin_submodule.def(
      "z", &spin_op::z<spin_handler>, py::arg("target"),
      "Returns a Pauli Z spin operator on the given target qubit index.");
  spin_submodule.def("plus", &spin_op::plus<spin_handler>, py::arg("target"),
                     "Return a sigma plus spin operator on the given "
                     "target qubit index.");
  spin_submodule.def("minus", &spin_op::minus<spin_handler>, py::arg("target"),
                     "Return a sigma minus spin operator on the given "
                     "target qubit index.");
  spin_submodule.def(
      "canonicalized",
      [](const spin_op_term &orig) { return spin_op_term::canonicalize(orig); },
      "Removes all identity operators from the operator.");
  spin_submodule.def(
      "canonicalized",
      [](const spin_op_term &orig, const std::set<std::size_t> &degrees) {
        return spin_op_term::canonicalize(orig, degrees);
      },
      "Expands the operator to act on all given degrees, applying identities "
      "as needed. "
      "The canonicalization will throw a runtime exception if the operator "
      "acts on any degrees "
      "of freedom that are not included in the given set.");
  spin_submodule.def(
      "canonicalized",
      [](const spin_op &orig) { return spin_op::canonicalize(orig); },
      "Removes all identity operators from the operator.");
  spin_submodule.def(
      "canonicalized",
      [](const spin_op &orig, const std::set<std::size_t> &degrees) {
        return spin_op::canonicalize(orig, degrees);
      },
      "Expands the operator to act on all given degrees, applying identities "
      "as needed. "
      "If an empty set is passed, canonicalizes all terms in the sum to act on "
      "the same "
      "degrees of freedom.");
}

void bindSpinOperator(py::module &mod) {

  auto spin_op_class = py::class_<spin_op>(mod, "SpinOperator");
  auto spin_op_term_class = py::class_<spin_op_term>(mod, "SpinOperatorTerm");

  spin_op_class
      .def(
          "__iter__",
          [](spin_op &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(), "Loop through each term of the operator.")

      // properties

      .def_property_readonly("parameters", &spin_op::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")
      .def_property_readonly("degrees", &spin_op::degrees,
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
      .def_property_readonly("min_degree", &spin_op::min_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("max_degree", &spin_op::max_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("term_count", &spin_op::num_terms,
                             "Returns the number of terms in the operator.")
      // only exists for spin operators
      .def_property_readonly(
          "qubit_count", &spin_op::num_qubits<spin_handler>,
          "Return the number of qubits this operator acts on.")

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
      .def(py::init<std::size_t>(), py::arg("size"),
           "Creates a sum operator with no terms, reserving "
           "space for the given number of terms (size).")
      // NOTE: only supported on spin ops so far
      .def(py::init<std::vector<double> &>(), py::arg("data"),
           "Creates an operator based on a serialized data representation.")
      // NOTE: only supported on spin ops so far
      .def(py::init([](const std::string &fileName) {
             binary_spin_op_reader reader;
             return reader.read(fileName);
           }),
           "Creates an operator based on a serialized data representation in "
           "the given file.")
      .def(py::init<const spin_op_term &>(),
           "Creates a sum operator with the given term.")
      .def(py::init<const spin_op &>(), "Copy constructor.")
      // NOTE: only supported on spin ops
      .def(py::init([](py::object obj) {
             return fromOpenFermionQubitOperator(obj);
           }),
           "Convert an OpenFermion operator to a CUDA-Q spin operator.")
      .def(
          "copy", [](const spin_op &self) { return spin_op(self); },
          "Creates a copy of the operator.")
      // NOTE: only supported on spin ops
      .def_static("from_word", &spin_op::from_word<spin_handler>,
                  "Creates an operator from a Pauli word string.")
      // NOTE: only supported on spin ops so far
      .def_static(
          "from_json",
          [](const std::string &json_str) {
            py::object json = py::module_::import("json");
            auto data = py::list(json.attr("loads")(json_str));
            return spin_op(data.cast<std::vector<double>>());
          },
          "Convert JSON string ('[d1, d2, d3, ...]') to spin_op")
      // NOTE: only supported on spin ops
      .def_static(
          "random", &spin_op::random<spin_handler>, py::arg("qubit_count"),
          py::arg("term_count"), py::arg("seed") = std::random_device{}(),
          "Return a random spin operator with the given number of terms "
          "(`term_count`) where each term acts on all targets in the open "
          "range "
          "[0, qubit_count). An optional seed value may also be provided.")

      // evaluations

      .def(
          "to_matrix",
          [](const spin_op &self, dimension_map &dimensions,
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
          [](const spin_op &self, dimension_map &dimensions, bool invert_order,
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
          [](const spin_op &self, dimension_map &dimensions,
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
          [](const spin_op &self, dimension_map &dimensions, bool invert_order,
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

      .def("__eq__", &spin_op::operator==, py::is_operator(),
           "Return true if the two operators are equivalent. The equivalence "
           "check takes "
           "commutation relations into account. Operators acting on different "
           "degrees of "
           "freedom are never equivalent, even if they only differ by an "
           "identity operator.")
      .def(
          "__eq__",
          [](const spin_op &self, const spin_op_term &other) {
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
      .def(py::self *= spin_op_term(), py::is_operator())
      .def(py::self += spin_op_term(), py::is_operator())
      .def(py::self -= spin_op_term(), py::is_operator())
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
      .def(py::self * spin_op_term(), py::is_operator())
      .def(py::self + spin_op_term(), py::is_operator())
      .def(py::self - spin_op_term(), py::is_operator())
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

      .def_static("empty", &spin_op::empty,
                  "Creates a sum operator with no terms. And empty sum is the "
                  "neutral element for addition; "
                  "multiplying an empty sum with anything will still result in "
                  "an empty sum.")
      .def_static(
          "identity", []() { return spin_op::identity(); },
          "Creates a product operator with constant value 1. The identity "
          "operator is the neutral "
          "element for multiplication.")
      .def_static(
          "identity",
          [](std::size_t target) { return spin_op::identity(target); },
          "Creates a product operator that applies the identity to the given "
          "target index.")

      // general utility functions

      .def(
          "__str__", [](const spin_op &self) { return self.to_string(); },
          "Returns the string representation of the operator.")
      .def("dump", &spin_op::dump,
           "Prints the string representation of the operator to the standard "
           "output.")
      // NOTE: only supported on spin ops so far
      .def("serialize", &spin_op::get_data_representation<spin_handler>,
           "Returns the serialized data representation of the operator. ")
      // NOTE: only supported on spin ops so far
      .def(
          "to_json",
          [](const spin_op &self) {
            py::object json = py::module_::import("json");
            auto data = self.get_data_representation();
            return json.attr("dumps")(data);
          },
          "Convert spin_op to JSON string: '[d1, d2, d3, ...]'")
      .def("trim", &spin_op::trim, py::arg("tol") = 0.0,
           py::arg("parameters") = parameter_map(),
           "Removes all terms from the sum for which the absolute value of the "
           "coefficient is below "
           "the given tolerance.")
      .def(
          "trim",
          [](spin_op &self, double tol, const py::kwargs &kwargs) {
            return self.trim(tol, details::kwargs_to_param_map(kwargs));
          },
          py::arg("tol") = 0.0,
          "Removes all terms from the sum for which the absolute value of the "
          "coefficient is below "
          "the given tolerance.")
      .def(
          "canonicalize", [](spin_op &self) { return self.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def(
          "canonicalize",
          [](spin_op &self, const std::set<std::size_t> &degrees) {
            return self.canonicalize(degrees);
          },
          "Expands the operator to act on all given degrees, applying "
          "identities as needed. "
          "If an empty set is passed, canonicalizes all terms in the sum to "
          "act on the same "
          "degrees of freedom.")
      .def("distribute_terms", &spin_op::distribute_terms,
           "Partitions the terms of the sums into the given number of separate "
           "sums.");

  // deprecated bindings

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  spin_op_class
      .def(
          "get_coefficient",
          [](const spin_op &op) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "use `evaluate_coefficient` on each term (product "
                         "operator) instead",
                         1);
            if (op.num_terms() == 0)
              return std::complex<double>(0.);
            if (op.num_terms() != 1)
              throw std::runtime_error(
                  "expecting a spin op with at most one term");
            return op.begin()->evaluate_coefficient();
          },
          "Deprecated - use `evaluate_coefficient` on each term (product "
          "operator) instead.")
      // deprecated just to make naming more consistent across the entire API
      .def(
          "get_term_count",
          [](const spin_op &op) {
            PyErr_WarnEx(PyExc_DeprecationWarning, "use `term_count` instead",
                         1);
            return op.num_terms();
          },
          "Deprecated - use `term_count` instead.")
      // deprecated just to make naming more consistent across the entire API
      .def(
          "get_qubit_count",
          [](const spin_op &op) {
            PyErr_WarnEx(PyExc_DeprecationWarning, "use `qubit_count` instead",
                         1);
            return op.num_qubits();
          },
          "Deprecated - use `qubit_count` instead.")
      .def(
          "get_raw_data",
          [](const spin_op &self) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "raw data access will no longer be supported", 1);
            return self.get_raw_data();
          },
          "Deprecated.")
      .def(
          "is_identity",
          [](const spin_op &self) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "is_identity will only be supported on each term "
                         "(product operator) in future releases",
                         1);
            return self.is_identity();
          },
          "Deprecated - is_identity will only be supported on each term "
          "(product operator) in future releases.")
      // constructor for old serialization format
      .def(
          py::init([](const std::vector<double> &data, std::size_t num_qubits) {
            PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "serialization format changed - use the constructor without a "
                "size_t argument to create a spin_op from the new format",
                1);
            return spin_op(data, num_qubits);
          }),
          py::arg("data"), py::arg("num_qubits"),
          "Deprecated - use constructor without the `num_qubits` argument "
          "instead.")
      // new constructor with deprecation warning provided only for backwards
      // compatibility (matching the deprecated data constructor for the old
      // serialization format above)
      .def(py::init([](const std::string &fileName, bool legacy) {
             binary_spin_op_reader reader;
             PyErr_WarnEx(
                 PyExc_DeprecationWarning,
                 "overload provided for compatibility with the deprecated "
                 "serialization format - please migrate to the new format and "
                 "use the constructor without boolean argument",
                 1);
             return reader.read(fileName, legacy);
           }),
           py::arg("filename"), py::arg("legacy"),
           "Constructor available for loading deprecated data representations "
           "from file - will be removed in future releases.")
      .def_static(
          "empty_op",
          []() {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "use `empty` instead to create an sum operator "
                         "without any terms",
                         1);
            return spin_op::empty();
          },
          "Deprecated - use `empty` instead.")
      .def(
          "to_string",
          [](const spin_op &self, bool print_coefficient) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "use overload without boolean argument or "
                         "`get_pauli_word` on each term instead",
                         1);
            return self.to_string(print_coefficient);
          },
          py::arg("print_coefficient") = true,
          "Deprecated - use the standard `str` conversion or `get_pauli_word` "
          "on each term instead.")
      .def(
          "for_each_term",
          [](spin_op &self, py::function functor) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "use standard iteration instead", 1);
            self.for_each_term(functor);
          },
          py::arg("function"), "Deprecated - use standard iteration instead.")
      .def(
          "for_each_pauli",
          [](spin_op &self, py::function functor) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "iterate over the sum to get each term and then "
                         "iterate over the term(s) instead",
                         1);
            self.for_each_pauli(functor);
          },
          py::arg("function"),
          "Deprecated - iterator over sum and then iterator over term "
          "instead.");
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

  spin_op_term_class
      .def(
          "__iter__",
          [](spin_op_term &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(), "Loop through each term of the operator.")

      // properties

      .def_property_readonly("parameters",
                             &spin_op_term::get_parameter_descriptions,
                             "Returns a dictionary that maps each parameter "
                             "name to its description.")
      .def_property_readonly("degrees", &spin_op_term::degrees,
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
      .def_property_readonly("min_degree", &spin_op_term::min_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("max_degree", &spin_op_term::max_degree,
                             "Returns the smallest index of the degrees of "
                             "freedom that the operator targets.")
      .def_property_readonly("ops_count", &spin_op_term::num_ops,
                             "Returns the number of operators in the product.")
      // only exists for spin operators
      .def_property_readonly(
          "qubit_count", &spin_op_term::num_qubits<spin_handler>,
          "Return the number of qubits this operator acts on.")
      .def_property_readonly(
          "term_id", &spin_op_term::get_term_id,
          "The term id uniquely identifies the operators and targets (degrees) "
          "that they act on, "
          "but does not include information about the coefficient.")
      .def_property_readonly(
          "coefficient", &spin_op_term::get_coefficient,
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
      // NOTE: only supported on spin ops so far
      .def(py::init([](const std::vector<double> &data) {
             spin_op op(data);
             if (op.num_terms() != 1)
               throw std::runtime_error(
                   "invalid data representation for product operator");
             return *op.begin();
           }),
           py::arg("data"),
           "Creates an operator based on a serialized data representation.")
      // NOTE: only supported on spin ops so far
      .def(py::init([](const std::string &fileName) {
             binary_spin_op_reader reader;
             spin_op op = reader.read(fileName);
             if (op.num_terms() != 1)
               throw std::runtime_error(
                   "invalid data representation for product operator");
             return *op.begin();
           }),
           "Creates an operator based on a serialized data representation in "
           "the given file.")
      .def(py::init<double>(),
           "Creates a product operator with the given constant value. "
           "The returned operator does not target any degrees of freedom.")
      .def(py::init<std::complex<double>>(),
           "Creates a product operator with the given "
           "constant value. The returned operator does not target any degrees "
           "of freedom.")
      .def(py::init([](const scalar_operator &scalar) {
             return spin_op_term() * scalar;
           }),
           "Creates a product operator with non-constant scalar value.")
      .def(py::init<spin_handler>(),
           "Creates a product operator with the given elementary operator.")
      .def(py::init<const spin_op_term &, std::size_t>(), py::arg("operator"),
           py::arg("size") = 0,
           "Creates a copy of the given operator and reserves space for "
           "storing the given "
           "number of product terms (if a size is provided).")
      .def_static(
          "from_json",
          [](const std::string &json_str) {
            py::object json = py::module_::import("json");
            auto data = py::list(json.attr("loads")(json_str));
            spin_op op(data.cast<std::vector<double>>());
            if (op.num_terms() != 1)
              throw std::runtime_error(
                  "invalid data representation for product operator");
            return *op.begin();
          },
          "Convert JSON string ('[d1, d2, d3, ...]') to spin_op")
      .def(
          "copy", [](const spin_op_term &self) { return spin_op_term(self); },
          "Creates a copy of the operator.")

      // evaluations

      .def("evaluate_coefficient", &spin_op_term::evaluate_coefficient,
           py::arg("parameters") = parameter_map(),
           "Returns the evaluated coefficient of the product operator. The "
           "parameters is a map of parameter names to their concrete, complex "
           "values.")
      .def(
          "to_matrix",
          [](const spin_op_term &self, dimension_map &dimensions,
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
          [](const spin_op_term &self, dimension_map &dimensions,
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
          [](const spin_op_term &self, dimension_map &dimensions,
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
          [](const spin_op_term &self, dimension_map &dimensions,
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

      .def("__eq__", &spin_op_term::operator==, py::is_operator(),
           "Return true if the two operators are equivalent. The equivalence "
           "check takes "
           "commutation relations into account. Operators acting on different "
           "degrees of "
           "freedom are never equivalent, even if they only differ by an "
           "identity operator.")
      .def(
          "__eq__",
          [](const spin_op_term &self, const spin_op &other) {
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
      .def(py::self * spin_op(), py::is_operator())
      .def(py::self + spin_op(), py::is_operator())
      .def(py::self - spin_op(), py::is_operator())
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

      .def("is_identity", &spin_op_term::is_identity,
           "Checks if all operators in the product are the identity. "
           "Note: this function returns true regardless of the value of the "
           "coefficient.")
      .def(
          "__str__", [](const spin_op_term &self) { return self.to_string(); },
          "Returns the string representation of the operator.")
      .def("dump", &spin_op_term::dump,
           "Prints the string representation of the operator to the standard "
           "output.")
      // NOTE: only supported on spin ops so far
      .def(
          "serialize",
          [](const spin_op_term &self) {
            return spin_op(self).get_data_representation();
          },
          "Returns the serialized data representation of the operator. ")
      // NOTE: only supported on spin ops so far
      .def(
          "to_json",
          [](const spin_op_term &self) {
            py::object json = py::module_::import("json");
            auto data = spin_op(self).get_data_representation();
            return json.attr("dumps")(data);
          },
          "Convert spin_op to JSON string: '[d1, d2, d3, ...]'")
      // only exists for spin operators
      .def(
          "get_pauli_word",
          [](spin_op_term &op, std::size_t pad_identities) {
            return op.get_pauli_word(pad_identities);
          },
          py::arg("pad_identities") = 0,
          "Gets the Pauli word representation of this product operator.")
      // only exists for spin operators
      .def("get_binary_symplectic_form",
           &spin_op_term::get_binary_symplectic_form<spin_handler>,
           "Gets the binary symplectic representation of this operator.")
      .def(
          "canonicalize",
          [](spin_op_term &self) { return self.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def(
          "canonicalize",
          [](spin_op_term &self, const std::set<std::size_t> &degrees) {
            return self.canonicalize(degrees);
          },
          "Expands the operator to act on all given degrees, applying "
          "identities as needed. "
          "The canonicalization will throw a runtime exception if the operator "
          "acts on any degrees "
          "of freedom that are not included in the given set.");

  // deprecated bindings

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  spin_op_term_class
      .def(
          "get_coefficient",
          [](const spin_op_term &op) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "use `evaluate_coefficient` instead", 1);
            return op.evaluate_coefficient();
          },
          "Deprecated - use `evaluate_coefficient` instead.")
      // deprecated just to make naming more consistent across the entire API
      .def(
          "get_qubit_count",
          [](const spin_op_term &op) {
            PyErr_WarnEx(PyExc_DeprecationWarning, "use `qubit_count` instead",
                         1);
            return op.num_qubits();
          },
          "Deprecated - use `qubit_count` instead.")
      .def(
          "get_raw_data",
          [](const spin_op_term &self) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "raw data access will no longer be supported", 1);
            return spin_op(self).get_raw_data();
          },
          "Deprecated.")
      .def(
          "to_string",
          [](const spin_op_term &self, bool print_coefficient) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "use overload without boolean argument or use "
                         "`get_pauli_word` instead",
                         1);
            return self.to_string(print_coefficient);
          },
          py::arg("print_coefficient") = true,
          "Deprecated - use the standard `str` conversion or use "
          "`get_pauli_word` instead.")
      .def(
          "distribute_terms",
          [](spin_op_term &op, std::size_t chunks) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "instantiate a `SpinOperator` from this "
                         "`SpinOperatorTerm` and call distribute_terms on that",
                         1);
            return spin_op(op).distribute_terms(chunks);
          },
          py::arg("chunk_count"),
          "Deprecated - instantiate a `SpinOperator` from this "
          "`SpinOperatorTerm` "
          "and call distribute_terms on that.")
      .def(
          "for_each_pauli",
          [](spin_op_term &self, py::function functor) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "use standard iteration instead", 1);
            spin_op(self).for_each_pauli(functor);
          },
          py::arg("function"), "Deprecated - use standard iteration instead.");
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

void bindSpinWrapper(py::module &mod) {
  bindSpinOperator(mod);
  py::implicitly_convertible<double, spin_op_term>();
  py::implicitly_convertible<std::complex<double>, spin_op_term>();
  py::implicitly_convertible<scalar_operator, spin_op_term>();
  py::implicitly_convertible<spin_op_term, spin_op>();
  bindSpinModule(mod);
}

} // namespace cudaq
