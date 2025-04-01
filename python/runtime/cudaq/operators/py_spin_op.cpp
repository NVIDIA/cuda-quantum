/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_spin_op.h"

#include <complex>

namespace cudaq {

/// @brief Map an OpenFermion QubitOperator to our own internal SpinOperator
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

// FIXME: add proper deprecation warnings to the bindings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

void bindSpinClass(py::module &mod) {
  // Binding the `cudaq::spin` class to `_pycudaq` as a submodule
  // so it's accessible directly in the cudaq namespace.
  auto spin_submodule = mod.def_submodule("spin");
  spin_submodule.def("i", &cudaq::spin_op::i<cudaq::spin_handler>,
                     py::arg("target"),
                     "Return an identity :class:`SpinOperator` on the given "
                     "target qubit index.");
  spin_submodule.def(
      "x", &cudaq::spin_op::x<cudaq::spin_handler>, py::arg("target"),
      "Return an X :class:`SpinOperator` on the given target qubit index.");
  spin_submodule.def(
      "y", &cudaq::spin_op::y<cudaq::spin_handler>, py::arg("target"),
      "Return a Y :class:`SpinOperator` on the given target qubit index.");
  spin_submodule.def(
      "z", &cudaq::spin_op::z<cudaq::spin_handler>, py::arg("target"),
      "Return a Z :class:`SpinOperator` on the given target qubit index.");
  spin_submodule.def("plus", &cudaq::spin_op::plus<cudaq::spin_handler>,
                     py::arg("target"),
                     "Return a sigma plus :class:`SpinOperator` on the given "
                     "target qubit index.");
  spin_submodule.def("minus", &cudaq::spin_op::minus<cudaq::spin_handler>,
                     py::arg("target"),
                     "Return a sigma minus :class:`SpinOperator` on the given "
                     "target qubit index.");
}

void bindSpinOperator(py::module &mod) {
  py::enum_<cudaq::pauli>(
      mod, "Pauli", "An enumeration representing the types of Pauli matrices.")
      .value("X", pauli::X)
      .value("Y", pauli::Y)
      .value("Z", pauli::Z)
      .value("I", pauli::I);

  py::class_<cudaq::spin_op_term>(mod, "SpinOperatorTerm")
      .def(py::init([]() { return cudaq::spin_op::identity(); }))
      .def(py::init<cudaq::spin_op_term>(), py::arg("spin_operator"),
           "Copy constructor, given another :class:`SpinOperatorTerm`.")

      /// @brief Bind the member functions.

      .def("get_ops_count", &cudaq::spin_op_term::num_ops,
           "Return the number of terms in this :class:`SpinOperator`.")
      .def(
          "get_coefficient",
          [](cudaq::spin_op_term &op) { return op.evaluate_coefficient(); },
          "Return the coefficient of this :class:`SpinOperatorTerm`.")
      .def("is_identity", &cudaq::spin_op_term::is_identity,
           "Returns a bool indicating if this :class:`SpinOperatorTerm` is "
           "equal "
           "to the "
           "identity.")
      // FIXME: deprecate
      .def(
          "to_string",
          [](cudaq::spin_op_term &op, bool print_coefficient) {
            return op.to_string(print_coefficient);
          },
          py::arg("print_coefficient") = true,
          "Return a string representation of this :class:`SpinOperatorTerm`.")
      .def(
          "__str__", [](cudaq::spin_op_term &op) { return op.to_string(); },
          "Return a string representation of this :class:`SpinOperatorTerm`.")
      .def("dump", &cudaq::spin_op_term::dump,
           "Print a string representation of this :class:`SpinOperatorTerm`.")
      .def(
          "canonicalize",
          [](cudaq::spin_op_term &op) { return op.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def("get_term_id", &cudaq::spin_op_term::get_term_id,
           "Gets the id with which counts and expectation values for this term "
           "can be retrieved.")
      .def(
          "get_pauli_word",
          [](cudaq::spin_op_term &op, std::size_t pad_identities) {
            return op.get_pauli_word(pad_identities);
          },
          py::arg("pad_identities") = 0,
          "Gets the Pauli word representation of this "
          ":class:`SpinOperatorTerm`.")
      .def(
          "get_binary_symplectic_form",
          &cudaq::spin_op_term::get_binary_symplectic_form<cudaq::spin_handler>,
          "Gets the binary symplectic representation of this "
          ":class:`SpinOperatorTerm`.")
      .def(
          "to_matrix",
          [](spin_op_term &self) {
            return self.to_matrix(); // can't bind function ref since it has
                                     // additional (optional) args
          },
          "Return `self` as a :class:`ComplexMatrix`.")
      // iteration over terms is not bound here since we didn't bind the
      // handlers
      .def("__eq__", &cudaq::spin_op_term::operator==, py::arg("other"),
           "Return true if the two :class:`SpinOperatorTerm`'s are equal. "
           "Equality "
           "does "
           "not consider the coefficients.")
      // not sure if we should consider a sum_op and a product_op equal if the
      // content matches...
      .def(
          "__eq__",
          [](spin_op_term &self, spin_op other) {
            return spin_op(self) == other;
          },
          py::arg("other"), "Return true if the two operators are equal.")

      // FIXME: deprecate these
      // The functions below are supported on SpinOperatorTerm for backwards
      // compatiblity, but are only supported by spin_op in C++ (and also only
      // for backward compatiblity).
      .def(
          "get_raw_data",
          [](cudaq::spin_op_term &op) {
            return cudaq::spin_op(op).get_raw_data();
          },
          "Return the raw data of this :class:`SpinOperatorTerm`.")
      .def(
          "to_json",
          [](const cudaq::spin_op_term &p) {
            cudaq::spin_op op(p);
            py::object json = py::module_::import("json");
            auto data = op.get_data_representation();
            return json.attr("dumps")(data);
          },
          "Convert spin_op to JSON string: '[d1, d2, d3, ...]'")
      .def(
          "get_qubit_count",
          [](cudaq::spin_op_term &op) {
            return cudaq::spin_op(op).num_qubits();
          },
          "Return the number of qubits this :class:`SpinOperatorTerm` acts on.")
      .def(
          "get_term_count", [](cudaq::spin_op_term &op) { return 1; },
          "Return the number of terms (always 1).")
      .def(
          "for_each_pauli",
          [](spin_op_term &self, py::function functor) {
            return cudaq::spin_op(self).for_each_pauli(functor);
          },
          py::arg("function"),
          "For a single :class:`SpinOperator` term, apply the given function "
          "to each pauli element in the term. The function must have "
          "`void(pauli, int)` signature where `pauli` is the Pauli matrix "
          "type and the `int` is the qubit index.")
      .def(
          "distribute_terms",
          [](cudaq::spin_op_term &op, std::size_t chunks) {
            return cudaq::spin_op(op).distribute_terms(chunks);
          },
          py::arg("chunk_count"),
          "Return a list of :class:`SpinOperator` representing a distribution "
          "of the "
          "terms in this :class:`SpinOperator` into `chunk_count` sized "
          "chunks.")

      /// @brief Arithmetic operators between different data types

      .def(
          "__add__",
          [](spin_op_term &self, spin_op other) { return self + other; },
          py::arg("other"),
          "Adds a :class:`SpinOperatorTerm` and a :class:`SpinOperator`.")
      .def(
          "__sub__",
          [](spin_op_term &self, spin_op other) { return self - other; },
          py::arg("other"),
          "Subtracts a :class:`SpinOperatorTerm` and a :class:`SpinOperator`.")
      .def(
          "__mul__",
          [](spin_op_term &self, spin_op other) { return self * other; },
          py::arg("other"),
          "Multiplies a :class:`SpinOperatorTerm` and a :class:`SpinOperator`.")

      /// @brief Bind overloaded operators that are in-place on
      /// :class:`SpinOperatorTerm`.

      // `this_spin_op_term` *= :class:`SpinOperatorTerm`
      .def(py::self *= py::self, py::arg("other"),
           "Multiply the given :class:`SpinOperatorTerm` with this one and "
           "return "
           "*this.")
      // `this_spin_op_term` *= `float`
      .def(py::self *= float(), py::arg("other"),
           "Multiply the :class:`SpinOperatorTerm` by the given "
           "float value and return *this.")
      // `this_spin_op_term` *= `double`
      .def(py::self *= double(), py::arg("other"),
           "Multiply the :class:`SpinOperatorTerm` by the given "
           "double value and return *this.")
      // `this_spin_op_term` *= `complex`
      .def(py::self *= std::complex<double>(), py::arg("other"),
           "Multiply the :class:`SpinOperatorTerm` by the given complex value "
           "and "
           "return "
           "*this.")

      /// @brief Bind overloaded operators that return a new
      /// :class:`SpinOperatorTerm`.

      // :class:`SpinOperatorTerm` + :class:`SpinOperatorTerm`
      .def(py::self + py::self, py::arg("other"),
           "Add the given :class:`SpinOperatorTerm` to this one and "
           "return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperatorTerm` + `double`
      .def(py::self + double(), py::arg("other"),
           "Add a double to the given :class:`SpinOperatorTerm` and "
           "return result as a new :class:`SpinOperator`.")
      // `double` + :class:`SpinOperatorTerm`
      .def(double() + py::self, py::arg("other"),
           "Add a :class:`SpinOperatorTerm` to the given double and "
           "return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperatorTerm` - :class:`SpinOperatorTerm`
      .def(py::self - py::self, py::arg("other"),
           "Subtract the given :class:`SpinOperatorTerm` from this one "
           "and return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperatorTerm` - `double`
      .def(py::self - double(), py::arg("other"),
           "Subtract a double from the given :class:`SpinOperatorTerm` "
           "and return result as a new :class:`SpinOperator`.")
      // `double` - :class:`SpinOperatorTerm`
      .def(double() - py::self, py::arg("other"),
           "Subtract a :class:`SpinOperatorTerm` from the given double "
           "and return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperatorTerm` * :class:`SpinOperatorTerm`
      .def(py::self * py::self, py::arg("other"),
           "Multiply the given :class:`SpinOperatorTerm`'s together "
           "and return result as a new :class:`SpinOperatorTerm`.")
      // :class:`SpinOperatorTerm` * `double`
      .def(py::self * double(), py::arg("other"),
           "Multiply the :class:`SpinOperatorTerm` by the given double "
           "and return result as a new :class:`SpinOperatorTerm`.")
      // `double` * :class:`SpinOperatorTerm`
      .def(double() * py::self, py::arg("other"),
           "Multiply the double by the given :class:`SpinOperatorTerm` "
           "and return result as a new :class:`SpinOperatorTerm`.")
      // :class:`SpinOperatorTerm` * `complex`
      .def(py::self * std::complex<double>(), py::arg("other"),
           "Multiply the :class:`SpinOperatorTerm` by the given complex value "
           "and "
           "return "
           "result as a new :class:`SpinOperatorTerm`.")
      // `complex` * :class:`SpinOperatorTerm`
      .def(std::complex<double>() * py::self, py::arg("other"),
           "Multiply the complex value by the given :class:`SpinOperatorTerm` "
           "and "
           "return "
           "result as a new :class:`SpinOperatorTerm`.");

  py::class_<cudaq::spin_op>(mod, "SpinOperator")
      // FIXME: deprecate this one
      .def(py::init([]() { return cudaq::spin_op::identity(); }),
           "Empty constructor, creates the identity term.")
      .def(py::init([](std::size_t size) { return cudaq::spin_op(size); }),
           "Empty constructor, creates a sum operator with no terms, reserving "
           "memory for the given number of terms.")
      // FIXME: deprecate name
      .def_static("empty_op", &cudaq::spin_op::empty)
      .def_static("empty", &cudaq::spin_op::empty)
      .def(py::init([](std::string fileName) {
             cudaq::binary_spin_op_reader reader;
             return reader.read(fileName);
           }),
           "Read in :class:`SpinOperator` from file.")
      .def(py::init<const cudaq::spin_op>(), py::arg("spin_operator"),
           "Copy constructor, given another :class:`SpinOperator`.")
      .def(py::init<const cudaq::spin_op_term>(), py::arg("spin_operator"),
           "Constructor given a :class:`SpinOperatorTerm`.")
      .def(py::init<std::vector<double> &>(), py::arg("data"),
           "Construct a :class:`SpinOperator` from a list of numeric values.")
      .def(py::init<std::vector<double> &, std::size_t>(), py::arg("data"),
           py::arg("num_qubits"),
           "Construct a :class:`SpinOperator` from a list of numeric values. "
           "The encoding is as follows: for each term, a list of doubles where "
           "element `i` is a 3.0 for a `Y`, a 1.0 for a `X`, and a 2.0 for a "
           "`Z` on qubit `i`, followed by the real and imaginary part of the "
           "coefficient. Each set of term elements is appended to the one "
           "list. The list is ended with the total number of terms.")
      .def(py::init(
               [](py::object o) { return fromOpenFermionQubitOperator(o); }),
           "Create from OpenFermion QubitOperator.")
      /*
      .def(py::init<std::size_t>(), py::arg("num_qubits"),
           "Construct the identity term on the given number of qubits.")
      */
      .def(
          "to_json",
          [](const cudaq::spin_op &p) {
            py::object json = py::module_::import("json");
            auto data = p.get_data_representation();
            return json.attr("dumps")(data);
          },
          "Convert spin_op to JSON string: '[d1, d2, d3, ...]]'")
      .def_static(
          "from_json",
          [](const std::string &j) {
            py::object json = py::module_::import("json");
            auto data = py::list(json.attr("loads")(j));
            cudaq::spin_op p(data.cast<std::vector<double>>());
            return p;
          },
          "Convert JSON string ('[[d1, d2, d3, ...], numQubits]') to spin_op")

      /// @brief Bind the member functions.

      // FIXME: deprecate
      .def("get_raw_data", &cudaq::spin_op::get_raw_data<cudaq::spin_handler>,
           "Return the raw data of this :class:`SpinOperator`.")
      .def("get_term_count", &cudaq::spin_op::num_terms,
           "Return the number of terms in this :class:`SpinOperator`.")
      .def("get_qubit_count", &cudaq::spin_op::num_qubits<cudaq::spin_handler>,
           "Return the number of qubits this :class:`SpinOperator` is on.")
      // FIXME: deprecate
      .def(
          "get_coefficient",
          [](cudaq::spin_op &op) {
            if (op.num_terms() == 0)
              return std::complex<double>(0.);
            if (op.num_terms() != 1)
              throw std::runtime_error(
                  "expecting a spin op with at most one term");
            return op.begin()->get_coefficient().evaluate();
          },
          "Return the coefficient of this :class:`SpinOperator`. Must be a "
          "`SpinOperator` with one term, otherwise an exception is thrown.")
      // FIXME: deprecate
      .def("is_identity", &cudaq::spin_op::is_identity<cudaq::spin_handler>,
           "Returns a bool indicating if this :class:`SpinOperator` is equal "
           "to the "
           "identity.")
      // FIXME: deprecate
      .def(
          "to_string",
          [](cudaq::spin_op &op, bool print_coefficient) {
            return op.to_string(print_coefficient);
          },
          py::arg("print_coefficient") = true,
          "Return a string representation of this :class:`SpinOperator`.")
      .def(
          "__str__", [](cudaq::spin_op &op) { return op.to_string(); },
          "Return a string representation of this :class:`SpinOperator`.")
      .def("dump", &cudaq::spin_op::dump,
           "Print a string representation of this :class:`SpinOperator`.")
      .def(
          "canonicalize", [](cudaq::spin_op &op) { return op.canonicalize(); },
          "Removes all identity operators from the operator.")
      .def("distribute_terms", &cudaq::spin_op::distribute_terms,
           py::arg("chunk_count"),
           "Return a list of :class:`SpinOperator` representing a distribution "
           "of the "
           "terms in this :class:`SpinOperator` into `chunk_count` sized "
           "chunks.")
      .def_static("random", &cudaq::spin_op::random<cudaq::spin_handler>,
                  py::arg("qubit_count"), py::arg("term_count"),
                  py::arg("seed") = std::random_device{}(),
                  "Return a random :class:`SpinOperator` on the given number "
                  "of qubits (`qubit_count`) and "
                  "composed of the given number of terms (`term_count`). An "
                  "optional seed value may also be provided.")
      .def_static(
          "from_word", &cudaq::spin_op::from_word<cudaq::spin_handler>,
          py::arg("word"),
          R"#(Return a :class:`SpinOperator` corresponding to the provided Pauli `word`.

.. code-block:: python

    # Example:
    # The first and third qubits will receive a Pauli X,
    # while the second qubit will receive a Pauli Y.
    word = "XYX"
    # Convert word to spin operator.
    spin_operator = cudaq.SpinOperator.from_word(word)
    print(spin_operator) # prints: `[1+0j] XYX`)#")
      .def(
          "for_each_term",
          [](spin_op &self, py::function functor) {
            self.for_each_term([&](const spin_op &term) { functor(term); });
          },
          py::arg("function"),
          "Apply the given function to all terms in this "
          ":class:`SpinOperator`. "
          "The input function must have `void(SpinOperator)` signature.")
      .def(
          "for_each_pauli",
          [](spin_op &self, py::function functor) {
            self.for_each_pauli(functor);
          },
          py::arg("function"),
          "For a single :class:`SpinOperator` term, apply the given function "
          "to each pauli element in the term. The function must have "
          "`void(pauli, int)` signature where `pauli` is the Pauli matrix "
          "type and the `int` is the qubit index.")
      .def("serialize", &spin_op::get_data_representation<cudaq::spin_handler>,
           "Return a serialized representation of the :class:`SpinOperator`. "
           "Specifically, this encoding is via a vector of doubles. The "
           "encoding is as follows: for each term, a list of doubles where the "
           "ith element is a 3.0 for a Y, a 1.0 for a X, and a 2.0 for a Z on "
           "qubit i, followed by the real and imaginary part of the "
           "coefficient. "
           "Each term is appended to the array forming one large 1d array of "
           "doubles. The array is ended with the total number of terms "
           "represented as a double.")
      .def(
          "to_matrix",
          [](spin_op &self) {
            return self.to_matrix(); // can't bind function ref since it has
                                     // additional (optional) args
          },
          "Return `self` as a :class:`ComplexMatrix`.")
      .def("to_sparse_matrix", &spin_op::to_sparse_matrix<cudaq::spin_handler>,
           "Return `self` as a sparse matrix. This representation is a "
           "`Tuple[list[complex], list[int], list[int]]`, encoding the "
           "non-zero values, rows, and columns of the matrix. "
           "This format is supported by `scipy.sparse.csr_array`.")
      .def(
          "__iter__",
          [](spin_op &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(),
          "Loop through each term of this :class:`SpinOperator`.")
      // :class:`SpinOperator` == :class:`SpinOperator`
      .def("__eq__", &cudaq::spin_op::operator==, py::arg("other"),
           "Return true if the two :class:`SpinOperator`'s are equal.")
      // not sure if we should consider a sum_op and a product_op equal if the
      // content matches...
      .def(
          "__eq__",
          [](spin_op &self, spin_op_term other) {
            return self == spin_op(other);
          },
          py::arg("other"), "Return true if the two operators are equal.")

      /// @brief Arithmetic operators between different data types

      .def(
          "__add__",
          [](spin_op &self, spin_op_term other) { return self + other; },
          py::arg("other"),
          "Adds a :class:`SpinOperator` and a :class:`SpinOperatorTerm`.")
      .def(
          "__iadd__",
          [](spin_op &self, spin_op_term other) { return self += other; },
          py::arg("other"),
          "Adds a :class:`SpinOperator` and a :class:`SpinOperatorTerm` and "
          "assigns it to self.")
      .def(
          "__sub__",
          [](spin_op &self, spin_op_term other) { return self - other; },
          py::arg("other"),
          "Subtracts a :class:`SpinOperator` and a :class:`SpinOperatorTerm`.")
      .def(
          "__isub__",
          [](spin_op &self, spin_op_term other) { return self -= other; },
          py::arg("other"),
          "Subtracts a :class:`SpinOperator` and a :class:`SpinOperatorTerm` "
          "and assigns it to self.")
      .def(
          "__mul__",
          [](spin_op &self, spin_op_term other) { return self * other; },
          py::arg("other"),
          "Multiplies a :class:`SpinOperator` and a :class:`SpinOperatorTerm`.")
      .def(
          "__imul__",
          [](spin_op &self, spin_op_term other) { return self *= other; },
          py::arg("other"),
          "Multiplies a :class:`SpinOperator` and a :class:`SpinOperatorTerm` "
          "and assigns it to self.")

      /// @brief Bind overloaded operators that are in-place on
      /// :class:`SpinOperator`.

      // `this_spin_op` += :class:`SpinOperator`
      .def(py::self += py::self,
           "Add the given :class:`SpinOperator` to this one and return *this.")
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
      // `this_spin_op` -= :class:`SpinOperator`
      .def(py::self -= py::self, "Subtract the given :class:`SpinOperator` "
                                 "from this one and return *this.")
#ifdef __clang__
#pragma clang diagnostic pop
#endif
      // `this_spin_op` *= :class:`SpinOperator`
      .def(py::self *= py::self, py::arg("other"),
           "Multiply the given :class:`SpinOperator` with this one and return "
           "*this.")
      // `this_spin_op` *= `float`
      .def(py::self *= float(), py::arg("other"),
           "Multiply the :class:`SpinOperator` by the given "
           "float value and return *this.")
      // `this_spin_op` *= `double`
      .def(py::self *= double(), py::arg("other"),
           "Multiply the :class:`SpinOperator` by the given "
           "double value and return *this.")
      // `this_spin_op` *= `complex`
      .def(py::self *= std::complex<double>(), py::arg("other"),
           "Multiply the :class:`SpinOperator` by the given complex value and "
           "return "
           "*this.")

      /// @brief Bind overloaded operators that return a new
      /// :class:`SpinOperator`.

      // :class:`SpinOperator` + :class:`SpinOperator`
      .def(py::self + py::self, py::arg("other"),
           "Add the given :class:`SpinOperator` to this one and "
           "return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperator` + `double`
      .def(py::self + double(), py::arg("other"),
           "Add a double to the given :class:`SpinOperator` and "
           "return result as a new :class:`SpinOperator`.")
      // `double` + :class:`SpinOperator`
      .def(double() + py::self, py::arg("other"),
           "Add a :class:`SpinOperator` to the given double and "
           "return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperator` - :class:`SpinOperator`
      .def(py::self - py::self, py::arg("other"),
           "Subtract the given :class:`SpinOperator` from this one "
           "and return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperator` - `double`
      .def(py::self - double(), py::arg("other"),
           "Subtract a double from the given :class:`SpinOperator` "
           "and return result as a new :class:`SpinOperator`.")
      // `double` - :class:`SpinOperator`
      .def(double() - py::self, py::arg("other"),
           "Subtract a :class:`SpinOperator` from the given double "
           "and return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperator` * :class:`SpinOperator`
      .def(py::self * py::self, py::arg("other"),
           "Multiply the given :class:`SpinOperator`'s together "
           "and return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperator` * `double`
      .def(py::self * double(), py::arg("other"),
           "Multiply the :class:`SpinOperator` by the given double "
           "and return result as a new :class:`SpinOperator`.")
      // `double` * :class:`SpinOperator`
      .def(double() * py::self, py::arg("other"),
           "Multiply the double by the given :class:`SpinOperator` "
           "and return result as a new :class:`SpinOperator`.")
      // :class:`SpinOperator` * `complex`
      .def(py::self * std::complex<double>(), py::arg("other"),
           "Multiply the :class:`SpinOperator` by the given complex value and "
           "return "
           "result as a new :class:`SpinOperator`.")
      // `complex` * :class:`SpinOperator`
      .def(std::complex<double>() * py::self, py::arg("other"),
           "Multiply the complex value by the given :class:`SpinOperator` and "
           "return "
           "result as a new :class:`SpinOperator`.");
}

#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

void bindSpinWrapper(py::module &mod) {
  bindSpinClass(mod);
  bindSpinOperator(mod);
  // If the spin op in the execution context is a pointer
  // rather than an actual copy, then we may run into trouble
  // that the pointer points to a temporary object due to
  // spin_op_term -> spin_op automatic conversion.
  // I am not sure why I didn't see a similar issue in C++,
  // but I am concerned it might be there as well.
  // I hence decided to have the context own it's spin -
  // I *think* is should only be one additional copy and seems
  // less likely to have any hidden issues.
  py::implicitly_convertible<spin_op_term, spin_op>();
}

} // namespace cudaq
