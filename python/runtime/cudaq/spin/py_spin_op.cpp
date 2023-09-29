/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/spin_op.h"
#include "py_spin_op.h"

#include <complex>

namespace cudaq {

/// @brief Map an OpenFermion QubitOperator to our own internal SpinOperator
spin_op fromOpenFermionQubitOperator(py::object &op) {
  if (!py::hasattr(op, "terms"))
    throw std::runtime_error(
        "This is not an openfermion operator, must have 'terms' attribute.");
  std::map<std::string, std::function<spin_op(std::size_t)>> creatorMap{
      {"X", [](std::size_t i) { return spin::x(i); }},
      {"Y", [](std::size_t i) { return spin::y(i); }},
      {"Z", [](std::size_t i) { return spin::z(i); }}};
  auto terms = op.attr("terms");
  spin_op H;
  for (auto term : terms) {
    auto termTuple = term.cast<py::tuple>();
    spin_op localTerm;
    for (auto &element : termTuple) {
      auto casted = element.cast<std::pair<std::size_t, std::string>>();
      localTerm *= creatorMap[casted.second](casted.first);
    }
    H += terms[term].cast<double>() * localTerm;
  }
  H -= spin::i(H.num_qubits() - 1);
  return H;
}

void bindSpinClass(py::module &mod) {
  // Binding the `cudaq::spin` class to `_pycudaq` as a submodule
  // so it's accessible directly in the cudaq namespace.
  auto spin_submodule = mod.def_submodule("spin");
  spin_submodule.def("i", &cudaq::spin::i, py::arg("target"),
                     "Return an identity :class:`SpinOperator` on the given "
                     "target qubit index.");
  spin_submodule.def(
      "x", &cudaq::spin::x, py::arg("target"),
      "Return an X :class:`SpinOperator` on the given target qubit index.");
  spin_submodule.def(
      "y", &cudaq::spin::y, py::arg("target"),
      "Return a Y :class:`SpinOperator` on the given target qubit index.");
  spin_submodule.def(
      "z", &cudaq::spin::z, py::arg("target"),
      "Return a Z :class:`SpinOperator` on the given target qubit index.");
}

void bindSpinOperator(py::module &mod) {
  py::enum_<cudaq::pauli>(
      mod, "Pauli", "An enumeration representing the types of Pauli matrices.")
      .value("X", pauli::X)
      .value("Y", pauli::Y)
      .value("Z", pauli::Z)
      .value("I", pauli::I);

  py::class_<cudaq::spin_op>(mod, "SpinOperator")
      /// @brief Bind the constructors.
      .def(py::init<>(), "Empty constructor, creates the identity term.")
      .def(py::init([](std::string fileName) {
             cudaq::binary_spin_op_reader reader;
             return reader.read(fileName);
           }),
           "Read in :class:`SpinOperator` from file.")
      .def(py::init<const cudaq::spin_op>(), py::arg("spin_operator"),
           "Copy constructor, given another :class:`SpinOperator`.")
      .def(py::init(
               [](py::object o) { return fromOpenFermionQubitOperator(o); }),
           "Create from OpenFermion QubitOperator.")
      .def(py::init<std::vector<double> &, std::size_t>(), py::arg("data"),
           py::arg("num_qubits"),
           "Construct a :class:`SpinOperator` from a list of numeric values. "
           "The encoding is as follows: for each term, a list of doubles where "
           "element `i` is a 3.0 for a `Y`, a 1.0 for a `X`, and a 2.0 for a "
           "`Z` on qubit `i`, followed by the real and imaginary part of the "
           "coefficient. Each set of term elements is appended to the one "
           "list. The list is ended with the total number of terms.")

      /// @brief Bind the member functions.
      .def("get_raw_data", &cudaq::spin_op::get_raw_data,
           "Return the raw data of this :class:`SpinOperator`.")
      .def("get_term_count", &cudaq::spin_op::num_terms,
           "Return the number of terms in this :class:`SpinOperator`.")
      .def("get_qubit_count", &cudaq::spin_op::num_qubits,
           "Return the number of qubits this :class:`SpinOperator` is on.")
      .def("get_coefficient", &cudaq::spin_op::get_coefficient,
           "Return the coefficient of this :class:`SpinOperator`. Must be a "
           "`SpinOperator` with one term, otherwise an exception is thrown.")
      .def("is_identity", &cudaq::spin_op::is_identity,
           "Returns a bool indicating if this :class:`SpinOperator` is equal "
           "to the "
           "identity.")
      .def(
          "to_string", [](cudaq::spin_op &op) { return op.to_string(); },
          "Return a string representation of this :class:`SpinOperator`.")
      .def(
          "__str__", [](cudaq::spin_op &op) { return op.to_string(); },
          "Return a string representation of this :class:`SpinOperator`.")
      .def("dump", &cudaq::spin_op::dump,
           "Print a string representation of this :class:`SpinOperator`.")
      .def("distribute_terms", &cudaq::spin_op::distribute_terms,
           py::arg("chunk_count"),
           "Return a list of :class:`SpinOperator` representing a distribution "
           "of the "
           "terms in this :class:`SpinOperator` into `chunk_count` sized "
           "chunks.")
      .def_static("random", &cudaq::spin_op::random, py::arg("qubit_count"),
                  py::arg("term_count"),
                  py::arg("seed") = std::random_device{}(),
                  "Return a random :class:`SpinOperator` on the given number "
                  "of qubits (`qubit_count`) and "
                  "composed of the given number of terms (`term_count`). An "
                  "optional seed value may also be provided.")
      .def_static(
          "from_word", &cudaq::spin_op::from_word, py::arg("word"),
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
      .def("serialize", &spin_op::getDataRepresentation,
           "Return a serialized representation of the :class:`SpinOperator`. "
           "Specifically, this encoding is via a vector of doubles. The "
           "encoding is as follows: for each term, a list of doubles where the "
           "ith element is a 3.0 for a Y, a 1.0 for a X, and a 2.0 for a Z on "
           "qubit i, followed by the real and imaginary part of the "
           "coefficient. "
           "Each term is appended to the array forming one large 1d array of "
           "doubles. The array is ended with the total number of terms "
           "represented as a double.")
      .def("to_matrix", &spin_op::to_matrix,
           "Return `self` as a :class:`ComplexMatrix`.")
      .def("to_sparse_matrix", &spin_op::to_sparse_matrix,
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
      // :class:`SpinOperator` == :class:`SpinOperator`
      .def("__eq__", &cudaq::spin_op::operator==, py::arg("other"),
           "Return true if the two :class:`SpinOperator`'s are equal. Equality "
           "does "
           "not consider the coefficients.")

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

void bindSpinWrapper(py::module &mod) {
  bindSpinClass(mod);
  bindSpinOperator(mod);
}

} // namespace cudaq
