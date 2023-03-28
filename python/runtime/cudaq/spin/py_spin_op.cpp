/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaq/spin_op.h"
#include "py_spin_op.h"

#include <complex>

namespace cudaq {

/// @brief Extract the array data from a buffer_info into our
/// own allocated data pointer.
void extractMatrixData(py::buffer_info &info, std::complex<double> *data) {
  if (info.format != py::format_descriptor<std::complex<double>>::format())
    throw std::runtime_error(
        "Incompatible buffer format, must be np.complex128.");

  if (info.ndim != 2)
    throw std::runtime_error("Incompatible buffer shape.");

  memcpy(data, info.ptr,
         sizeof(std::complex<double>) * (info.shape[0] * info.shape[1]));
}

void bindSpinClass(py::module &mod) {
  // Binding the `cudaq::spin` class to `_pycudaq` as a submodule
  // so it's accessible directly in the cudaq namespace.
  auto spin_submodule = mod.def_submodule("spin");
  spin_submodule.def("i", &cudaq::spin::i, py::arg("target"),
                     "Return an identity `cudaq.SpinOperator` on the given "
                     "target qubit index.");
  spin_submodule.def(
      "x", &cudaq::spin::x, py::arg("target"),
      "Return an X `cudaq.SpinOperator` on the given target qubit index.");
  spin_submodule.def(
      "y", &cudaq::spin::y, py::arg("target"),
      "Return a Y `cudaq.SpinOperator` on the given target qubit index.");
  spin_submodule.def(
      "z", &cudaq::spin::z, py::arg("target"),
      "Return a Z `cudaq.SpinOperator` on the given target qubit index.");
}

void bindComplexMatrix(py::module &mod) {
  py::class_<complex_matrix>(mod, "ComplexMatrix", py::buffer_protocol(),
                             "The ComplexMatrix is a thin wrapper around a "
                             "matrix of complex<double> elements.")
      /// The following makes this fully compatible with NumPy
      .def_buffer([](complex_matrix &op) -> py::buffer_info {
        return py::buffer_info(
            op.data(), sizeof(std::complex<double>),
            py::format_descriptor<std::complex<double>>::format(), 2,
            {op.rows(), op.cols()},
            {sizeof(std::complex<double>) * op.cols(),
             sizeof(std::complex<double>)});
      })
      .def(py::init([](const py::buffer &b) {
             py::buffer_info info = b.request();
             complex_matrix m(info.shape[0], info.shape[1]);
             extractMatrixData(info, m.data());
             return m;
           }),
           "Create a ComplexMatrix from a buffer of data, like a numpy array.")
      .def("__getitem__", &complex_matrix::operator(),
           "Return the matrix element at i, j.")
      .def(
          "__str__",
          [](complex_matrix &self) {
            std::stringstream ss;
            self.dump(ss);
            return ss.str();
          },
          "Write this matrix to a string representation.");
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
      .def(py::init<const cudaq::spin_op>(), py::arg("spin_operator"),
           "Copy constructor, given another `cudaq.SpinOperator`.")

      /// @brief Bind the member functions.
      .def("get_term_count", &cudaq::spin_op::n_terms,
           "Return the number of terms in this `SpinOperator`.")
      .def("get_qubit_count", &cudaq::spin_op::n_qubits,
           "Return the number of qubits this `SpinOperator` is on.")
      .def("get_term_coefficient", &cudaq::spin_op::get_term_coefficient,
           py::arg("term"),
           "Return the coefficient of this `SpinOperator` at the given term "
           "index.")
      .def("get_coefficients", &cudaq::spin_op::get_coefficients,
           "Return all term coefficients in this `SpinOperator`.")
      .def("is_identity", &cudaq::spin_op::is_identity,
           "Returns a bool indicating if this `SpinOperator` is equal to the "
           "identity.")
      .def(
          "to_string", [](cudaq::spin_op &op) { return op.to_string(); },
          "Return a string representation of this `SpinOperator`.")
      .def(
          "__str__", [](cudaq::spin_op &op) { return op.to_string(); },
          "Return a string representation of this `SpinOperator`.")
      .def("dump", &cudaq::spin_op::dump,
           "Print a string representation of this `SpinOperator`.")
      .def("slice", &cudaq::spin_op::slice,
           "Return a slice of this `SpinOperator`. The slice starts at the "
           "term index and contains the following `count` terms.")
      .def_static("random", &cudaq::spin_op::random,
                  "Return a random spin_op on the given number of qubits and "
                  "composed of the given number of terms.")
      .def(
          "for_each_term",
          [](spin_op &self, py::function functor) {
            self.for_each_term([&](const spin_op &term) { functor(term); });
          },
          "Apply the given function to all terms in this `cudaq.SpinOperator`. "
          "The input function must have `void(SpinOperator)` signature.")
      .def(
          "for_each_pauli",
          [](spin_op &self, py::function functor) {
            self.for_each_pauli(functor);
          },
          "For a single `cudaq.SpinOperator` term, apply the given function "
          "to each pauli element in the term. The function must have "
          "`void(pauli, int)` signature where `pauli` is the Pauli matrix "
          "type and the `int` is the qubit index.")

      .def("to_matrix", &spin_op::to_matrix, "Return a matrix representation for this `cudaq.SpinOperator`.")
      /// @brief Bind overloaded operators that are in-place on
      /// `cudaq.SpinOperator`.
      // `this_spin_op` += `cudaq.SpinOperator`
      .def(py::self += py::self,
           "Add the given `SpinOperator` to this one and return *this.")
      // `this_spin_op` -= `cudaq.SpinOperator`
      .def(py::self -= py::self,
           "Subtract the given `SpinOperator` from this one and return *this.")
      // `this_spin_op` *= `cudaq.SpinOperator`
      .def(py::self *= py::self,
           "Multiply the given `SpinOperator` with this one and return *this.")
      // `this_spin_op` *= `float`
      .def(py::self *= float(), "Multiply the `SpinOperator` by the given "
                                "float value and return *this.")
      // `this_spin_op` *= `double`
      .def(py::self *= double(), "Multiply the `SpinOperator` by the given "
                                 "double value and return *this.")
      // `this_spin_op` *= `complex`
      .def(py::self *= std::complex<double>(),
           "Multiply the `SpinOperator` by the given complex value and return "
           "*this.")
      // `cudaq.SpinOperator` == `cudaq.SpinOperator`
      .def("__eq__", &cudaq::spin_op::operator==,
           "Return true if the two `SpinOperator`'s are equal. Equality does "
           "not consider the coefficients.")

      /// @brief Bind overloaded operators that return a new
      /// `cudaq.SpinOperator`.
      // `this_spin_op[idx]`
      .def("__getitem__", &cudaq::spin_op::operator[], py::arg("index"),
           "Return the term of this `SpinOperator` at the provided index as a "
           "new `cudaq.SpinOperator`.")
      // `cudaq.SpinOperator` + `cudaq.SpinOperator`
      .def(py::self + py::self, "Add the given `SpinOperator` to this one and "
                                "return result as a new `cudaq.SpinOperator`.")
      // `cudaq.SpinOperator` + `double`
      .def(py::self + double(), "Add a double to the given `SpinOperator` and "
                                "return result as a new `cudaq.SpinOperator`.")
      // `double` + `cudaq.SpinOperator`
      .def(double() + py::self, "Add a `SpinOperator` to the given double and "
                                "return result as a new `cudaq.SpinOperator`.")
      // `cudaq.SpinOperator` - `cudaq.SpinOperator`
      .def(py::self - py::self,
           "Subtract the given `SpinOperator` from this one "
           "and return result as a new `cudaq.SpinOperator`.")
      // `cudaq.SpinOperator` - `double`
      .def(py::self - double(),
           "Subtract a double from the given `SpinOperator` "
           "and return result as a new `cudaq.SpinOperator`.")
      // `double` - `cudaq.SpinOperator`
      .def(double() - py::self,
           "Subtract a `SpinOperator` from the given double "
           "and return result as a new `cudaq.v=SpinOperator`.")
      // `cudaq.SpinOperator` * `cudaq.SpinOperator`
      .def(py::self * py::self,
           "Multiply the given `cudaq.SpinOperator`'s together "
           "and return result as a new `cudaq.SpinOperator`.")
      // `cudaq.SpinOperator` * `double`
      .def(py::self * double(),
           "Multiply the `SpinOperator` by the given double "
           "and return result as a new `cudaq.SpinOperator`.")
      // `double` * `cudaq.SpinOperator`
      .def(double() * py::self,
           "Multiply the double by the given `SpinOperator` "
           "and return result as a new `cudaq.SpinOperator`.")
      // `cudaq.SpinOperator` * `complex`
      .def(py::self * std::complex<double>(),
           "Multiply the `SpinOperator` by the given complex value and return "
           "result as a new `cudaq.SpinOperator`.")
      // `complex` * `cudaq.SpinOperator`
      .def(std::complex<double>() * py::self,
           "Multiply the complex value by the given `SpinOperator` and return "
           "result as a new `cudaq.SpinOperator`.");
}

void bindSpinWrapper(py::module &mod) {
  bindSpinClass(mod);
  bindComplexMatrix(mod);
  bindSpinOperator(mod);
}

} // namespace cudaq
