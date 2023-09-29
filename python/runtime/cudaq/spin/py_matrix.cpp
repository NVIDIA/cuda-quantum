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

void bindComplexMatrix(py::module &mod) {
  py::class_<complex_matrix>(
      mod, "ComplexMatrix", py::buffer_protocol(),
      "The :class:`ComplexMatrix` is a thin wrapper around a "
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
           "Create a :class:`ComplexMatrix` from a buffer of data, such as a "
           "numpy.ndarray.")
      .def(
          "__getitem__",
          [](complex_matrix &m, std::size_t i, std::size_t j) {
            return m(i, j);
          },
          "Return the matrix element at i, j.")
      .def(
          "__getitem__",
          [](complex_matrix &m, std::tuple<std::size_t, std::size_t> rowCol) {
            return m(std::get<0>(rowCol), std::get<1>(rowCol));
          },
          "Return the matrix element at i, j.")
      .def("minimal_eigenvalue", &complex_matrix::minimal_eigenvalue,
           "Return the lowest eigenvalue for this :class:`ComplexMatrix`.")
      .def(
          "__str__",
          [](complex_matrix &self) {
            std::stringstream ss;
            self.dump(ss);
            return ss.str();
          },
          "Write this matrix to a string representation.");
}

} // namespace cudaq
