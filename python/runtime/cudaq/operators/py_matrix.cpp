/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/stl/complex.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/map.h>

#include "cudaq/operators/matrix.h"
#include "py_helpers.h"
#include "py_matrix.h"

#include <complex>
#include <cstring>

namespace cudaq {

void bindComplexMatrix(py::module_ &mod) {
  py::class_<complex_matrix>(
      mod, "ComplexMatrix",
      "The :class:`ComplexMatrix` is a thin wrapper around a "
      "matrix of complex<double> elements.")
      .def("__init__",
           [](complex_matrix *self, py::object b) {
             auto arr = py::cast<py::ndarray<>>(b);
             if (arr.ndim() != 2)
               throw std::runtime_error("ComplexMatrix requires a 2D array");
             if (arr.shape(0) == 0 || arr.shape(1) == 0)
               throw std::runtime_error("Matrix dimensions must be non-zero.");

             new (self) complex_matrix(arr.shape(0), arr.shape(1));

             // Stride-aware element-wise copy so both row-major (C) and
             // column-major (Fortran) layouts are handled correctly.
             // nanobind strides are counted in elements, not bytes.
             auto *dest = self->get_data(complex_matrix::order::row_major);
             auto *src = static_cast<std::complex<double> *>(arr.data());
             auto stride0 = arr.stride(0);
             auto stride1 = arr.stride(1);
             for (size_t i = 0; i < arr.shape(0); ++i)
               for (size_t j = 0; j < arr.shape(1); ++j)
                 dest[i * arr.shape(1) + j] =
                     src[i * stride0 + j * stride1];
           },
           "Create a :class:`ComplexMatrix` from a buffer of data, such as a "
           "numpy.ndarray.")
      .def(
          "to_numpy",
          [](complex_matrix &op) { return details::cmat_to_numpy(op); },
          "Convert to a NumPy array.")
      .def(
          "num_rows", [](complex_matrix &m) { return m.rows(); },
          "Returns the number of rows in the matrix.")
      .def(
          "num_columns", [](complex_matrix &m) { return m.cols(); },
          "Returns the number of columns in the matrix.")
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
          "dump", [](const complex_matrix &self) { self.dump(); },
          "Prints the matrix to the standard output.")
      .def(
          "__eq__",
          [](const complex_matrix &lhs, const complex_matrix &rhs) {
            return lhs == rhs;
          },
          py::is_operator())
      .def("__str__", &complex_matrix::to_string,
           "Returns the string representation of the matrix.")
      .def(
          "to_numpy",
          [](complex_matrix &m) { return details::cmat_to_numpy(m); },
          "Convert :class:`ComplexMatrix` to numpy.ndarray.");
}

} // namespace cudaq
