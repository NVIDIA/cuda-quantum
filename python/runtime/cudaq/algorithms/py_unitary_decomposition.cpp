/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/UnitaryDecomposition.h"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {

void bindPyUnitaryDecomposition(py::module &mod) {

  auto decomposition = mod.def_submodule("decompose");

  // Register ZYZComponents structure
  py::class_<cudaq::detail::ZYZComponents>(decomposition, "ZYZComponents")
      .def_readonly("alpha", &cudaq::detail::ZYZComponents::alpha)
      .def_readonly("beta", &cudaq::detail::ZYZComponents::beta)
      .def_readonly("gamma", &cudaq::detail::ZYZComponents::gamma)
      .def_readonly("phase", &cudaq::detail::ZYZComponents::phase);

  // Register KAKComponents structure
  py::class_<cudaq::detail::KAKComponents>(decomposition, "KAKComponents")
      .def_readonly("a0", &cudaq::detail::KAKComponents::a0)
      .def_readonly("a1", &cudaq::detail::KAKComponents::a1)
      .def_readonly("b0", &cudaq::detail::KAKComponents::b0)
      .def_readonly("b1", &cudaq::detail::KAKComponents::b1)
      .def_readonly("x", &cudaq::detail::KAKComponents::x)
      .def_readonly("y", &cudaq::detail::KAKComponents::y)
      .def_readonly("z", &cudaq::detail::KAKComponents::z)
      .def_readonly("phase", &cudaq::detail::KAKComponents::phase);

  // Register the ZYZ decomposition function
  decomposition.def(
      "zyz_components",
      [](py::array_t<std::complex<double>, py::array::c_style> matrix) {
        // Validate input shape
        if (matrix.ndim() != 2 || matrix.shape(0) != 2 ||
            matrix.shape(1) != 2) {
          throw py::value_error("Input must be a 2x2 complex matrix");
        }
        // Create an Eigen matrix from the NumPy array
        Eigen::Matrix2cd eigenMatrix;
        auto matrix_buffer = matrix.unchecked<2>();
        for (py::ssize_t i = 0; i < 2; i++) {
          for (py::ssize_t j = 0; j < 2; j++) {
            eigenMatrix(i, j) = matrix_buffer(i, j);
          }
        }
        return detail::decomposeZYZ(eigenMatrix);
      },
      "Decompose single-qubit unitary into ZYZ Euler angles",
      py::arg("matrix"));

  // Register the KAK decomposition function
  decomposition.def(
      "kak_components",
      [](py::array_t<std::complex<double>, py::array::c_style> matrix) {
        // Validate input shape
        if (matrix.ndim() != 2 || matrix.shape(0) != 4 ||
            matrix.shape(1) != 4) {
          throw py::value_error("Input must be a 4x4 complex matrix");
        }
        // Create an Eigen matrix from the NumPy array
        Eigen::Matrix4cd eigenMatrix;
        auto matrix_buffer = matrix.unchecked<2>();
        for (py::ssize_t i = 0; i < 4; i++) {
          for (py::ssize_t j = 0; j < 4; j++) {
            eigenMatrix(i, j) = matrix_buffer(i, j);
          }
        }
        return detail::decomposeKAK(eigenMatrix);
      },
      "Decompose two-qubit unitary using KAK decomposition", py::arg("matrix"));
}

} // namespace cudaq
