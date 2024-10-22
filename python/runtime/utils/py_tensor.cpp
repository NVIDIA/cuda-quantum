/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <limits>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "cudaq/utils/tensor.h"

namespace py = pybind11;

namespace cudaq {

template <typename Scalar>
void declare_tensor(py::module_ &m, const std::string &typestr) {
  using Tensor = cudaq::tensor<Scalar>;
  std::string pyclass_name = std::string("Tensor_") + typestr;

  py::class_<Tensor>(m, pyclass_name.c_str(), py::buffer_protocol(), R"#(
A multi-dimensional tensor class for complex-valued data.

This class represents a multi-dimensional array (tensor) of complex numbers,
providing various operations and supporting the Python buffer protocol for
efficient data access and interoperability with NumPy.

Attributes:
-----------
The tensor supports the buffer protocol, allowing direct access to its
underlying data. This enables efficient interoperability with NumPy arrays
and other Python libraries that support the buffer protocol.

Examples:
---------
>>> t = Tensor([2, 2])
>>> t[0, 0] = 1+2j
>>> t[1, 1] = 3-4j
>>> print(t[0, 0])
(1+2j)
>>> print(t.shape())
[2, 2]
>>> print(t.size())
4

>>> import numpy as np
>>> np_array = np.array(t, copy=False)
>>> print(np_array)
[[1.+2.j 0.+0.j]
 [0.+0.j 3.-4.j]]

Notes:
------
- The tensor uses row-major order (C-style) for multi-dimensional indexing.
- Complex numbers are stored as pairs of double-precision floating-point numbers.
- The buffer protocol implementation allows zero-copy access from NumPy,
  enabling efficient data sharing between C++ and Python.
)#")
      .def(py::init<>())
      .def(py::init<const std::vector<std::size_t> &>())
      .def(py::init([](py::array_t<Scalar, py::array::c_style> b) {
        py::buffer_info info = b.request();
        std::vector<std::size_t> shape(info.shape.begin(), info.shape.end());
        return Tensor(static_cast<Scalar *>(info.ptr), shape);
      }))
      .def("rank", &Tensor::rank)
      .def("size", &Tensor::size)
      .def("shape", &Tensor::shape)
      .def(
          "__setitem__",
          [](Tensor &t, const std::vector<size_t> &indices, Scalar value) {
            t.at(indices) = value;
          },
          "Set the tensor element at the specified indices to the given value.")
      .def(
          "at",
          [](Tensor &t, const std::vector<size_t> &indices) {
            return t.at(indices);
          },
          py::return_value_policy::reference)
      .def("copy",
           [](Tensor &t, py::array_t<Scalar, py::array::c_style> b) {
             py::buffer_info info = b.request();
             std::vector<std::size_t> shape(info.shape.begin(),
                                            info.shape.end());
             t.copy(static_cast<Scalar *>(info.ptr), shape);
           })
      .def("take",
           [](Tensor &t, py::array_t<Scalar, py::array::c_style> b) {
             py::buffer_info info = b.request();
             std::vector<std::size_t> shape(info.shape.begin(),
                                            info.shape.end());
             t.take(static_cast<Scalar *>(info.ptr), shape);
           })
      .def("borrow",
           [](Tensor &t, py::array_t<Scalar, py::array::c_style> b) {
             py::buffer_info info = b.request();
             std::vector<std::size_t> shape(info.shape.begin(),
                                            info.shape.end());
             t.borrow(static_cast<Scalar *>(info.ptr), shape);
           })
      .def("dump", &Tensor::dump)
      .def_buffer([](Tensor &t) -> py::buffer_info {
        auto calculateStrides = [](const std::vector<std::size_t> &shape_) {
          std::vector<size_t> strides(shape_.size());
          strides.back() = sizeof(Scalar);
          for (int i = shape_.size() - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape_[i + 1];

          return strides;
        };

        return py::buffer_info(
            t.data(),                                // Pointer to buffer
            sizeof(Scalar),                          // Size of one scalar
            py::format_descriptor<Scalar>::format(), // Python struct-style
                                                     // format descriptor
            t.shape().size(),                        // Number of dimensions
            t.shape(),                               // Buffer dimensions
            calculateStrides(t.shape()) // Strides (in bytes) for each index
        );
      });
}

void bindTensor(py::module &mod) {
  auto utils = mod.def_submodule("utils");
  declare_tensor<float>(utils, "float");
  declare_tensor<std::complex<float>>(utils, "complex64");
  declare_tensor<int>(utils, "int32");
  declare_tensor<std::size_t>(utils, "int64");
  declare_tensor<std::complex<double>>(utils, "complex128");
  declare_tensor<uint8_t>(utils, "uint8");
}

} // namespace cudaq
