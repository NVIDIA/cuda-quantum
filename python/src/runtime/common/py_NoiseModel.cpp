/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "py_NoiseModel.h"

#include "common/NoiseModel.h"
#include "cudaq.h"

#include <iostream>

namespace cudaq {

/// @brief Extract the array data from a buffer_info into our
/// own allocated data pointer.
void extractKrausData(py::buffer_info &info, complex *data) {
  if (info.format != py::format_descriptor<complex>::format())
    throw std::runtime_error(
        "Incompatible buffer format, must be np.complex128.");

  if (info.ndim != 2)
    throw std::runtime_error("Incompatible buffer shape.");

  memcpy(data, info.ptr, sizeof(complex) * (info.shape[0] * info.shape[1]));
}

/// @brief Bind the cudaq::noise_model, kraus_op, and kraus_channel.
void bindNoiseModel(py::module &mod) {

  mod.def("set_noise", &set_noise, "Set the underlying noise model.");
  mod.def("unset_noise", &unset_noise,
          "Clear backend simulation from any existing noise models.");

  py::class_<kraus_op>(mod, "KrausOperator", py::buffer_protocol(),
                       "The KrausOperator is represented by a matrix and "
                       "serves as an element of a "
                       "quantum channel such that Sum Ki Ki^dag = I.")
      .def_buffer([](kraus_op &op) -> py::buffer_info {
        return py::buffer_info(op.data.data(), sizeof(complex),
                               py::format_descriptor<complex>::format(), 2,
                               {op.nRows, op.nCols},
                               {sizeof(complex) * op.nCols, sizeof(complex)});
      })
      .def(py::init([](const py::buffer &b) {
             py::buffer_info info = b.request();
             std::vector<complex> v(info.shape[0] * info.shape[1]);
             extractKrausData(info, v.data());
             return kraus_op(v);
           }),
           "Create a KrausOperator from a buffer of data, like a numpy array.")
      .def_readonly("row_count", &kraus_op::nRows,
                    "The number of rows in the matrix representation of this "
                    "KrausOperator")
      .def_readonly("col_count", &kraus_op::nCols,
                    "The number of columns in the matrix representation of "
                    "this KrausOperator");

  py::class_<kraus_channel>(mod, "KrausChannel",
                            "The KrausChannel is composed of a list of "
                            "KrausOperators and is applied to "
                            "a specific qubit or set of qubits.")
      .def(py::init<std::vector<kraus_op>>(),
           "Create a KrausChannel composed of a list "
           "of kraus_ops.")
      .def(py::init([](py::list ops) {
             std::vector<kraus_op> kops;
             for (std::size_t i = 0; i < ops.size(); i++) {
               auto buffer = ops[i].cast<py::buffer>();
               auto info = buffer.request();
               auto shape = info.shape;
               std::vector<complex> v(shape[0] * shape[1]);
               extractKrausData(info, v.data());
               kops.emplace_back(v);
             }
             return kraus_channel(kops);
           }),
           "Create a KrausChannel given a list of KrausOperators.")
      .def(
          "__getitem__",
          [](kraus_channel &c, std::size_t idx) { return c[idx]; },
          "Return the KrausOperator at the given index in this KrausChannel.")
      .def("append", &kraus_channel::push_back,
           "Add a KrausOperator to this KrausChannel.");

  py::class_<noise_model>(
      mod, "NoiseModel",
      "The cudaq NoiseModel defines a set of KrausChannels applied to "
      "specific qubits after the invocation specified quantum operations.")
      .def(py::init<>(), "Construct an empty noise model.")
      .def(
          "add_channel",
          [](noise_model &n, std::string &opName,
             std::vector<std::size_t> &qubits,
             kraus_channel &c) { n.add_channel(opName, qubits, c); },
          "Add the given KrausChannel to be applied after invocation of the "
          "specified quantum operation.")
      .def(
          "get_channels",
          [](noise_model n, const std::string &op,
             std::vector<std::size_t> qbits) {
            return n.get_channels(op, qbits);
          },
          "Return the KrausChannels that make up this noise model.");

  py::class_<depolarization_channel, kraus_channel>(mod,
                                                    "DepolarizationChannel")
      .def(py::init<double>());

  py::class_<amplitude_damping_channel, kraus_channel>(
      mod, "AmplitudeDampingChannel")
      .def(py::init<double>());

  py::class_<bit_flip_channel, kraus_channel>(mod, "BitFlipChannel")
      .def(py::init<double>());

  py::class_<phase_flip_channel, kraus_channel>(mod, "PhaseFlipChannel")
      .def(py::init<double>());
}
} // namespace cudaq
