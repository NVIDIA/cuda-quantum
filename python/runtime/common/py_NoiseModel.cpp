/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
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

  py::class_<noise_model>(
      mod, "NoiseModel",
      "The `NoiseModel` defines a set of :class:`KrausChannel`'s applied to "
      "specific qubits after the invocation of specified quantum operations.")
      .def(py::init<>(), "Construct an empty noise model.")
      .def(
          "add_channel",
          [](noise_model &self, std::string &opName,
             std::vector<std::size_t> &qubits, kraus_channel &channel) {
            self.add_channel(opName, qubits, channel);
          },
          py::arg("operator"), py::arg("qubits"), py::arg("channel"),
          R"#(Add the given :class:`KrausChannel` to be applied after invocation 
of the specified quantum operation.

Args:
  operator (str): The quantum operator to apply the noise channel to.
  qubits (List[int]): The qubit/s to apply the noise channel to.
  channel (cudaq.KrausChannel): The :class:`KrausChannel` to apply 
    to the specified `operator` on the specified `qubits`.)#")
      .def(
          "get_channels",
          [](noise_model self, const std::string &op,
             std::vector<std::size_t> &qubits) {
            return self.get_channels(op, qubits);
          },
          py::arg("operator"), py::arg("qubits"),
          "Return the :class:`KrausChannel`'s that make up this noise model.");
}

void bindKrausOp(py::module &mod) {
  py::class_<kraus_op>(
      mod, "KrausOperator", py::buffer_protocol(),
      "The `KrausOperator` is represented by a matrix and serves as an element "
      "of a quantum channel such that :code:`Sum Ki Ki^dag = I.`")
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
           "Create a :class:`KrausOperator` from a buffer of data, like a "
           "numpy array.")
      .def_readonly("row_count", &kraus_op::nRows,
                    "The number of rows in the matrix representation of this "
                    ":class:`KrausOperator`.")
      .def_readonly("col_count", &kraus_op::nCols,
                    "The number of columns in the matrix representation of "
                    "this :class:`KrausOperator`.");
}

void bindNoiseChannels(py::module &mod) {
  py::class_<kraus_channel>(mod, "KrausChannel",
                            "The `KrausChannel` is composed of a list of "
                            ":class:`KrausOperator`'s and "
                            "is applied to a specific qubit or set of qubits.")
      .def(py::init<std::vector<kraus_op>>(),
           "Create a :class:`KrausChannel` composed of a list of "
           ":class:`KrausOperator`'s.")
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
           "Create a :class:`KrausChannel` given a list of "
           ":class:`KrausOperator`'s.")
      .def(
          "__getitem__",
          [](kraus_channel &self, std::size_t idx) { return self[idx]; },
          py::arg("index"),
          "Return the :class:`KrausOperator` at the given index in this "
          ":class:`KrausChannel`.")
      .def("append", &kraus_channel::push_back,
           "Add a :class:`KrausOperator` to this :class:`KrausChannel`.");

  py::class_<depolarization_channel, kraus_channel>(
      mod, "DepolarizationChannel",
      "Models the decoherence of the qubit state and phase into a mixture "
      "of the computational basis states, `|0>` and `|1>`. Its constructor "
      "expects a float value, `probability`, representing the probability "
      "that this decay will occur. The qubit will remain untouched, "
      "therefore, with a probability of `1 - probability`.")
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `DepolarizationChannel` with the provided "
           "`probability`.");

  py::class_<amplitude_damping_channel, kraus_channel>(
      mod, "AmplitudeDampingChannel",
      "Models the dissipation of energy due to system interactions with the "
      "environment. Its constructor expects a float value, `probability`, "
      "representing the probablity that the qubit will decay to its ground "
      "state. The probability of the qubit remaining in the same state is "
      "therefore `1 - probability`.")
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `AmplitudeDampingChannel` with the provided "
           "`probability`.");

  py::class_<bit_flip_channel, kraus_channel>(
      mod, "BitFlipChannel",
      "Models the decoherence of the qubit state. Its constructor expects a "
      "float value, `probability`, representing the probability that the qubit "
      "flips from the 1-state to the 0-state, or vice versa. E.g, the "
      "probability of a random X-180 rotation being applied to the qubit. The "
      "probability of the qubit remaining in the same state is therefore `1 - "
      "probability`.")
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `BitFlipChannel` with the provided `probability`.");

  py::class_<phase_flip_channel, kraus_channel>(
      mod, "PhaseFlipChannel",
      "Models the decoherence of the qubit phase. Its constructor expects a "
      "float value, `probability`, representing the probability of a random "
      "Z-180 rotation being applied to the qubit. The probability of the qubit "
      "phase remaining untouched is therefore `1 - probability`.")
      .def(
          py::init<double>(), py::arg("probability"),
          "Initialize the `PhaseFlipChannel` with the provided `probability`.");
}

void bindNoise(py::module &mod) {
  bindNoiseModel(mod);
  bindKrausOp(mod);
  bindNoiseChannels(mod);
}

} // namespace cudaq