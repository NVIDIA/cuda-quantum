/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "py_NoiseModel.h"
#include "common/EigenDense.h"
#include "common/NoiseModel.h"
#include "cudaq.h"
#include <cstring>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace cudaq {

/// @brief Extract the array data from a nanobind ndarray into our
/// own allocated data pointer.
/// This supports 2-d array in either row or column major.
void extractKrausData(nanobind::ndarray<> &arr, complex *data) {
  size_t rows = arr.shape(0);
  size_t cols = arr.shape(1);

  // Use stride-aware element-wise copy so that both row-major (C) and
  // column-major (Fortran) layouts are handled correctly.
  // nanobind strides are counted in elements, not bytes.
  auto stride0 = arr.stride(0); // row stride
  auto stride1 = arr.stride(1); // col stride
  auto *src = static_cast<std::complex<double> *>(arr.data());

  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      data[i * cols + j] = src[i * stride0 + j * stride1];
}

/// @brief Bind the cudaq::noise_model, kraus_op, and kraus_channel.
void bindNoiseModel(py::module_ &mod) {

  mod.def("set_noise", &set_noise, "Set the underlying noise model.");
  mod.def("unset_noise", &unset_noise,
          "Clear backend simulation from any existing noise models.");
  mod.def(
      "get_noise", []() { return cudaq::get_platform().get_noise(); },
      "Get the underlying noise model.");
  py::class_<noise_model>(
      mod, "NoiseModel",
      "The `NoiseModel` defines a set of :class:`KrausChannel`'s applied to "
      "specific qubits after the invocation of specified quantum operations.")
      .def(
          "__init__",
          [mod](noise_model *self) {
            new (self) noise_model();

            // Define a map of channel names to generator functions
            static std::map<std::string, std::function<kraus_channel(
                                             const std::vector<double> &)>>
                channelGenerators = {
                    {"DepolarizationChannel",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return depolarization_channel(p);
                     }},
                    {"AmplitudeDampingChannel",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return amplitude_damping_channel(p);
                     }},
                    {"BitFlipChannel",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return bit_flip_channel(p);
                     }},
                    {"PhaseFlipChannel",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return phase_flip_channel(p);
                     }},
                    {"XError",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return x_error(p);
                     }},
                    {"YError",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return y_error(p);
                     }},
                    {"ZError",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return z_error(p);
                     }},
                    {"PhaseDamping",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return phase_damping(p);
                     }},
                    {"Pauli1",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return pauli1(p);
                     }},
                    {"Pauli2",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return pauli2(p);
                     }},
                    {"Depolarization1",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return depolarization1(p);
                     }},
                    {"Depolarization2",
                     [](const std::vector<double> &p) -> kraus_channel {
                       return depolarization2(p);
                     }}};

            // Register each channel generator
            for (const auto &[name, generator] : channelGenerators) {
              if (py::hasattr(mod, name.c_str())) {
                py::object channelType = py::getattr(mod, name.c_str());
                auto key = py::hash(channelType);
                self->register_channel(key, generator);
              }
            }
          },
          "Construct a noise model with all built-in channels pre-registered.")
      .def(
          "register_channel",
          [](noise_model &self, const py::object krausT) {
            auto key = py::hash(krausT);
            std::function<kraus_channel(const std::vector<double> &)> lambda =
                [krausT](const std::vector<double> &p) -> kraus_channel {
              return py::cast<kraus_channel>(krausT(p));
            };
            self.register_channel(key, lambda);
          },
          "")
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
          "add_channel",
          [](noise_model &self, std::string &opName,
             const noise_model::PredicateFuncTy &pre) {
            self.add_channel(opName, pre);
          },
          py::arg("operator"), py::arg("pre"),
          R"#(Add the given :class:`KrausChannel` generator callback to be applied after invocation 
of the specified quantum operation.

Args:
  operator (str): The quantum operator to apply the noise channel to.
  pre (Callable): The callback which takes qubits operands and gate parameters and returns a concrete :class:`KrausChannel` to apply 
    to the specified `operator`.)#")
      .def(
          "add_all_qubit_channel",
          [](noise_model &self, std::string &opName, kraus_channel &channel,
             std::size_t num_controls = 0) {
            self.add_all_qubit_channel(opName, channel, num_controls);
          },
          py::arg("operator"), py::arg("channel"), py::arg("num_controls") = 0,

          R"#(Add the given :class:`KrausChannel` to be applied after invocation 
of the specified quantum operation on arbitrary qubits.

Args:
  operator (str): The quantum operator to apply the noise channel to.
  channel (cudaq.KrausChannel): The :class:`KrausChannel` to apply 
    to the specified `operator` on any arbitrary qubits.
  num_controls: Number of control bits. Default is 0 (no control bits).)#")
      .def(
          "get_channels",
          [](noise_model self, const std::string &op,
             const std::vector<std::size_t> &qubits) {
            return self.get_channels(op, qubits);
          },
          py::arg("operator"), py::arg("qubits"),
          "Return the :class:`KrausChannel`'s that make up this noise model.")
      .def(
          "get_channels",
          [](noise_model self, const std::string &op,
             const std::vector<std::size_t> &qubits,
             const std::vector<std::size_t> &controls) {
            return self.get_channels(op, qubits, controls);
          },
          py::arg("operator"), py::arg("qubits"), py::arg("controls"),
          "Return the :class:`KrausChannel`'s that make up this noise model.");
}

void bindKrausOp(py::module_ &mod) {
  py::class_<kraus_op>(
      mod, "KrausOperator",
      "The `KrausOperator` is represented by a matrix and serves as an element "
      "of a quantum channel such that :code:`Sum Ki Ki^dag = I.`")
      .def(
          "__init__",
          [](kraus_op *self, py::object b) {
            // Accept any array-like object via buffer protocol
            auto arr = py::cast<nanobind::ndarray<>>(b);
            if (arr.ndim() != 2)
              throw std::runtime_error("KrausOperator requires a 2D array");
            std::vector<complex> v(arr.shape(0) * arr.shape(1));
            extractKrausData(arr, v.data());
            new (self) kraus_op(v);
          },
          "Create a :class:`KrausOperator` from a buffer of data, like a "
          "numpy array.")
      .def_ro("row_count", &kraus_op::nRows,
              "The number of rows in the matrix representation of this "
              ":class:`KrausOperator`.")
      .def_ro("col_count", &kraus_op::nCols,
              "The number of columns in the matrix representation of "
              "this :class:`KrausOperator`.")
      .def(
          "to_numpy",
          [](kraus_op &self) -> py::object {
            size_t rows = self.nRows;
            size_t cols = self.nCols;
            // kraus_op::data is row-major std::vector<complex>
            // Make a copy so the numpy array owns its data.
            auto *copy = new std::complex<double>[rows * cols];
            std::memcpy(copy, self.data.data(),
                        sizeof(std::complex<double>) * rows * cols);

            py::capsule owner(copy, [](void *p) noexcept {
              delete[] static_cast<std::complex<double> *>(p);
            });

            size_t shape[2] = {rows, cols};
            return py::cast(py::ndarray<py::numpy, std::complex<double>>(
                copy, 2, shape, owner));
          },
          "Convert to a NumPy array.")
      .def(
          "__array__",
          [](py::object self, py::args, py::kwargs) {
            return self.attr("to_numpy")();
          },
          "NumPy array protocol support.");
}

// Need a trampoline class to make this sub-class-able from Python
class PyKrausChannel : public kraus_channel {
public:
  using kraus_channel::kraus_channel;
};

void bindNoiseChannels(py::module_ &mod) {
  py::enum_<cudaq::noise_model_type>(mod, "NoiseModelType")
      .value("Unknown", cudaq::noise_model_type::unknown)
      .value("DepolarizationChannel",
             cudaq::noise_model_type::depolarization_channel)
      .value("AmplitudeDampingChannel",
             cudaq::noise_model_type::amplitude_damping_channel)
      .value("BitFlipChannel", cudaq::noise_model_type::bit_flip_channel)
      .value("PhaseFlipChannel", cudaq::noise_model_type::phase_flip_channel)
      .value("XError", cudaq::noise_model_type::x_error)
      .value("YError", cudaq::noise_model_type::y_error)
      .value("ZError", cudaq::noise_model_type::z_error)
      .value("AmplitudeDamping", cudaq::noise_model_type::amplitude_damping)
      .value("PhaseDamping", cudaq::noise_model_type::phase_damping)
      .value("Pauli1", cudaq::noise_model_type::pauli1)
      .value("Pauli2", cudaq::noise_model_type::pauli2)
      .value("Depolarization1", cudaq::noise_model_type::depolarization1)
      .value("Depolarization2", cudaq::noise_model_type::depolarization2);

  py::class_<kraus_channel, PyKrausChannel>(
      mod, "KrausChannel", py::dynamic_attr(),
      "The `KrausChannel` is composed of a list of "
      ":class:`KrausOperator`'s and "
      "is applied to a specific qubit or set of qubits.")
      .def(py::init<>(), "Create an empty :class:`KrausChannel`")
      .def(py::init<const std::vector<kraus_op> &>(),
           "Create a :class:`KrausChannel` composed of a list of "
           ":class:`KrausOperator`'s.")
      .def(
          "__init__",
          [](kraus_channel *self, py::list ops) {
            std::vector<kraus_op> kops;
            for (std::size_t i = 0; i < ops.size(); i++) {
              py::object item = ops[i];
              // Try to cast to ndarray
              try {
                auto arr = py::cast<nanobind::ndarray<>>(item);
                if (arr.ndim() != 2)
                  throw std::runtime_error(
                      "Each Kraus operator must be a 2D array");
                std::vector<complex> v(arr.shape(0) * arr.shape(1));
                extractKrausData(arr, v.data());
                kops.emplace_back(v);
              } catch (const py::cast_error &) {
                throw std::runtime_error(
                    "KrausChannel expects a list of 2D complex arrays");
              }
            }
            new (self) kraus_channel(kops);
          },
          "Create a :class:`KrausChannel` given a list of "
          ":class:`KrausOperator`'s.")
      .def_rw("parameters", &kraus_channel::parameters)
      .def_rw("noise_type", &kraus_channel::noise_type)
      .def("get_ops", &kraus_channel::get_ops,
           "Return the :class:`KrausOperator`'s in this :class:`KrausChannel`.")
      .def(
          "__getitem__",
          [](kraus_channel &self, std::size_t idx) { return self[idx]; },
          py::arg("index"),
          "Return the :class:`KrausOperator` at the given index in this "
          ":class:`KrausChannel`.")
      .def(
          "append",
          [](kraus_channel &self, kraus_op op) { self.push_back(op); },
          py::arg("operator"),
          "Add a :class:`KrausOperator` to this :class:`KrausChannel`.");

#define BIND_NOISE_CHANNEL(CppType, PyName, DocString)                         \
  py::class_<CppType, kraus_channel>(mod, PyName, DocString)                   \
      .def(py::init<std::vector<double>>())                                    \
      .def(py::init<double>(), py::arg("probability"),                         \
           "Initialize the `" PyName "` with the provided `probability`.")     \
      .def_static(                                                             \
          "get_num_parameters",                                                \
          []() -> std::size_t { return CppType::num_parameters; },             \
          "The number of parameters this channel requires at "                 \
          "construction.");

  BIND_NOISE_CHANNEL(
      depolarization_channel, "DepolarizationChannel",
      R"#(Models the decoherence of the qubit state and phase into a mixture 
      of the computational basis states.)#")

  BIND_NOISE_CHANNEL(
      amplitude_damping_channel, "AmplitudeDampingChannel",
      R"#(Models the dissipation of energy due to system interactions with the
      environment.)#")

  BIND_NOISE_CHANNEL(bit_flip_channel, "BitFlipChannel",
                     R"#(Models the decoherence of the qubit state.)#")

  BIND_NOISE_CHANNEL(phase_flip_channel, "PhaseFlipChannel",
                     R"#(Models the decoherence of the qubit phase.)#")

  BIND_NOISE_CHANNEL(
      phase_damping, "PhaseDamping",
      R"#(A Kraus channel that models the single-qubit phase damping error.)#")

  BIND_NOISE_CHANNEL(
      z_error, "ZError",
      R"#(A Pauli error that applies the Z operator when an error occurs.)#")

  BIND_NOISE_CHANNEL(
      x_error, "XError",
      R"#(A Pauli error that applies the X operator when an error occurs.)#")

  BIND_NOISE_CHANNEL(
      y_error, "YError",
      R"#(A Pauli error that applies the Y operator when an error occurs.)#")

#undef BIND_NOISE_CHANNEL

  // Pauli1 and Pauli2 take vector<double> only (no single double constructor)
  py::class_<pauli1, kraus_channel>(mod, "Pauli1",
                                    R"#(A single-qubit Pauli error.)#")
      .def(py::init<std::vector<double>>())
      .def_static(
          "get_num_parameters",
          []() -> std::size_t { return pauli1::num_parameters; },
          "The number of parameters this channel requires at construction.");

  py::class_<pauli2, kraus_channel>(mod, "Pauli2",
                                    R"#(A 2-qubit Pauli error.)#")
      .def(py::init<std::vector<double>>())
      .def_static(
          "get_num_parameters",
          []() -> std::size_t { return pauli2::num_parameters; },
          "The number of parameters this channel requires at construction.");

  py::class_<depolarization1, kraus_channel>(
      mod, "Depolarization1",
      R"#(The same as DepolarizationChannel (single qubit depolarization))#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>())
      .def_static(
          "get_num_parameters",
          []() -> std::size_t { return depolarization1::num_parameters; },
          "The number of parameters this channel requires at construction.");

  py::class_<depolarization2, kraus_channel>(
      mod, "Depolarization2", R"#(A 2-qubit depolarization error.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>())
      .def_static(
          "get_num_parameters",
          []() -> std::size_t { return depolarization2::num_parameters; },
          "The number of parameters this channel requires at construction.");
}

void bindNoise(py::module_ &mod) {
  bindNoiseModel(mod);
  bindKrausOp(mod);
  bindNoiseChannels(mod);
}

} // namespace cudaq
