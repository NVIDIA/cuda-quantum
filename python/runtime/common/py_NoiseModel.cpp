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
#include <iostream>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace cudaq {

/// @brief Extract the array data from a buffer_info into our
/// own allocated data pointer.
/// This supports 2-d array in either row or column major.
void extractKrausData(py::buffer_info &info, complex *data) {
  if (info.format != py::format_descriptor<complex>::format())
    throw std::runtime_error(
        "Incompatible buffer format, must be np.complex128.");

  if (info.ndim != 2)
    throw std::runtime_error("Incompatible buffer shape " +
                             std::to_string(info.ndim) + ".");

  constexpr bool rowMajor = true;
  typedef Eigen::MatrixXcd::Scalar Scalar;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      RowMajorMat;
  auto strides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
      info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Scalar),
      info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Scalar));
  auto map =
      Eigen::Map<RowMajorMat, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
          static_cast<Scalar *>(info.ptr), info.shape[0], info.shape[1],
          strides);
  RowMajorMat eigenMat(map);
  memcpy(data, eigenMat.data(),
         sizeof(complex) * (info.shape[0] * info.shape[1]));
}

/// @brief Bind the cudaq::noise_model, kraus_op, and kraus_channel.
void bindNoiseModel(py::module &mod) {

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
      .def(py::init<>([mod]() {
             // Create the noise model
             auto model = std::make_unique<noise_model>();

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
                 py::type channelType = py::getattr(mod, name.c_str());
                 auto key = py::hash(channelType);
                 model->register_channel(key, generator);
               }
             }

             return model;
           }),
           "Construct a noise model with all built-in channels pre-registered.")
      .def(
          "register_channel",
          [](noise_model &self, const py::type krausT) {
            auto key = py::hash(krausT);
            std::function<kraus_channel(const std::vector<double> &)> lambda =
                [krausT](const std::vector<double> &p) -> kraus_channel {
              return krausT(p).cast<kraus_channel>();
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

// Need a trampoline class to make this sub-class-able from Python
class PyKrausChannel : public kraus_channel {
public:
  using kraus_channel::kraus_channel;
};

void bindNoiseChannels(py::module &mod) {
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
      .def_readwrite("parameters", &kraus_channel::parameters)
      .def_readwrite("noise_type", &kraus_channel::noise_type)
      .def("get_ops", &kraus_channel::get_ops,
           "Return the :class:`KrausOperator`'s in this :class:`KrausChannel`.")
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
      R"#(Models the decoherence of the qubit state and phase into a mixture "
      of the computational basis states, `|0>` and `|1>`.

      The Kraus Channels are thereby defined to be:

      K_0 = sqrt(1 - probability) * I

      K_1 = sqrt(probability / 3) * X

      K_2 = sqrt(probability / 3) * Y

      K_3 = sqrt(probability / 3) * Z

      where I, X, Y, Z are the 2x2 Pauli matrices.
      
      The constructor expects a float value, `probability`, representing the 
      probability the state decay will occur. The qubit will remain untouched,
      therefore, with a probability of `1 - probability`. And the X,Y,Z operators
      will be applied with a probability of `probability / 3`.
      
      For `probability = 0.0`, the channel will behave noise-free. 
      For `probability = 0.75`, the channel will fully depolarize the state.
      For `probability = 1.0`, the channel will be uniform.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `DepolarizationChannel` with the provided "
           "`probability`.")
      .def_readonly_static(
          "num_parameters", &depolarization_channel::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<amplitude_damping_channel, kraus_channel>(
      mod, "AmplitudeDampingChannel",
      R"#(Models the dissipation of energy due to system interactions with the
      environment. 

      The Kraus Channels are thereby defined to be:

      K_0 = sqrt(1 - probability) * I

      K_1 = sqrt(probability) * 0.5 * (X + iY) 
      
      Its constructor expects a float value, `probability`,
      representing the probablity that the qubit will decay to its ground
      state. The probability of the qubit remaining in the same state is
      therefore `1 - probability`.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `AmplitudeDampingChannel` with the provided "
           "`probability`.")
      .def_readonly_static(
          "num_parameters", &amplitude_damping_channel::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<bit_flip_channel, kraus_channel>(
      mod, "BitFlipChannel",
      R"#(Models the decoherence of the qubit state. Its constructor expects a 
      float value, `probability`, representing the probability that the qubit 
      flips from the 1-state to the 0-state, or vice versa. E.g, the 
      probability of a random X-180 rotation being applied to the qubit. 
      
      The Kraus Channels are thereby defined to be:

      K_0 = sqrt(1 - probability) * I

      K_1 = sqrt(probability ) * X     
      
      The probability of the qubit remaining in the same state is therefore `1 - 
      probability`.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `BitFlipChannel` with the provided `probability`.")
      .def_readonly_static(
          "num_parameters", &bit_flip_channel::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<phase_flip_channel, kraus_channel>(
      mod, "PhaseFlipChannel",
      R"#(Models the decoherence of the qubit phase. Its constructor expects a
      float value, `probability`, representing the probability of a random
      Z-180 rotation being applied to the qubit. 
      
      The Kraus Channels are thereby defined to be:

      K_0 = sqrt(1 - probability) * I

      K_1 = sqrt(probability ) * Z  

      The probability of the qubit phase remaining untouched is therefore
      `1 - probability`.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>(), py::arg("probability"),
           "Initialize the `PhaseFlipChannel` with the provided `probability`.")
      .def_readonly_static(
          "num_parameters", &phase_flip_channel::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<phase_damping, kraus_channel>(
      mod, "PhaseDamping",
      R"#(A Kraus channel that models the single-qubit phase damping error. This
      is similar to AmplitudeDamping, but for phase.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>())
      .def_readonly_static(
          "num_parameters", &phase_damping::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<z_error, kraus_channel>(
      mod, "ZError",
      R"#(A Pauli error that applies the Z operator when an error
      occurs. It is the same as PhaseFlipChannel.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>())
      .def_readonly_static(
          "num_parameters", &z_error::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<x_error, kraus_channel>(
      mod, "XError",
      R"#(A Pauli error that applies the X operator when an error
      occurs. It is the same as BitFlipChannel.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>())
      .def_readonly_static(
          "num_parameters", &x_error::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<y_error, kraus_channel>(
      mod, "YError",
      R"#(A Pauli error that applies the Y operator when an error
      occurs.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>())
      .def_readonly_static(
          "num_parameters", &y_error::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<pauli1, kraus_channel>(
      mod, "Pauli1",
      R"#(A single-qubit Pauli error that applies either an X error, Y error,
      or Z error. The probability of each X, Y, or Z error is supplied as a
      parameter.)#")
      .def(py::init<std::vector<double>>())
      .def_readonly_static(
          "num_parameters", &pauli1::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<pauli2, kraus_channel>(
      mod, "Pauli2",
      R"#(A 2-qubit Pauli error that applies one of the following errors, with
      the probabilities specified as a vector. Possible errors: IX, IY, IZ, XI, XX,
      XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, and ZZ.)#")
      .def(py::init<std::vector<double>>())
      .def_readonly_static(
          "num_parameters", &pauli2::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<depolarization1, kraus_channel>(
      mod, "Depolarization1",
      R"#(The same as DepolarizationChannel (single qubit depolarization))#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>())
      .def_readonly_static(
          "num_parameters", &depolarization1::num_parameters,
          "The number of parameters this channel requires at construction.");

  py::class_<depolarization2, kraus_channel>(
      mod, "Depolarization2",
      R"#(A 2-qubit depolarization error that applies one of the following
      errors. Possible errors: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX,
      ZY, and ZZ.)#")
      .def(py::init<std::vector<double>>())
      .def(py::init<double>())
      .def_readonly_static(
          "num_parameters", &depolarization2::num_parameters,
          "The number of parameters this channel requires at construction.");
}

void bindNoise(py::module &mod) {
  bindNoiseModel(mod);
  bindKrausOp(mod);
  bindNoiseChannels(mod);
}

} // namespace cudaq
