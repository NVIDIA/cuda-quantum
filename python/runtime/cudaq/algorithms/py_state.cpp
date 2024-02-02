/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "py_observe.h"
#include "py_state.h"
#include "utils/OpaqueArguments.h"

#include "common/Logger.h"
#include "cudaq/algorithms/get_state.h"

namespace cudaq {

/// @brief If we have any implicit device-to-host data transfers
/// we will store that data here and ensure it is deleted properly.
std::vector<std::unique_ptr<void, std::function<void(void *)>>>
    hostDataFromDevice;

/// @brief Run `cudaq::get_state` on the provided kernel and spin operator.
state pyGetState(kernel_builder<> &kernel, py::args args) {
  // Ensure the user input is correct.
  auto validatedArgs = validateInputArguments(kernel, args);
  kernel.jitCode();
  OpaqueArguments argData;
  packArgs(argData, validatedArgs);
  return details::extractState(
      [&]() mutable { kernel.jitAndInvoke(argData.data()); });
}

/// @brief Bind the get_state cudaq function
void bindPyState(py::module &mod) {

  // A cudaq.State instance can not be constructed from Python code.
  // It should only be constructed from an internal simulator, and that
  // handle to the data (the cudaq.State instance) is returned to the user.
  // State-relevant data can be generated from a cudaq.State with
  // pertinent cudaq.State methods. (e.g. the overlap with an existing state
  // reference). These methods should first and foremost operate on
  // cudaq.State references. For ease of use, we also allow user-specified
  // py::buffer_info adherent types (e.g. Numpy arrays). In the case where
  // a py::buffer_info type is provided, care must be taken to ensure the data
  // is made available in the right memory space (e.g. transferred from host
  // to GPU memory). We should also allow a user to leverage GPU device memory
  // for these types of computations, with something like CuPy.

  py::class_<state>(
      mod, "State", py::buffer_protocol(),
      "A data-type representing the quantum state of the internal simulator. "
      "Returns a state vector by default. If the target is set to "
      "`density-matrix-cpu`, a density matrix will be returned. This type is "
      "not user-constructible and instances can only be retrieved via the "
      "`cudaq.get_state(...)` function. \n")
      .def(
          py::init([](const py::buffer &b) -> cudaq::state {
            throw std::runtime_error(
                "cudaq.State can only be constructed from an internal "
                "simulator. "
                "State comparison operations (e.g. overlap) take Python buffer "
                "data (e.g. NumPy array) as input by default, therefore there "
                "is "
                "no need to construct a cudaq.State from data buffers "
                "directly.");
          }),
          R"#(`State` is not constructible from user code. Throw an error.)#")
      .def_buffer([](const state &self) -> py::buffer_info {
        // This method is used by Pybind to enable interoperability
        // with NumPy array data. We therefore must be careful since the
        // state data may actually be on GPU device.

        // Get the data pointer.
        // Data may be on GPU device, if so we must make a copy to host.
        // If users do not want this copy, they will have to operate apart from
        // Numpy
        void *dataPtr = nullptr;
        if (self.data_holder()->isDeviceData()) {
          // This is device data, transfer to host, which gives us
          // ownership of a new data pointer on host. Store it globally
          // here so we ensure that it gets cleaned up.
          auto numElements = self.data_holder()->getNumElements();
          if (self.data_holder()->getPrecision() ==
              SimulationState::precision::fp32) {
            auto *hostData = new std::complex<float>[numElements];
            self.data_holder()->toHost(hostData, numElements);
            dataPtr = reinterpret_cast<void *>(hostData);
          } else {
            auto *hostData = new std::complex<double>[numElements];
            self.data_holder()->toHost(hostData, numElements);
            dataPtr = reinterpret_cast<void *>(hostData);
          }
          hostDataFromDevice.emplace_back(dataPtr, [](void *data) {
            cudaq::info("freeing data that was copied from GPU device for "
                        "compatibility with NumPy");
            free(data);
          });
        } else
          dataPtr = self.data_holder()->ptr();

        // We need to know the precision of the simulation data
        // to get the data type size and the format descriptor
        auto precision = self.data_holder()->getPrecision();
        auto [dataTypeSize, desc] =
            precision == SimulationState::precision::fp32
                ? std::make_tuple(
                      sizeof(std::complex<float>),
                      py::format_descriptor<std::complex<float>>::format())
                : std::make_tuple(
                      sizeof(std::complex<double>),
                      py::format_descriptor<std::complex<double>>::format());

        // Get the shape of the data. Return buffer info in a
        // correctly shaped manner.
        auto shape = self.get_shape();
        if (shape.size() != 1)
          return py::buffer_info(dataPtr, dataTypeSize, /*itemsize */
                                 desc, 2,               /* ndim */
                                 {shape[0], shape[1]},  /* shape */
                                 {dataTypeSize * static_cast<ssize_t>(shape[1]),
                                  dataTypeSize}, /* strides */
                                 true            /* readonly */
          );

        return py::buffer_info(dataPtr, dataTypeSize, /*itemsize */
                               desc, 1,               /* ndim */
                               {shape[0]},            /* shape */
                               {dataTypeSize});
      })
      .def(
          "device_ptr",
          [](state &self) {
            if (!self.data_holder()->isDeviceData())
              PyErr_WarnEx(PyExc_RuntimeWarning,
                           "[cudaq warning] device pointer requested on state "
                           "vector in host memory. Returning host pointer, do "
                           "not pass to GPU library like cuPy.",
                           1);

            return reinterpret_cast<intptr_t>(self.data_holder()->ptr());
          },
          "Return the GPU device pointer for this `cudaq.State` instance data.")
      .def(
          "__getitem__", [](state &s, std::size_t idx) { return s[idx]; },
          R"#(Return the `index`-th element of the state vector.
          
.. code-block:: python

  # Example:
  import numpy as np

  # Create a simple state vector.
  state = cudaq.get_state(kernel)
  # Return the 0-th entry.
  value = state[0])#")
      .def(
          "__getitem__",
          [](state &s, std::vector<std::size_t> idx) {
            return s(idx[0], idx[1]);
          },
          R"#(Return the element of the density matrix at the provided
index pair.

.. code-block:: python

  # Example:
  import numpy as np

  # Create a simple density matrix.
  cudaq.set_target('density-matrix-cpu')
  densityMatrix = cudaq.get_state(kernel)
  # Return the upper-left most entry of the matrix.
  value = densityMatrix[0,0])#")
      .def(
          "dump",
          [](state &self) {
            std::stringstream ss;
            self.dump(ss);
            py::print(ss.str());
          },
          "Print the state to the console.")
      .def("__str__",
           [](state &self) {
             std::stringstream ss;
             self.dump(ss);
             return ss.str();
           })
      .def(
          "overlap",
          [](state &self, state &other) { return self.overlap(other); },
          "Compute the overlap between the provided :class:`State`'s.")
      .def(
          "overlap",
          [](state &self, py::buffer &other) {
            py::buffer_info info = other.request();

            // Check that the shapes are compatible
            std::size_t otherNumElements = 1;
            for (std::size_t i = 0; std::size_t shapeElement : info.shape) {
              otherNumElements *= shapeElement;
              if (shapeElement != self.get_shape()[i++])
                throw std::runtime_error(
                    "overlap error - invalid shape of input buffer.");
            }

            // Compute the overlap in the case that the
            // input buffer is FP64
            if (info.itemsize == 16) {
              // if this state is FP32, then we have to throw an error
              if (self.data_holder()->getPrecision() ==
                  SimulationState::precision::fp32)
                throw std::runtime_error(
                    "simulation state is FP32 but provided state buffer for "
                    "overlap is FP64.");

              return self.overlap(reinterpret_cast<complex *>(info.ptr),
                                  otherNumElements);
            }

            // Compute the overlap in the case that the
            // input buffer is FP32
            if (info.itemsize == 8) {
              // if this state is FP64, then we have to throw an error
              if (self.data_holder()->getPrecision() ==
                  SimulationState::precision::fp64)
                throw std::runtime_error(
                    "simulation state is FP64 but provided state buffer for "
                    "overlap is FP32.");
              return self.overlap(
                  reinterpret_cast<std::complex<float> *>(info.ptr),
                  otherNumElements);
            }

            // We only support complex f32 and f64 types
            throw std::runtime_error(
                "invalid buffer element type size for overlap computation.");
          },
          "Compute the overlap between the provided :class:`State`'s.")
      .def(
          "overlap",
          [](state &self, py::object other) {
            // Make sure this is a CuPy array
            if (!py::hasattr(other, "data"))
              throw std::runtime_error(
                  "invalid overlap operation on py::object - "
                  "only cupy array supported.");
            auto data = other.attr("data");
            if (!py::hasattr(data, "ptr"))
              throw std::runtime_error(
                  "invalid overlap operation on py::object - "
                  "only cupy array supported.");

            // We know this is a cupy device pointer.

            // Start by ensuring it is of complex type
            auto typeStr = py::str(other.attr("dtype")).cast<std::string>();
            if (typeStr.find("float") != std::string::npos)
              throw std::runtime_error(
                  "CuPy array with only floating point elements passed to "
                  "state.overlap. input must be "
                  "of "
                  "complex float type, please add to your cupy array creation "
                  "`dtype=cupy.complex64` if simulation "
                  "is FP32 and `dtype=cupy.complex128` if simulation if FP64.");
            auto precision = self.data_holder()->getPrecision();
            if (typeStr == "complex64") {
              if (precision == cudaq::SimulationState::precision::fp64)
                throw std::runtime_error(
                    "underlying simulation state is FP64, but "
                    "input cupy array is FP32.");
            } else if (typeStr == "complex128") {
              if (precision == cudaq::SimulationState::precision::fp32)
                throw std::runtime_error(
                    "underlying simulation state is FP32, but "
                    "input cupy array is FP64.");
            } else
              throw std::runtime_error("invalid cupy element type " + typeStr);

            // Compute the number of elements in the other array
            auto numOtherElements = [&]() {
              auto shape = other.attr("shape").cast<py::tuple>();
              std::size_t numElements = 1;
              for (auto el : shape)
                numElements *= el.cast<std::size_t>();
              return numElements;
            }();

            // Cast the device ptr and perform the overlap
            long ptr = data.attr("ptr").cast<long>();
            if (self.data_holder()->getPrecision() ==
                SimulationState::precision::fp32)
              return self.overlap(reinterpret_cast<complex64 *>(ptr),
                                  numOtherElements);

            return self.overlap(reinterpret_cast<complex128 *>(ptr),
                                numOtherElements);
          },
          "Compute overlap with general CuPy device array.");

  mod.def(
      "get_state",
      [](kernel_builder<> &kernel, py::args args) {
        return pyGetState(kernel, args);
      },
      R"#(Return the :class:`State` of the system after execution of the provided `kernel`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.

.. code-block:: python

  # Example:
  import numpy as np

  # Define a kernel that will produced the all |11...1> state.
  kernel = cudaq.make_kernel()
  qubits = kernel.qalloc(3)
  # Prepare qubits in the 1-state.
  kernel.x(qubits)

  # Get the state of the system. This will execute the provided kernel
  # and, depending on the selected target, will return the state as a
  # vector or matrix.
  state = cudaq.get_state(kernel)
  print(state))#");

  py::class_<async_state_result>(
      mod, "AsyncStateResult",
      R"#(A data-type containing the results of a call to :func:`get_state_async`. 
The `AsyncStateResult` models a future-like type, whose 
:class:`State` may be returned via an invocation of the `get` method. This 
kicks off a wait on the current thread until the results are available.
See `future <https://en.cppreference.com/w/cpp/thread/future>`_ 
for more information on this programming pattern.)#")
      .def(
          "get", [](async_state_result &self) { return self.get(); },
          "Return the :class:`State` from the asynchronous `get_state` "
          "accessor execution.\n");

  mod.def(
      "get_state_async",
      [](kernel_builder<> &kernel, py::args args, std::size_t qpu_id) {
        // Ensure the user input is correct.
        auto validatedArgs = validateInputArguments(kernel, args);
        auto &platform = cudaq::get_platform();
        kernel.jitCode();
        auto argDataPtr = std::make_unique<OpaqueArguments>();
        packArgs(*argDataPtr, validatedArgs);
        return cudaq::details::runGetStateAsync(
            [&, argsPtr = std::move(argDataPtr)]() mutable {
              kernel.jitAndInvoke(argsPtr->data());
            },
            platform, qpu_id);
      },
      py::arg("kernel"), py::kw_only(), py::arg("qpu_id") = 0,
      R"#(Asynchronously retrieve the state generated by the given quantum kernel. 
When targeting a quantum platform with more than one QPU, the optional
`qpu_id` allows for control over which QPU to enable. Will return a
future whose results can be retrieved via `future.get()`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.
  qpu_id (Optional[int]): The optional identification for which QPU 
    on the platform to target. Defaults to zero. Key-word only.

Returns:
  :class:`AsyncStateResult`: Quantum state (state vector or density matrix) data).)#");
}

} // namespace cudaq
