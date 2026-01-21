/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_state.h"
#include "LinkedLibraryHolder.h"
#include "common/ArgumentWrapper.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "cudaq/algorithms/get_state.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace cudaq;

// FIXME: This is using a thread unsafe global?

/// If we have any implicit device-to-host data transfers we will store that
/// data here and ensure it is deleted properly.
static std::vector<std::unique_ptr<void, std::function<void(void *)>>>
    hostDataFromDevice;

static std::vector<int> bitStringToIntVec(const std::string &bitString) {
  // Check that this is a valid bit string.
  const bool isValidBitString =
      std::all_of(bitString.begin(), bitString.end(),
                  [](char c) { return c == '0' || c == '1'; });
  if (!isValidBitString)
    throw std::invalid_argument("Invalid bitstring: " + bitString);
  std::vector<int> result;
  result.reserve(bitString.size());
  for (const auto c : bitString)
    result.emplace_back(c == '0' ? 0 : 1);
  return result;
}

/// @brief Run `cudaq::get_state` on the provided kernel and spin operator.
static state get_state_impl(const std::string &shortName, MlirModule mod,
                            MlirType retTy, py::args args) {
  auto closure = [=]() {
    return marshal_and_launch_module(shortName, mod, retTy, args);
  };
  return details::extractState(std::move(closure));
}

static std::future<state>
get_state_async_impl(const std::string &shortName, MlirModule module,
                     MlirType returnTy, std::size_t qpu_id, py::args args) {
  // Launch the asynchronous execution.
  auto mod = unwrap(module);
  std::string kernelName = shortName;
  auto retTy = unwrap(returnTy);
  auto &platform = get_platform();
  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, args, fnOp);

  py::gil_scoped_release release;
  return details::runGetStateAsync(
      detail::make_copyable_function([opaques = std::move(opaques), kernelName,
                                      retTy, mod = mod.clone()]() mutable {
        [[maybe_unused]] auto result =
            clean_launch_module(kernelName, mod, retTy, opaques);
      }),
      platform, qpu_id);
}

namespace {
struct state_view : public state {
  state_view(const state &st) : state(st) {}
};
} // namespace

/// @brief Python implementation of the `RemoteSimulationState`.
// Note: Python kernel arguments are wrapped hence need to be unwrapped
// accordingly.
class PyRemoteSimulationState : public RemoteSimulationState {
  // Holder of args data for clean-up.
  OpaqueArguments *argsData;
  mlir::ModuleOp kernelMod;

public:
  PyRemoteSimulationState(const std::string &in_kernelName, ArgWrapper args,
                          OpaqueArguments *argsDataToOwn, std::size_t size,
                          std::size_t returnOffset)
      : argsData(argsDataToOwn), kernelMod(args.mod) {
    this->kernelName = in_kernelName;
    this->args = argsData->getArgs();
  }

  void execute() const override {
    if (!state) {
      auto &platform = get_platform();
      // Create an execution context, indicate this is for
      // extracting the state representation
      ExecutionContext context("extract-state");
      // Perform the usual pattern set the context,
      // execute and then reset
      platform.set_exec_ctx(&context);
      // Note: in Python, the platform QPU (`PyRemoteSimulatorQPU`) expects an
      // ModuleOp pointer as the first element in the args array in StreamLined
      // mode.
      auto args = argsData->getArgs();
      args.insert(args.begin(),
                  const_cast<void *>(static_cast<const void *>(&kernelMod)));
      platform.launchKernel(kernelName, args);
      platform.reset_exec_ctx();
      state = std::move(context.simulationState);
    }
  }

  std::complex<double> overlap(const SimulationState &other) override {
    const auto &otherState =
        dynamic_cast<const PyRemoteSimulationState &>(other);
    auto &platform = get_platform();
    ExecutionContext context("state-overlap");
    context.overlapComputeStates =
        std::make_pair(static_cast<const SimulationState *>(this),
                       static_cast<const SimulationState *>(&otherState));
    platform.set_exec_ctx(&context);
    auto args = argsData->getArgs();
    args.insert(args.begin(),
                const_cast<void *>(static_cast<const void *>(&kernelMod)));
    platform.launchKernel(kernelName, args);
    platform.reset_exec_ctx();
    assert(context.overlapResult.has_value());
    return context.overlapResult.value();
  }

  virtual ~PyRemoteSimulationState() override { delete argsData; }
};

/// @brief Run `cudaq::get_state` for remote execution targets on the provided
/// kernel and args
state pyGetStateRemote(py::object kernel, py::args args) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("uniqName").cast<std::string>();
  auto kernelMod = kernel.attr("qkeModule").cast<MlirModule>();
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);
#if 0
  auto [argWrapper, size, returnOffset] =
      pyCreateNativeKernel(kernelName, kernelMod, *argData);
#endif
  return state(new PyRemoteSimulationState(kernelName, /*argWrapper*/ {},
                                           argData,
                                           /*size*/ 0, /*returnOffset*/ 0));
}

/// @brief Python implementation of the `QPUState`.
// Note: Python kernel arguments are wrapped hence need to be unwrapped
// accordingly.
class PyQPUState : public QPUState {
  // Holder of args data for clean-up.
  OpaqueArguments *argsData;

public:
  PyQPUState(const std::string &in_kernelName, OpaqueArguments *argsDataToOwn)
      : argsData(argsDataToOwn) {
    this->kernelName = in_kernelName;
    this->args = argsData->getArgs();
  }

  virtual ~PyQPUState() override { delete argsData; }
};

/// @brief Run `cudaq::get_state` for qpu targets on the provided
/// kernel and args
state pyGetStateQPU(py::object kernel, py::args args) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("uniqName").cast<std::string>();
  auto kernelMod = kernel.attr("qkeModule").cast<MlirModule>();
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);
#if 0
  auto [argWrapper, size, returnOffset] =
      pyCreateNativeKernel(kernelName, kernelMod, *argData);
#endif
  return state(new PyQPUState(kernelName, argData));
}

state pyGetStateLibraryMode(py::object kernel, py::args args) {
  return details::extractState([&]() mutable {
    if (0 == args.size())
      kernel();
    else {
      std::vector<py::object> argsData;
      for (size_t i = 0; i < args.size(); i++) {
        py::object arg = args[i];
        argsData.emplace_back(std::forward<py::object>(arg));
      }
      kernel(std::move(argsData));
    }
  });
}

static py::buffer_info getCupyBufferInfo(py::buffer cupy_buffer) {
  // Note: cupy 13.5+ arrays will bind (overload resolution) to a py::buffer
  // type. However, we cannot access the underlying buffer info via a
  // `.request()` as it will throw unless that is managed memory. Here, we
  // retrieve and construct buffer_info from the CuPy array interface.

  if (!py::hasattr(cupy_buffer, "__cuda_array_interface__")) {
    throw std::runtime_error("Buffer is not a CuPy array");
  }

  py::dict cupy_array_info = cupy_buffer.attr("__cuda_array_interface__");
  // Ref: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
  // example: {'shape': (2, 2), 'typestr': '<c16', 'descr': [('', '<c16')],
  // 'stream': 1, 'version': 3, 'strides': None, 'data': (140222144708608,
  // False)}
  py::tuple dataInfo = cupy_array_info["data"].cast<py::tuple>();
  void *dataPtr = (void *)dataInfo[0].cast<int64_t>();
  const bool readOnly = dataInfo[1].cast<bool>();
  auto shapeTuple = cupy_array_info["shape"].cast<py::tuple>();
  std::vector<std::size_t> extents;
  for (std::size_t i = 0; i < shapeTuple.size(); i++) {
    extents.push_back(shapeTuple[i].cast<std::size_t>());
  }
  const std::string typeStr = cupy_array_info["typestr"].cast<std::string>();
  if (typeStr != "<c16" && typeStr != "<c8") {
    throw std::runtime_error("Unsupported typestr in CuPy array: " + typeStr +
                             ". Supported types are: <c16 and <c8.");
  }

  const bool isDoublePrecision = typeStr == "<c16";

  auto [dataTypeSize, desc] =
      !isDoublePrecision
          ? std::make_tuple(
                sizeof(std::complex<float>),
                py::format_descriptor<std::complex<float>>::format())
          : std::make_tuple(
                sizeof(std::complex<double>),
                py::format_descriptor<std::complex<double>>::format());

  std::vector<ssize_t> strides(extents.size(), dataTypeSize);
  for (size_t i = 1; i < extents.size(); ++i)
    strides[i] = strides[i - 1] * extents[i - 1];

  return py::buffer_info(dataPtr, dataTypeSize, /*itemsize */
                         desc, extents.size(),  /* ndim */
                         extents,               /* shape */
                         strides,               /* strides */
                         readOnly               /* readonly */
  );
}

static cudaq::state createStateFromPyBuffer(py::buffer data,
                                            LinkedLibraryHolder &holder) {
  const bool isHostData = !py::hasattr(data, "__cuda_array_interface__");
  // Check that the target is GPU-based, i.e., can handle device
  // pointer.
  if (!holder.getTarget().config.GpuRequired && !isHostData)
    throw std::runtime_error(
        fmt::format("Current target '{}' does not support CuPy arrays.",
                    holder.getTarget().name));

  auto info = isHostData ? data.request() : getCupyBufferInfo(data);
  if (info.shape.size() > 2)
    throw std::runtime_error(
        "state.from_data only supports 1D or 2D array data.");
  if (info.format != py::format_descriptor<std::complex<float>>::format() &&
      info.format != py::format_descriptor<std::complex<double>>::format())
    throw std::runtime_error(
        "A numpy array with only floating point elements passed to "
        "`state.from_data`. Input must be of complex float type. Please add to "
        "your array creation `dtype=numpy.complex64` if simulation is FP32 and "
        "`dtype=numpy.complex128` if simulation is FP64, or "
        "`dtype=cudaq.complex()` for precision-agnostic code.");

  if (!isHostData || info.shape.size() == 1) {
    if (info.format == py::format_descriptor<std::complex<float>>::format())
      return state::from_data(std::make_pair(
          reinterpret_cast<std::complex<float> *>(info.ptr), info.size));

    return state::from_data(std::make_pair(
        reinterpret_cast<std::complex<double> *>(info.ptr), info.size));
  } else { // 2D array
    const std::size_t rows = info.shape[0];
    const std::size_t cols = info.shape[1];
    if (rows != cols)
      throw std::runtime_error(
          "state.from_data 2D array (density matrix) input must be "
          "square matrix data.");
    const bool isDoublePrecision =
        info.format == py::format_descriptor<std::complex<double>>::format();
    const int64_t dataSize = isDoublePrecision ? sizeof(std::complex<double>)
                                               : sizeof(std::complex<float>);
    const bool rowMajor =
        info.strides[1] ==
        dataSize; // check row-major: second stride == element size
    const cudaq::complex_matrix::order matOrder =
        rowMajor ? cudaq::complex_matrix::order::row_major
                 : cudaq::complex_matrix::order::column_major;
    const cudaq::complex_matrix::Dimensions dim = {rows, cols};
    if (isDoublePrecision)
      return state::from_data(cudaq::complex_matrix(
          std::vector<cudaq::complex_matrix::value_type>(
              reinterpret_cast<std::complex<double> *>(info.ptr),
              reinterpret_cast<std::complex<double> *>(info.ptr) + info.size),
          dim, matOrder));

    return state::from_data(cudaq::complex_matrix(
        std::vector<cudaq::complex_matrix::value_type>(
            reinterpret_cast<std::complex<float> *>(info.ptr),
            reinterpret_cast<std::complex<float> *>(info.ptr) + info.size),
        dim, matOrder));
  }
}

/// @brief Bind the get_state cudaq function
void cudaq::bindPyState(py::module &mod, LinkedLibraryHolder &holder) {
  py::enum_<InitialState>(mod, "InitialStateType",
                          "Enumeration describing the initial state "
                          "type to be created in the backend")
      .value("ZERO", InitialState::ZERO)
      .value("UNIFORM", InitialState::UNIFORM)
      .export_values();

  py::class_<SimulationState::Tensor>(
      mod, "Tensor",
      "The `Tensor` describes a pointer to simulation data as well as the rank "
      "and extents for that tensorial data it represents.")
      .def("data",
           [](SimulationState::Tensor &tensor) {
             return reinterpret_cast<intptr_t>(tensor.data);
           })
      .def_readonly("extents", &SimulationState::Tensor::extents)
      .def("get_rank", &SimulationState::Tensor::get_rank)
      .def("get_element_size", &SimulationState::Tensor::element_size)
      .def("get_num_elements", &SimulationState::Tensor::get_num_elements);

  py::class_<state>(mod, "State", "FIXME: document")
      .def(
          "__len__",
          [](state &self) {
            if (self.get_num_tensors() > 1 ||
                self.get_tensor().extents.size() != 1)
              throw std::runtime_error(
                  "len(state) only supported for state-vector like data.");

            return self.get_tensor().extents[0];
          },
          "For vector-like state data, return the number of state vector "
          "elements.")
      .def(
          "num_qubits", [](state &self) { return self.get_num_qubits(); },
          "Returns the number of qubits represented by this state.")
      .def(
          "get_state_refval",
          [](const state &s) -> std::intptr_t {
            return reinterpret_cast<std::intptr_t>(&s);
          },
          "Convert the address of the state object to an integer.")
      .def_static(
          "from_data",
          [&](py::buffer data) {
            return createStateFromPyBuffer(data, holder);
          },
          "Return a state from data.")
      .def_static(
          "from_data",
          [&holder](const std::vector<py::buffer> &tensors) {
            const bool isHostData =
                tensors.empty() ||
                !py::hasattr(tensors[0], "__cuda_array_interface__");
            // Check that the target is GPU-based, i.e., can handle device
            // pointer.
            if (!holder.getTarget().config.GpuRequired && !isHostData)
              throw std::runtime_error(fmt::format(
                  "Current target '{}' does not support CuPy arrays.",
                  holder.getTarget().name));
            TensorStateData tensorData;
            for (auto &tensor : tensors) {
              auto info =
                  isHostData ? tensor.request() : getCupyBufferInfo(tensor);
              const std::vector<std::size_t> extents(info.shape.begin(),
                                                     info.shape.end());
              tensorData.emplace_back(
                  std::pair<const void *, std::vector<std::size_t>>{info.ptr,
                                                                    extents});
            }
            return state::from_data(tensorData);
          },
          "Return a state from matrix product state tensor data.")
      .def_static(
          "from_data",
          [](const std::vector<SimulationState::Tensor> &tensors) {
            TensorStateData tensorData;
            for (auto &tensor : tensors) {

              tensorData.emplace_back(
                  std::pair<const void *, std::vector<std::size_t>>{
                      tensor.data, tensor.extents});
            }
            return state::from_data(tensorData);
          },
          "Return a state from matrix product state tensor data.")
      .def_static(
          "from_data",
          [](const py::list &tensors) {
            // Note: we must use Python type (py::list) for proper overload
            // resolution. The overload for py::object, intended for cupy arrays
            // (implementing Python array interface), may be overshadowed by any
            // std::vector overloads.
            TensorStateData tensorData;
            for (auto &tensor : tensors) {
              // Make sure this is a CuPy array
              if (!py::hasattr(tensor, "data"))
                throw std::runtime_error(
                    "invalid from_data operation on py::object - "
                    "only cupy array supported.");
              auto data = tensor.attr("data");
              if (!py::hasattr(data, "ptr"))
                throw std::runtime_error(
                    "invalid from_data operation on py::object tensors - "
                    "only cupy array supported.");

              // We know this is a cupy device pointer. Start by ensuring it is
              // of proper complex type
              auto typeStr = py::str(tensor.attr("dtype")).cast<std::string>();
              if (typeStr != "complex128")
                throw std::runtime_error(
                    "invalid from_data operation on py::object tensors - "
                    "only cupy complex128 tensors supported.");
              auto shape = tensor.attr("shape").cast<py::tuple>();
              std::vector<std::size_t> extents;
              for (auto el : shape)
                extents.emplace_back(el.cast<std::size_t>());
              long ptr = data.attr("ptr").cast<long>();
              tensorData.emplace_back(
                  std::pair<const void *, std::vector<std::size_t>>{
                      reinterpret_cast<std::complex<double> *>(ptr), extents});
            }
            return state::from_data(tensorData);
          },
          "Return a state from matrix product state tensor data (as CuPy "
          "ndarray).")
      .def_static(
          "from_data",
          [&holder](py::object opaqueData) {
            // Note: This overload is no longer needed from cupy 13.5+ onward.
            // We can remove it in future releases.
            // Make sure this is a CuPy array
            if (!py::hasattr(opaqueData, "data"))
              throw std::runtime_error(
                  "invalid from_data operation on py::object - "
                  "only cupy array supported.");
            auto data = opaqueData.attr("data");
            if (!py::hasattr(data, "ptr"))
              throw std::runtime_error(
                  "invalid from_data operation on py::object - "
                  "only cupy array supported.");

            // We know this is a cupy device pointer. Start by ensuring it is of
            // complex type
            auto typeStr =
                py::str(opaqueData.attr("dtype")).cast<std::string>();
            if (typeStr.find("float") != std::string::npos)
              throw std::runtime_error(
                  "CuPy array with only floating point elements passed to "
                  "state.from_data. input must be of complex float type, "
                  "please add to your cupy array creation "
                  "`dtype=cupy.complex64` if simulation is FP32 and "
                  "`dtype=cupy.complex128` if simulation if FP64.");

            // Compute the number of elements in the array
            std::vector<std::size_t> extents;
            auto numElements = [&]() {
              auto shape = opaqueData.attr("shape").cast<py::tuple>();
              std::size_t numElements = 1;
              for (auto el : shape) {
                numElements *= el.cast<std::size_t>();
                extents.emplace_back(el.cast<std::size_t>());
              }
              return numElements;
            }();

            long ptr = data.attr("ptr").cast<long>();
            if (holder.getTarget().name == "dynamics") {
              // For dynamics, we need to send on the extents to distinguish
              // state vector vs density matrix.
              TensorStateData tensorData{
                  std::pair<const void *, std::vector<std::size_t>>{
                      reinterpret_cast<std::complex<double> *>(ptr), extents}};
              return state::from_data(tensorData);
            }

            // Check that the target is GPU-based, i.e., can handle device
            // pointer.
            if (!holder.getTarget().config.GpuRequired)
              throw std::runtime_error(fmt::format(
                  "Current target '{}' does not support CuPy arrays.",
                  holder.getTarget().name));

            if (typeStr == "complex64")
              return state::from_data(std::make_pair(
                  reinterpret_cast<std::complex<float> *>(ptr), numElements));
            else if (typeStr == "complex128")
              return state::from_data(std::make_pair(
                  reinterpret_cast<std::complex<double> *>(ptr), numElements));
            else
              throw std::runtime_error("invalid cupy element type " + typeStr);
          },
          "Return a state from CuPy device array.")
      .def("is_on_gpu", &state::is_on_gpu,
           "Return True if this state is on the GPU.")
      .def(
          "getTensor",
          [](state &self, std::size_t idx) { return self.get_tensor(idx); },
          py::arg("idx") = 0,
          "Return the `idx` tensor making up this state representation.")
      .def(
          "getTensors", [](state &self) { return self.get_tensors(); },
          "Return all the tensors that comprise this state representation.")
      .def(
          "__getitem__",
          [](state &s, int idx) {
            // Support Pythonic negative index
            if (idx < 0)
              idx += (1 << s.get_num_qubits());
            return s[idx];
          },
          R"#(Return the `index`-th element of the state vector.
          
.. code-block:: python

  # Example:
  # Create a simple state vector.
  # Requires state-vector simulator
  state = cudaq.get_state(kernel)
  # Return the 0-th entry.
  value = state[0])#")
      .def(
          "__getitem__",
          [](state &s, std::vector<int> idx) {
            if (idx.size() != 2)
              throw std::runtime_error("Density matrix needs 2 indices; " +
                                       std::to_string(idx.size()) +
                                       " provided.");
            for (auto &val : idx)
              // Support Pythonic negative index
              if (val < 0)
                val += (1 << s.get_num_qubits());
            return s(idx[0], idx[1]);
          },
          R"#(Return the element of the density matrix at the provided
index pair.

.. code-block:: python

  # Example:
  # Create a simple density matrix.
  cudaq.set_target('density-matrix-cpu')
  densityMatrix = cudaq.get_state(kernel)
  # Return the upper-left most entry of the matrix.
  value = densityMatrix[0,0])#")
      .def(
          "amplitude",
          [](state &s, std::vector<int> basisState) {
            return s.amplitude(basisState);
          },
          R"#(Return the amplitude of a state in computational basis.
          
.. code-block:: python

  # Example:
  # Create a simulation state.
  state = cudaq.get_state(kernel)
  # Return the amplitude of |0101>, assuming this is a 4-qubit state.
  amplitude = state.amplitude([0,1,0,1]))#")
      .def(
          "amplitude",
          [](state &s, const std::string &bitString) {
            return s.amplitude(bitStringToIntVec(bitString));
          },
          R"#(Return the amplitude of a state in computational basis.
          
.. code-block:: python

  # Example:
  # Create a simulation state.
  state = cudaq.get_state(kernel)
  # Return the amplitude of |0101>, assuming this is a 4-qubit state.
  amplitude = state.amplitude('0101'))#")
      .def(
          "amplitudes",
          [](state &s, const std::vector<std::vector<int>> &basisStates) {
            return s.amplitudes(basisStates);
          },
          R"#(Return the amplitude of a list of states in computational basis.
          
.. code-block:: python

  # Example:
  # Create a simulation state.
  state = cudaq.get_state(kernel)
  # Return the amplitude of |0101> and |1010>, assuming this is a 4-qubit state.
  amplitudes = state.amplitudes([[0,1,0,1], [1,0,1,0]]))#")
      .def(
          "amplitudes",
          [](state &s, const std::vector<std::string> &bitStrings) {
            std::vector<std::vector<int>> basisStates;
            basisStates.reserve(bitStrings.size());
            for (const auto &bitString : bitStrings)
              basisStates.emplace_back(bitStringToIntVec(bitString));
            return s.amplitudes(basisStates);
          },
          R"#(Return the amplitudes of a list of states in computational basis.
          
.. code-block:: python

  # Example:
  # Create a simulation state.
  state = cudaq.get_state(kernel)
  # Return the amplitudes of |0101> and |1010>, assuming this is a 4-qubit state.
  amplitudes = state.amplitudes(['0101', '1010']))#")
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
          [&holder](state &self, py::buffer &other) {
            if (self.get_num_tensors() != 1)
              throw std::runtime_error("overlap NumPy interop only supported "
                                       "for vector and matrix state data.");
            auto otherState = createStateFromPyBuffer(other, holder);
            return self.overlap(otherState);
          },
          "Compute the overlap between the provided :class:`State`'s.")
      .def(
          "overlap",
          [](state &self, py::object other) {
            // Note: This overload is no longer needed from cupy 13.5+ onward.
            // We can remove it in future releases. Make sure this is a CuPy
            // array
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
                  "state.overlap. input must be of complex float type, please "
                  "add to your cupy array creation `dtype=cupy.complex64` if "
                  "simulation is FP32 and `dtype=cupy.complex128` if "
                  "simulation if FP64.");
            auto precision = self.get_precision();
            if (typeStr == "complex64") {
              if (precision == SimulationState::precision::fp64)
                throw std::runtime_error(
                    "underlying simulation state is FP64, but "
                    "input cupy array is FP32.");
            } else if (typeStr == "complex128") {
              if (precision == SimulationState::precision::fp32)
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
            if (precision == SimulationState::precision::fp32)
              return self.overlap(state::from_data(
                  std::make_pair(reinterpret_cast<std::complex<float> *>(ptr),
                                 numOtherElements)));

            return self.overlap(state::from_data(
                std::make_pair(reinterpret_cast<std::complex<double> *>(ptr),
                               numOtherElements)));
          },
          "Compute overlap with general CuPy device array.");

  py::class_<state_view>(mod, "StateMemoryView", py::buffer_protocol())
      .def(py::init<state>())
      .def_buffer([](const state_view &self) {
        if (self.get_num_tensors() != 1)
          throw std::runtime_error("Numpy interop is only supported for vector "
                                   "and matrix state data.");

        // This method is used by Pybind to enable interoperability with NumPy
        // array data. We therefore must be careful since the state data may
        // actually be on GPU device.

        // Get the data pointer.
        // Data may be on GPU device, if so we must make a copy to host.
        // If users do not want this copy, they will have to operate apart
        // from Numpy
        void *dataPtr = nullptr;
        auto stateVector = self.get_tensor();
        auto precision = self.get_precision();
        if (self.is_on_gpu()) {
          // This is device data, transfer to host, which gives us
          // ownership of a new data pointer on host. Store it globally
          // here so we ensure that it gets cleaned up.
          auto numElements = stateVector.get_num_elements();
          if (precision == SimulationState::precision::fp32) {
            auto *hostData = new std::complex<float>[numElements];
            self.to_host(hostData, numElements);
            dataPtr = reinterpret_cast<void *>(hostData);
          } else {
            auto *hostData = new std::complex<double>[numElements];
            self.to_host(hostData, numElements);
            dataPtr = reinterpret_cast<void *>(hostData);
          }
          hostDataFromDevice.emplace_back(dataPtr, [](void *data) {
            CUDAQ_INFO("freeing data that was copied from GPU device for "
                       "compatibility with NumPy");
            free(data);
          });
        } else {
          dataPtr = self.get_tensor().data;
        }

        // We need to know the precision of the simulation data to get the
        // data type size and the format descriptor
        auto [dataTypeSize, desc] =
            precision == SimulationState::precision::fp32
                ? std::make_tuple(
                      sizeof(std::complex<float>),
                      py::format_descriptor<std::complex<float>>::format())
                : std::make_tuple(
                      sizeof(std::complex<double>),
                      py::format_descriptor<std::complex<double>>::format());

        // Get the shape of the data. Return buffer info in a correctly
        // shaped manner.
        auto shape = self.get_tensor().extents;
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
      .def("__getitem__",
           [](state_view &s, int idx) {
             // Support Pythonic negative index
             if (idx < 0)
               idx += (1 << s.get_num_qubits());
             return s[idx];
           })
      .def("__getitem__",
           [](state_view &s, std::vector<int> idx) {
             if (idx.size() != 2)
               throw std::runtime_error("Density matrix needs 2 indices; " +
                                        std::to_string(idx.size()) +
                                        " provided.");
             for (auto &val : idx)
               // Support Pythonic negative index
               if (val < 0)
                 val += (1 << s.get_num_qubits());
             return s(idx[0], idx[1]);
           })
      .def("dump",
           [](state_view &self) {
             std::stringstream ss;
             self.dump(ss);
             py::print(ss.str());
           })
      .def("__str__",
           [](state_view &self) {
             std::stringstream ss;
             self.dump(ss);
             return ss.str();
           })
      .def("__len__", [](state_view &self) {
        if (self.get_num_tensors() > 1 || self.get_tensor().extents.size() != 1)
          throw std::runtime_error(
              "len(state) only supported for state-vector like data.");

        return self.get_tensor().extents[0];
      });

  mod.def(
      "get_state_impl",
      [&](const std::string &shortName, MlirModule module, MlirType retTy,
          py::args args) {
        // Check for unsupported cases.
        if (holder.getTarget().name == "remote-mqpu" ||
            holder.getTarget().name == "orca-photonics" ||
            is_remote_platform() || is_emulated_platform())
          throw std::runtime_error(
              "get_state is not supported in this context.");
        return get_state_impl(shortName, module, retTy, args);
      },
      "See the python documenation for get_state.");

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
          py::call_guard<py::gil_scoped_release>(),
          "Return the :class:`State` from the asynchronous `get_state` "
          "accessor execution.\n");

  mod.def(
      "get_state_async_impl",
      [&](const std::string &shortName, MlirModule module, MlirType retTy,
          std::size_t qpu_id, py::args args) {
        // Check for unsupported cases.
        if (holder.getTarget().name == "remote-mqpu" ||
            holder.getTarget().name == "nvqc" ||
            holder.getTarget().name == "orca-photonics" ||
            is_remote_platform() || is_emulated_platform())
          throw std::runtime_error(
              "get_state_async is not supported in this context.");

        return get_state_async_impl(shortName, module, retTy, qpu_id, args);
      },
      "See the python documentation for get_state_async.");

  mod.def("get_state_library_mode", &pyGetStateLibraryMode,
          "Run `cudaq.get_state` in library mode on the provided kernel "
          "and args.");
}
