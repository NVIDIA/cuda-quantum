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
#include "common/KernelArgs.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/runtime/logger/logger.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/NanobindAdaptors.h"
#include "utils/OpaqueArguments.h"
#include <cstdint>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <numeric>

using namespace cudaq;

// FIXME: This is using a thread unsafe global?
/// If we have any implicit device-to-host data transfers we will store that
/// data here and ensure it is deleted properly.
static std::vector<std::unique_ptr<void, std::function<void(void *)>>>
    hostDataFromDevice;

namespace {
// CuPy interop helpers.
struct BufferInfo {
  void *ptr = nullptr;
  std::size_t itemsize = 0;
  std::string format;
  std::vector<std::size_t> shape;
  std::vector<ssize_t> strides;
  bool readonly = false;
  std::size_t size = 0;
};
} // namespace

static nanobind::dict getCupyArrayInterface(nanobind::handle cupyArray) {
  if (!nanobind::hasattr(cupyArray, "__cuda_array_interface__"))
    throw std::runtime_error("Buffer is not a CuPy array");

  return nanobind::cast<nanobind::dict>(
      nanobind::borrow<nanobind::object>(cupyArray).attr(
          "__cuda_array_interface__"));
}

static std::vector<ssize_t>
getCContiguousStrides(const std::vector<std::size_t> &shape,
                      std::size_t itemsize) {
  std::vector<ssize_t> strides(shape.size(), itemsize);
  if (shape.empty())
    return strides;

  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * static_cast<ssize_t>(shape[i + 1]);

  return strides;
}

static std::vector<ssize_t>
getCupyArrayStrides(const nanobind::dict &cupyArrayInfo,
                    const std::vector<std::size_t> &shape,
                    std::size_t itemsize) {
  auto stridesObj = cupyArrayInfo["strides"];
  if (stridesObj.is_none())
    return getCContiguousStrides(shape, itemsize);

  auto stridesTuple = nanobind::cast<nanobind::tuple>(stridesObj);
  std::vector<ssize_t> strides;
  strides.reserve(stridesTuple.size());
  for (auto stride : stridesTuple)
    strides.push_back(nanobind::cast<ssize_t>(stride));

  return strides;
}

static std::pair<std::size_t, std::string>
getCupyComplexTypeInfo(const std::string &typeStr) {
  if (typeStr == "<c8")
    return {sizeof(std::complex<float>), "Zf"};
  if (typeStr == "<c16")
    return {sizeof(std::complex<double>), "Zd"};

  throw std::runtime_error("Unsupported typestr in CuPy array: " + typeStr +
                           ". Supported types are: <c16 and <c8.");
}

static BufferInfo getCupyBufferInfo(nanobind::object cupyArray) {
  auto cupyArrayInfo = getCupyArrayInterface(cupyArray);
  auto dataInfo = nanobind::cast<nanobind::tuple>(cupyArrayInfo["data"]);
  auto shapeTuple = nanobind::cast<nanobind::tuple>(cupyArrayInfo["shape"]);
  std::vector<std::size_t> shape;
  shape.reserve(shapeTuple.size());
  for (auto dim : shapeTuple)
    shape.push_back(nanobind::cast<std::size_t>(dim));

  const std::string typeStr =
      nanobind::cast<std::string>(cupyArrayInfo["typestr"]);
  auto [dataTypeSize, formatDescriptor] = getCupyComplexTypeInfo(typeStr);
  auto strides = getCupyArrayStrides(cupyArrayInfo, shape, dataTypeSize);
  auto numElements = std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                                     std::multiplies<std::size_t>());

  BufferInfo info;
  info.ptr =
      reinterpret_cast<void *>(nanobind::cast<std::uintptr_t>(dataInfo[0]));
  info.itemsize = dataTypeSize;
  info.format = formatDescriptor;
  info.shape = std::move(shape);
  info.strides = std::move(strides);
  info.readonly = nanobind::cast<bool>(dataInfo[1]);
  info.size = numElements;
  return info;
}

static bool isCContiguous(const std::vector<std::size_t> &shape,
                          const std::vector<ssize_t> &strides,
                          std::size_t itemsize) {
  // Treat inconsistent metadata as non-contiguous so callers fall back to
  // canonicalization instead of taking an unsafe fast path.
  if (shape.size() != strides.size())
    return false;

  if (shape.empty())
    return true;

  ssize_t expectedStride = static_cast<ssize_t>(itemsize);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] > 1 && strides[i] != expectedStride)
      return false;
    expectedStride *= static_cast<ssize_t>(shape[i]);
  }

  return true;
}

static bool shouldCanonicalizeCupyArray(const BufferInfo &info,
                                        const std::string &targetName) {
  if (info.shape.empty())
    return false;

  // Only 2D arrays for the dynamics target or non-contiguous 1D arrays
  // need canonicalization.
  bool needsCanon = (info.shape.size() == 1) ||
                    (info.shape.size() == 2 && targetName == "dynamics");
  return needsCanon && !isCContiguous(info.shape, info.strides, info.itemsize);
}

static nanobind::object
canonicalizeCupyArrayToNumpy(nanobind::handle cupyArray) {
  return nanobind::module_::import_("cupy").attr("asnumpy")(
      nanobind::borrow<nanobind::object>(cupyArray));
}

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
                            nanobind::args args) {
  auto closure = [=]() {
    return marshal_and_launch_module(shortName, mod, args);
  };
  return details::extractState(std::move(closure));
}

static std::future<state> get_state_async_impl(const std::string &shortName,
                                               MlirModule module,
                                               std::size_t qpu_id,
                                               nanobind::args args) {
  // Launch the asynchronous execution.
  auto mod = unwrap(module);
  std::string kernelName = shortName;
  auto &platform = get_platform();
  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, args, fnOp);

  nanobind::gil_scoped_release release;
  auto clonedMod = std::shared_ptr<mlir::ModuleOp>(
      new mlir::ModuleOp(mod.clone()), [](mlir::ModuleOp *p) {
        p->erase();
        delete p;
      });
  return details::runGetStateAsync(
      detail::make_copyable_function(
          [opaques = std::move(opaques), kernelName, clonedMod]() mutable {
            [[maybe_unused]] auto result =
                clean_launch_module(kernelName, *clonedMod, opaques);
          }),
      platform, qpu_id);
}

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
      // Note: in Python, the platform QPU (`PyRemoteSimulatorQPU`) expects an
      // ModuleOp pointer as the first element in the args array in StreamLined
      // mode.
      auto args = argsData->getArgs();
      args.insert(args.begin(),
                  const_cast<void *>(static_cast<const void *>(&kernelMod)));
      cudaq::SourceModule src{kernelName};
      platform.with_execution_context(context, [&]() {
        [[maybe_unused]] auto r = platform.unifiedLaunchModule(src, {args});
      });
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
    auto args = argsData->getArgs();
    args.insert(args.begin(),
                const_cast<void *>(static_cast<const void *>(&kernelMod)));

    cudaq::SourceModule src{kernelName};
    platform.with_execution_context(context, [&]() {
      [[maybe_unused]] auto r = platform.unifiedLaunchModule(src, {args});
    });
    assert(context.overlapResult.has_value());
    return context.overlapResult.value();
  }

  virtual ~PyRemoteSimulationState() override { delete argsData; }
};

/// @brief Run `cudaq::get_state` for remote execution targets on the provided
/// kernel and args
state pyGetStateRemote(nanobind::object kernel, nanobind::args args) {
  if (nanobind::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = nanobind::cast<std::string>(kernel.attr("uniqName"));
  auto kernelMod = nanobind::cast<MlirModule>(kernel.attr("qkeModule"));
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
  PyQPUState(const std::string &in_kernelName, const std::string &in_kernelCode,
             OpaqueArguments *argsDataToOwn)
      : argsData(argsDataToOwn) {
    this->kernelName = in_kernelName;
    this->kernelQuake = in_kernelCode;
    this->args = argsData->getArgs();
  }

  virtual ~PyQPUState() override { delete argsData; }
};

/// @brief Run `cudaq::get_state` for qpu targets on the provided
/// kernel and args
state pyGetStateQPU(const std::string &kernelName, MlirModule kernelMod,
                    nanobind::args args) {
  auto moduleOp = unwrap(kernelMod);
  std::string mlirCode;
  llvm::raw_string_ostream outStr(mlirCode);
  mlir::OpPrintingFlags opf;
  opf.enableDebugInfo(/*enable=*/true, /*pretty=*/false);
  moduleOp.print(outStr, opf);
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);
  return state(new PyQPUState(kernelName, mlirCode, argData));
}

state pyGetStateLibraryMode(nanobind::object kernel, nanobind::args args) {
  return details::extractState([&]() mutable {
    if (0 == args.size())
      kernel();
    else {
      std::vector<nanobind::object> argsData;
      for (size_t i = 0; i < args.size(); i++) {
        nanobind::object arg = args[i];
        argsData.emplace_back(std::forward<nanobind::object>(arg));
      }
      kernel(std::move(argsData));
    }
  });
}

static BufferInfo getNumpyBufferInfo(nanobind::object numpy_array) {
  auto dtype = numpy_array.attr("dtype");
  std::string dtypeStr = nanobind::cast<std::string>(dtype.attr("name"));

  BufferInfo info;
  if (dtypeStr == "complex64") {
    info.itemsize = sizeof(std::complex<float>);
    info.format = "Zf";
  } else if (dtypeStr == "complex128") {
    info.itemsize = sizeof(std::complex<double>);
    info.format = "Zd";
  } else {
    info.format = dtypeStr;
    info.itemsize = nanobind::cast<std::size_t>(dtype.attr("itemsize"));
  }
  auto shapeTuple = nanobind::cast<nanobind::tuple>(numpy_array.attr("shape"));
  info.size = 1;
  for (std::size_t i = 0; i < shapeTuple.size(); i++) {
    auto ext = nanobind::cast<std::size_t>(shapeTuple[i]);
    info.shape.push_back(ext);
    info.size *= ext;
  }
  auto stridesTuple =
      nanobind::cast<nanobind::tuple>(numpy_array.attr("strides"));
  for (std::size_t i = 0; i < stridesTuple.size(); i++) {
    info.strides.push_back(nanobind::cast<ssize_t>(stridesTuple[i]));
  }
  // Get the raw data pointer via numpy's ctypes interface
  info.ptr = reinterpret_cast<void *>(
      nanobind::cast<intptr_t>(numpy_array.attr("ctypes").attr("data")));
  info.readonly = false;
  return info;
}

static cudaq::state createStateFromPyBuffer(nanobind::object data,
                                            LinkedLibraryHolder &holder) {
  const bool isHostData = !nanobind::hasattr(data, "__cuda_array_interface__");
  // Check that the target is GPU-based, i.e., can handle device
  // pointer.
  if (!holder.getTarget().config.GpuRequired && !isHostData)
    throw std::runtime_error(
        fmt::format("Current target '{}' does not support CuPy arrays.",
                    holder.getTarget().name));

  auto info = isHostData ? getNumpyBufferInfo(data) : getCupyBufferInfo(data);
  if (info.shape.size() > 2)
    throw std::runtime_error(
        "state.from_data only supports 1D or 2D array data.");
  if (info.format != "Zf" && info.format != "Zd")
    throw std::runtime_error(
        "A numpy array with only floating point elements passed to "
        "`state.from_data`. Input must be of complex float type. Please add to "
        "your array creation `dtype=numpy.complex64` if simulation is FP32 and "
        "`dtype=numpy.complex128` if simulation is FP64, or "
        "`dtype=cudaq.complex()` for precision-agnostic code.");

  if (!isHostData && shouldCanonicalizeCupyArray(info, holder.getTarget().name))
    return createStateFromPyBuffer(canonicalizeCupyArrayToNumpy(data), holder);

  if (!isHostData) {
    if (holder.getTarget().name == "dynamics") {
      if (info.shape.size() == 2 && info.shape[0] != info.shape[1])
        throw std::runtime_error(
            "state.from_data 2D array (density matrix) input must be "
            "square matrix data.");
      TensorStateData tensorData{
          std::pair<const void *, std::vector<std::size_t>>{info.ptr,
                                                            info.shape}};
      return state::from_data(tensorData);
    }

    if (info.format == "Zf")
      return state::from_data(std::make_pair(
          reinterpret_cast<std::complex<float> *>(info.ptr), info.size));

    return state::from_data(std::make_pair(
        reinterpret_cast<std::complex<double> *>(info.ptr), info.size));
  }

  if (info.shape.size() == 1) {
    if (info.format == "Zf")
      return state::from_data(std::make_pair(
          reinterpret_cast<std::complex<float> *>(info.ptr), info.size));

    return state::from_data(std::make_pair(
        reinterpret_cast<std::complex<double> *>(info.ptr), info.size));
  }

  const std::size_t rows = info.shape[0];
  const std::size_t cols = info.shape[1];
  if (rows != cols)
    throw std::runtime_error(
        "state.from_data 2D array (density matrix) input must be "
        "square matrix data.");
  const bool isDoublePrecision = (info.format == "Zd");
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

/// @brief Bind the get_state cudaq function
void cudaq::bindPyState(nanobind::module_ &mod, LinkedLibraryHolder &holder) {
  nanobind::enum_<InitialState>(mod, "InitialStateType",
                                "Enumeration describing the initial state "
                                "type to be created in the backend")
      .value("ZERO", InitialState::ZERO)
      .value("UNIFORM", InitialState::UNIFORM)
      .export_values();

  nanobind::class_<SimulationState::Tensor>(
      mod, "Tensor",
      "The `Tensor` describes a pointer to simulation data as well as the rank "
      "and extents for that tensorial data it represents.")
      .def("data",
           [](SimulationState::Tensor &tensor) {
             return reinterpret_cast<intptr_t>(tensor.data);
           })
      .def_ro("extents", &SimulationState::Tensor::extents)
      .def("get_rank", &SimulationState::Tensor::get_rank)
      .def("get_element_size", &SimulationState::Tensor::element_size)
      .def("get_num_elements", &SimulationState::Tensor::get_num_elements);

  nanobind::class_<state>(
      mod, "State",
      "A data-type representing the quantum state of the internal simulator. "
      "This type is not user-constructible and instances can only be retrieved "
      "via the `cudaq.get_state(...)` function or the static "
      "`cudaq.State.from_data()` method.\n")
      .def(
          "__array__",
          [](const state &self, nanobind::object dtype_obj,
             nanobind::object copy_obj) {
            if (self.get_num_tensors() != 1)
              throw std::runtime_error(
                  "Numpy interop is only supported for vector "
                  "and matrix state data.");

            // This method enables interoperability with NumPy array data.
            // We must be careful since the state data may actually be on GPU
            // device.

            nanobind::module_ np = nanobind::module_::import_("numpy");
            auto stateVector = self.get_tensor();
            auto precision = self.get_precision();
            auto shape = self.get_tensor().extents;

            // Determine numpy dtype
            nanobind::object np_dtype =
                precision == SimulationState::precision::fp32
                    ? np.attr("complex64")
                    : np.attr("complex128");

            if (self.is_on_gpu()) {
              // This is device data, transfer to host
              auto numElements = stateVector.get_num_elements();
              nanobind::object arr;
              if (precision == SimulationState::precision::fp32) {
                auto *hostData = new std::complex<float>[numElements];
                self.to_host(hostData, numElements);
                // Create numpy array and copy data
                if (shape.size() != 1) {
                  nanobind::tuple np_shape =
                      nanobind::make_tuple(shape[0], shape[1]);
                  arr = np.attr("empty")(np_shape, np_dtype);
                } else {
                  nanobind::tuple np_shape = nanobind::make_tuple(shape[0]);
                  arr = np.attr("empty")(np_shape, np_dtype);
                }
                auto *destPtr = reinterpret_cast<std::complex<float> *>(
                    nanobind::cast<intptr_t>(arr.attr("ctypes").attr("data")));
                std::memcpy(destPtr, hostData,
                            numElements * sizeof(std::complex<float>));
                delete[] hostData;
              } else {
                auto *hostData = new std::complex<double>[numElements];
                self.to_host(hostData, numElements);
                if (shape.size() != 1) {
                  nanobind::tuple np_shape =
                      nanobind::make_tuple(shape[0], shape[1]);
                  arr = np.attr("empty")(np_shape, np_dtype);
                } else {
                  nanobind::tuple np_shape = nanobind::make_tuple(shape[0]);
                  arr = np.attr("empty")(np_shape, np_dtype);
                }
                auto *destPtr = reinterpret_cast<std::complex<double> *>(
                    nanobind::cast<intptr_t>(arr.attr("ctypes").attr("data")));
                std::memcpy(destPtr, hostData,
                            numElements * sizeof(std::complex<double>));
                delete[] hostData;
              }
              return arr;
            }

            // Host data path - wrap existing memory
            void *dataPtr = self.get_tensor().data;
            auto numElements = stateVector.get_num_elements();
            if (shape.size() != 1) {
              nanobind::tuple np_shape =
                  nanobind::make_tuple(shape[0], shape[1]);
              // Use np.frombuffer-like approach: create array from pointer
              nanobind::object arr = np.attr("empty")(np_shape, np_dtype);
              auto *destPtr = reinterpret_cast<void *>(
                  nanobind::cast<intptr_t>(arr.attr("ctypes").attr("data")));
              std::size_t dataTypeSize =
                  precision == SimulationState::precision::fp32
                      ? sizeof(std::complex<float>)
                      : sizeof(std::complex<double>);
              std::memcpy(destPtr, dataPtr, numElements * dataTypeSize);
              return arr;
            }
            nanobind::tuple np_shape = nanobind::make_tuple(shape[0]);
            nanobind::object arr = np.attr("empty")(np_shape, np_dtype);
            auto *destPtr = reinterpret_cast<void *>(
                nanobind::cast<intptr_t>(arr.attr("ctypes").attr("data")));
            std::size_t dataTypeSize =
                precision == SimulationState::precision::fp32
                    ? sizeof(std::complex<float>)
                    : sizeof(std::complex<double>);
            std::memcpy(destPtr, dataPtr, numElements * dataTypeSize);
            return arr;
          },
          nanobind::arg("dtype") = nanobind::none(),
          nanobind::arg("copy") = nanobind::none())
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
          [&](nanobind::object data) {
            // Reject Python sequences (list/tuple) overload — they should be
            // dispatched to the vector overload below. In pybind11, py::buffer
            // excluded lists; nanobind::object accepts anything, so we must
            // guard explicitly.
            if (nanobind::isinstance<nanobind::list>(data) ||
                nanobind::isinstance<nanobind::tuple>(data))
              throw nanobind::next_overload();
            return createStateFromPyBuffer(data, holder);
          },
          "Return a state from data.")
      .def_static(
          "from_data",
          [&holder](const std::vector<nanobind::object> &tensors) {
            // Reject SimulationState::Tensor objects overload — they're handled
            // by the next overload and don't have numpy/cupy buffer attributes.
            if (!tensors.empty() &&
                nanobind::isinstance<SimulationState::Tensor>(tensors[0]))
              throw nanobind::next_overload();
            const bool isHostData =
                tensors.empty() ||
                !nanobind::hasattr(tensors[0], "__cuda_array_interface__");
            // Check that the target is GPU-based, i.e., can handle device
            // pointer.
            if (!holder.getTarget().config.GpuRequired && !isHostData)
              throw std::runtime_error(fmt::format(
                  "Current target '{}' does not support CuPy arrays.",
                  holder.getTarget().name));
            TensorStateData tensorData;
            for (auto &tensor : tensors) {
              auto info = isHostData ? getNumpyBufferInfo(tensor)
                                     : getCupyBufferInfo(tensor);
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
          [](const nanobind::list &tensors) {
            // Note: we must use Python type (nanobind::list) for proper
            // overload resolution. The overload for nanobind::object, intended
            // for cupy arrays (implementing Python array interface), may be
            // overshadowed by any std::vector overloads.
            TensorStateData tensorData;
            for (auto tensor : tensors) {
              // Make sure this is a CuPy array
              if (!nanobind::hasattr(tensor, "data"))
                throw std::runtime_error(
                    "invalid from_data operation on nanobind::object - "
                    "only cupy array supported.");
              auto data = tensor.attr("data");
              if (!nanobind::hasattr(data, "ptr"))
                throw std::runtime_error(
                    "invalid from_data operation on nanobind::object tensors - "
                    "only cupy array supported.");

              // We know this is a cupy device pointer. Start by ensuring it is
              // of proper complex type
              auto typeStr = nanobind::cast<std::string>(
                  tensor.attr("dtype").attr("name"));
              if (typeStr != "complex128")
                throw std::runtime_error(
                    "invalid from_data operation on nanobind::object tensors - "
                    "only cupy complex128 tensors supported.");
              auto shape =
                  nanobind::cast<nanobind::tuple>(tensor.attr("shape"));
              std::vector<std::size_t> extents;
              for (auto el : shape)
                extents.emplace_back(nanobind::cast<std::size_t>(el));
              long ptr = nanobind::cast<long>(data.attr("ptr"));
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
          [&holder](nanobind::object opaqueData) {
            // Note: This overload is no longer needed from cupy 13.5+ onward.
            // We can remove it in future releases.
            // Make sure this is a CuPy array
            if (!nanobind::hasattr(opaqueData, "data"))
              throw std::runtime_error(
                  "invalid from_data operation on nanobind::object - "
                  "only cupy array supported.");
            auto data = opaqueData.attr("data");
            if (!nanobind::hasattr(data, "ptr"))
              throw std::runtime_error(
                  "invalid from_data operation on nanobind::object - "
                  "only cupy array supported.");

            // We know this is a cupy device pointer. Start by ensuring it is of
            // complex type
            auto typeStr = nanobind::cast<std::string>(
                opaqueData.attr("dtype").attr("name"));
            if (typeStr.find("float") != std::string::npos)
              throw std::runtime_error(
                  "CuPy array with only floating point elements passed to "
                  "state.from_data. input must be of complex float type, "
                  "please add to your cupy array creation "
                  "`dtype=cupy.complex64` if simulation is FP32 and "
                  "`dtype=cupy.complex128` if simulation if FP64.");
            return createStateFromPyBuffer(opaqueData, holder);
          },
          "Return a state from CuPy device array.")
      .def("is_on_gpu", &state::is_on_gpu,
           "Return True if this state is on the GPU.")
      .def(
          "getTensor",
          [](state &self, std::size_t idx) { return self.get_tensor(idx); },
          nanobind::arg("idx") = 0,
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
            nanobind::print(ss.str().c_str());
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
          [&holder](state &self, nanobind::object &other) {
            if (self.get_num_tensors() != 1)
              throw std::runtime_error("overlap NumPy interop only supported "
                                       "for vector and matrix state data.");
            auto otherState = createStateFromPyBuffer(other, holder);
            return self.overlap(otherState);
          },
          "Compute the overlap between the provided :class:`State`'s.")
      .def(
          "overlap",
          [](state &self, nanobind::object other) {
            // Note: This overload is no longer needed from cupy 13.5+ onward.
            // We can remove it in future releases. Make sure this is a CuPy
            // array
            if (!nanobind::hasattr(other, "data"))
              throw std::runtime_error(
                  "invalid overlap operation on nanobind::object - "
                  "only cupy array supported.");
            auto data = other.attr("data");
            if (!nanobind::hasattr(data, "ptr"))
              throw std::runtime_error(
                  "invalid overlap operation on nanobind::object - "
                  "only cupy array supported.");

            // We know this is a cupy device pointer.

            // Start by ensuring it is of complex type
            auto typeStr =
                nanobind::cast<std::string>(other.attr("dtype").attr("name"));
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
              auto shape = nanobind::cast<nanobind::tuple>(other.attr("shape"));
              std::size_t numElements = 1;
              for (auto el : shape)
                numElements *= nanobind::cast<std::size_t>(el);
              return numElements;
            }();

            // Cast the device ptr and perform the overlap
            long ptr = nanobind::cast<long>(data.attr("ptr"));
            if (precision == SimulationState::precision::fp32)
              return self.overlap(state::from_data(
                  std::make_pair(reinterpret_cast<std::complex<float> *>(ptr),
                                 numOtherElements)));

            return self.overlap(state::from_data(
                std::make_pair(reinterpret_cast<std::complex<double> *>(ptr),
                               numOtherElements)));
          },
          "Compute overlap with general CuPy device array.");

  mod.def(
      "get_state_impl",
      [&](const std::string &shortName, MlirModule module,
          nanobind::args args) {
        // Check for unsupported cases.
        if (holder.getTarget().name == "remote-mqpu" ||
            holder.getTarget().name == "orca-photonics")
          throw std::runtime_error(
              "get_state is not supported in this context.");

        if (is_remote_platform() || is_emulated_platform())
          return pyGetStateQPU(shortName, module, args);
        return get_state_impl(shortName, module, args);
      },
      "See the python documentation for get_state.");

  nanobind::class_<async_state_result>(
      mod, "AsyncStateResult",
      R"#(A data-type containing the results of a call to :func:`get_state_async`.
The `AsyncStateResult` models a future-like type, whose
:class:`State` may be returned via an invocation of the `get` method. This
kicks off a wait on the current thread until the results are available.
See `future <https://en.cppreference.com/w/cpp/thread/future>`_
for more information on this programming pattern.)#")
      .def(
          "get", [](async_state_result &self) { return self.get(); },
          nanobind::call_guard<nanobind::gil_scoped_release>(),
          "Return the :class:`State` from the asynchronous `get_state` "
          "accessor execution.\n");

  mod.def(
      "get_state_async_impl",
      [&](const std::string &shortName, MlirModule module, std::size_t qpu_id,
          nanobind::args args) {
        // Check for unsupported cases.
        if (holder.getTarget().name == "remote-mqpu" ||
            holder.getTarget().name == "nvqc" ||
            holder.getTarget().name == "orca-photonics" ||
            is_remote_platform() || is_emulated_platform())
          throw std::runtime_error(
              "get_state_async is not supported in this context.");

        return get_state_async_impl(shortName, module, qpu_id, args);
      },
      "See the python documentation for get_state_async.");

  mod.def("get_state_library_mode", &pyGetStateLibraryMode,
          "Run `cudaq.get_state` in library mode on the provided kernel "
          "and args.");
}
