/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_run.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/algorithms/run.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <future>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <tuple>
#include <vector>

using namespace cudaq;

static std::vector<py::object> readRunResults(mlir::ModuleOp module,
                                              mlir::func::FuncOp kernelFuncOp,
                                              mlir::Type ty,
                                              details::RunResultSpan &results,
                                              std::size_t count) {
  std::vector<py::object> ret;
  std::size_t byteSize = results.lengthInBytes / count;
  for (std::size_t i = 0; i < results.lengthInBytes; i += byteSize) {
    py::object obj = convertResult(module, kernelFuncOp, ty, results.data + i);
    ret.push_back(obj);
  }
  return ret;
}

static std::tuple<std::string, MlirModule, OpaqueArguments *,
                  mlir::func::FuncOp, std::string, mlir::func::FuncOp,
                  std::vector<std::string>>
getKernelLaunchParameters(py::object &kernel, py::args args) {
  if (!py::hasattr(kernel, "arguments"))
    throw std::runtime_error(
        "unrecognized kernel - did you forget the @kernel attribute?");
  if (py::len(kernel.attr("arguments")) != args.size())
    throw std::runtime_error("Invalid number of arguments passed to run:" +
                             std::to_string(args.size()) + " expected " +
                             std::to_string(py::len(kernel.attr("arguments"))));

  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  // Process any callable args
  const auto callableNames = getCallableNames(kernel, args);

  auto origKernName = kernel.attr("name").cast<std::string>();
  auto kernelName = origKernName + ".run";
  if (!py::hasattr(kernel, "module") || kernel.attr("module").is_none())
    throw std::runtime_error(
        "Unsupported target / Invalid kernel for `run`: missing module");
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  args = simplifiedValidateInputArguments(args);
  auto origKern = getKernelFuncOp(kernelMod, origKernName);

  // Lookup the runnable kernel.
  auto mod = unwrap(kernelMod);
  auto runKern = mod.lookupSymbol<mlir::func::FuncOp>(
      cudaq::runtime::cudaqGenPrefixName + kernelName);
  if (!runKern) {
    if (origKern.getResultTypes().empty())
      throw std::runtime_error(
          "`cudaq.run` only supports kernels that return a value.");
    mlir::PassManager pm(mod.getContext());
    pm.addPass(cudaq::opt::createGenerateKernelExecution(
        {.genRunStack = true, .deferToJIT = true}));
    if (mlir::failed(pm.run(mod)))
      throw std::runtime_error(
          "failed to autogenerate the runnable variant of the kernel.");
  }
  auto *argData =
      toOpaqueArgs(args, kernelMod, kernelName, getCallableArgHandler());
  auto funcOp = getKernelFuncOp(kernelMod, kernelName);
  return {kernelName,   kernelMod, argData,      funcOp,
          origKernName, origKern,  callableNames};
}

static details::RunResultSpan
pyRunTheKernel(const std::string &name, const std::string &origName,
               MlirModule module, mlir::func::FuncOp funcOp,
               mlir::func::FuncOp origKernel, OpaqueArguments &runtimeArgs,
               quantum_platform &platform, std::size_t shots_count,
               const std::vector<std::string> &callableNames,
               std::size_t qpu_id = 0) {
  auto returnTypes = origKernel.getResultTypes();
  if (returnTypes.empty() || returnTypes.size() > 1)
    throw std::runtime_error(
        "`cudaq.run` only supports kernels that return a value.");

  auto returnTy = returnTypes[0];
  // Disallow returning nested vectors/vectors of structs from entry-point
  // kernels.
  if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(returnTy)) {
    auto elemTy = vecTy.getElementType();
    if (elemTy.isa<cudaq::cc::StdvecType>())
      throw std::runtime_error(
          "`cudaq.run` does not yet support returning nested `list` from "
          "entry-point kernels.");
    if (elemTy.isa<cudaq::cc::StructType>())
      throw std::runtime_error("`cudaq.run` does not yet support returning "
                               "`list` of `dataclass`/`tuple` from "
                               "entry-point kernels.");
  }

  auto mod = unwrap(module);

  auto [rawArgs, size, returnOffset, thunk] = pyAltLaunchKernelBase(
      name, module, returnTy, runtimeArgs, callableNames, 0, false);

  auto results = details::runTheKernel(
      [&]() mutable {
        pyLaunchKernel(name, thunk, mod, runtimeArgs, rawArgs, size,
                       returnOffset, callableNames);
      },
      platform, name, origName, shots_count, qpu_id);

  std::free(rawArgs);
  return results;
}

static std::vector<py::object> pyReadResults(details::RunResultSpan results,
                                             MlirModule module,
                                             mlir::func::FuncOp funcOp,
                                             mlir::func::FuncOp origKern,
                                             std::size_t shots_count) {
  auto mod = unwrap(module);
  auto returnTy = origKern.getResultTypes()[0];
  return readRunResults(mod, funcOp, returnTy, results, shots_count);
}

namespace cudaq {
/// @brief Run `cudaq::run` on the provided kernel.
std::vector<py::object> pyRun(py::object &kernel, py::args args,
                              std::size_t shots_count,
                              std::optional<noise_model> noise_model) {
  if (shots_count == 0)
    return {};

  auto [name, module, argData, func, origName, origKern, callableNames] =
      getKernelLaunchParameters(kernel, args);

  auto mod = unwrap(module);
  mod->setAttr(runtime::enableCudaqRun, mlir::UnitAttr::get(mod->getContext()));

  auto &platform = get_platform();
  if (noise_model.has_value()) {
    if (platform.is_remote())
      throw std::runtime_error(
          "Noise model is not supported on remote platforms.");
    // Launch the kernel in the appropriate context.
    platform.set_noise(&noise_model.value());
  }

  auto span = pyRunTheKernel(name, origName, module, func, origKern, *argData,
                             platform, shots_count, callableNames);
  delete argData;
  auto results = pyReadResults(span, module, func, origKern, shots_count);

  if (noise_model.has_value())
    platform.reset_noise();

  return results;
}
} // namespace cudaq

namespace {
// Internal struct representing buffer to be filled asynchronously.
// When the `ready` future is set, the content of the buffer is filled.
struct async_run_result {
  std::future<void> ready;
  std::vector<py::object> *results;
  std::string *error;
};
} // namespace

namespace cudaq {
/// @brief Run `cudaq::run_async` on the provided kernel.
async_run_result pyRunAsync(py::object &kernel, py::args args,
                            std::size_t shots_count,
                            std::optional<noise_model> noise_model,
                            std::size_t qpu_id) {
  kernel.inc_ref();
  auto &platform = get_platform();
  auto numQPUs = platform.num_qpus();
  if (qpu_id >= numQPUs)
    throw std::runtime_error("qpu_id (" + std::to_string(qpu_id) +
                             ") exceeds the number of available QPUs (" +
                             std::to_string(numQPUs) + ")");

  auto [name, module, argData, func, origName, origKern, callableNames] =
      getKernelLaunchParameters(kernel, args);

  auto mod = unwrap(module);
  mod->setAttr(runtime::enableCudaqRun, mlir::UnitAttr::get(mod->getContext()));

  if (noise_model.has_value() && platform.is_remote())
    throw std::runtime_error(
        "Noise model is not supported on remote platforms.");

  async_run_result result;
  result.results = new std::vector<py::object>();
  result.error = new std::string();

  if (shots_count == 0) {
    std::promise<void> promise;
    result.ready = promise.get_future();
    promise.set_value();
    return result;
  }

  std::promise<details::RunResultSpan> spanPromise;
  auto spanFuture = spanPromise.get_future();

  std::promise<std::string> errorPromise;
  auto errorFuture = errorPromise.get_future();

  // Run the kernel and compute results span.
  {
    // Release GIL to allow c++ threads, all code inside the scope is c++, so
    // there is no need to re-acquire the GIL inside the thread.
    py::gil_scoped_release gil_release{};
    QuantumTask wrapped = detail::make_copyable_function(
        [sp = std::move(spanPromise), ep = std::move(errorPromise), shots_count,
         qpu_id, argData, name, module, func, origKern, origName,
         noise_model = std::move(noise_model), callableNames]() mutable {
          auto &platform = get_platform();

          // Launch the kernel in the appropriate context.
          if (noise_model.has_value())
            platform.set_noise(&noise_model.value());

          try {
            auto span =
                pyRunTheKernel(name, origName, module, func, origKern, *argData,
                               platform, shots_count, callableNames, qpu_id);
            delete argData;
            sp.set_value(span);
            ep.set_value("");
          } catch (std::runtime_error &e) {
            auto message = std::string(e.what());
            sp.set_value({});
            ep.set_value(message);
          }
          platform.reset_noise();
        });
    platform.enqueueAsyncTask(qpu_id, wrapped);
  }

  // Convert results after the span is computed.
  {
    // Release GIL to allow c++ threads, re-acquire for conversion of the
    // results to python objects.
    py::gil_scoped_release gil_release{};
    auto resultFuture =
        std::async(std::launch::deferred,
                   [sf = std::move(spanFuture), ef = std::move(errorFuture),
                    errorPtr = result.error, resultsPtr = result.results,
                    module, func, origKern, shots_count]() mutable {
                     auto error = ef.get();
                     std::swap(*errorPtr, error);
                     if (error.empty()) {
                       auto span = sf.get();
                       py::gil_scoped_acquire gil{};
                       auto results = pyReadResults(span, module, func,
                                                    origKern, shots_count);
                       std::swap(*resultsPtr, results);
                     }
                   });
    result.ready = std::move(resultFuture);
  }

  return result;
}
} // namespace cudaq

/// @brief Bind the run cudaq function.
void cudaq::bindPyRun(py::module &mod) {
  mod.def("run", &pyRun, py::arg("kernel"), py::kw_only(),
          py::arg("shots_count") = 100, py::arg("noise_model") = py::none(),
          R"#(Run the provided `kernel` with the given kernel arguments over 
the specified number of circuit executions (`shots_count`).

Args:
  kernel: The kernel to execute `shots_count` times on the QPU.
  *arguments: The concrete values to evaluate the kernel function at.
  shots_count: The number of kernel executions on the QPU. Defaults to 100.
  noise_model: The optional noise model to add noise to the kernel execution.

Returns:
  A list of kernel return values from each execution. The length equals `shots_count`.
)#");
}

/// @brief Bind the run_async cudaq function.
void cudaq::bindPyRunAsync(py::module &mod) {
  py::class_<async_run_result>(mod, "AsyncRunResult", "")
      .def(
          "get",
          [](async_run_result &self) {
            self.ready.get();
            auto err = *self.error;
            if (!err.empty()) {
              delete self.error;
              throw std::runtime_error(err);
            }
            auto ret = *self.results;
            delete self.results;
            return ret;
          },
          "");
  mod.def("run_async_internal", &pyRunAsync, py::arg("kernel"), py::kw_only(),
          py::arg("shots_count") = 100, py::arg("noise_model") = py::none(),
          py::arg("qpu_id") = 0,
          R"#(Run the provided `kernel` with the given kernel arguments over 
the specified number of circuit executions (`shots_count`) asynchronously on the 
specified `qpu_id`.

Args:
  kernel: The kernel to execute `shots_count` times on the QPU.
  *arguments: The concrete values to evaluate the kernel function at.
  shots_count: The number of kernel executions on the QPU. Defaults to 100.
  noise_model: The optional noise model to add noise to the kernel execution.
  qpu_id: The id of the QPU. Defaults to 0.

Returns:
  AsyncRunResult: A handle which can be waited on via a `get()` method.
)#");
}
