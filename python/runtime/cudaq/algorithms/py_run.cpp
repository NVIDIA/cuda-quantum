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

using namespace cudaq;

static std::vector<py::object> readRunResults(mlir::ModuleOp module,
                                              mlir::Type ty,
                                              details::RunResultSpan &results,
                                              std::size_t count) {
  std::vector<py::object> ret;
  std::size_t byteSize = results.lengthInBytes / count;
  for (std::size_t i = 0; i < results.lengthInBytes; i += byteSize) {
    py::object obj = convertResult(module, ty, results.data + i);
    ret.push_back(obj);
  }
  return ret;
}

static mlir::Type recoverReturnType(mlir::ModuleOp mod,
                                    const std::string &name) {
  auto *fn = mod.lookupSymbol(runtime::cudaqGenPrefixName + name);
  auto retTys =
      mlir::cast<mlir::ArrayAttr>(fn->getAttr(runtime::enableCudaqRun));
  if (retTys.size() != 1)
    throw std::runtime_error("runnable kernel must return exactly one result.");
  return mlir::cast<mlir::TypeAttr>(retTys[0]).getValue();
}

static mlir::func::FuncOp
getFuncOpAndCheckResult(mlir::ModuleOp mod, const std::string &shortName) {
  auto fn = getKernelFuncOp</*noThrow=*/true>(mod, shortName);
  if (!fn)
    throw std::runtime_error("a runnable kernel must return a value.");
  if (!fn->hasAttr(runtime::enableCudaqRun))
    throw std::runtime_error("runnable kernel must be properly constructed.");
  return fn;
}

static details::RunResultSpan
pyRunTheKernel(const std::string &name, quantum_platform &platform,
               mlir::ModuleOp mod, mlir::Type retTy, std::size_t shots_count,
               std::size_t qpu_id, OpaqueArguments &opaques) {
  if (!name.ends_with(".run"))
    throw std::runtime_error("`cudaq.run` only supports runnable kernels.");
  // Set the `run` attribute on the module to indicate this is a run context
  // (for result handling).
  mod->setAttr(runtime::enableCudaqRun, mlir::UnitAttr::get(mod->getContext()));

  auto returnTy = recoverReturnType(mod, name);
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
  auto results = details::runTheKernel(
      [&]() mutable {
        [[maybe_unused]] auto result =
            clean_launch_module(name, mod, retTy, opaques);
      },
      platform, name, name, shots_count, qpu_id, mod.getOperation());

  return results;
}

static std::vector<py::object> pyReadResults(details::RunResultSpan results,
                                             mlir::ModuleOp mod,
                                             std::size_t shots_count,
                                             const std::string &name) {
  auto returnTy = recoverReturnType(mod, name);
  return readRunResults(mod, returnTy, results, shots_count);
}

/// @brief Run `cudaq::run` on the provided kernel.
static std::vector<py::object> pyRun(const std::string &shortName,
                                     MlirModule module, MlirType returnTy,
                                     std::size_t shots_count,
                                     std::optional<noise_model> noise_model,
                                     std::size_t qpu_id, py::args runtimeArgs) {
  if (shots_count == 0)
    return {};

  auto mod = unwrap(module);
  auto &platform = get_platform();
  if (noise_model.has_value()) {
    if (platform.is_remote())
      throw std::runtime_error(
          "Noise model is not supported on remote platforms.");
    // Launch the kernel in the appropriate context.
    platform.set_noise(&noise_model.value());
  }
  auto retTy = unwrap(returnTy);
  auto fnOp = getFuncOpAndCheckResult(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, runtimeArgs, fnOp);
  auto span = pyRunTheKernel(shortName, platform, mod, retTy, shots_count,
                             qpu_id, opaques);
  auto results = pyReadResults(span, mod, shots_count, shortName);

  if (noise_model.has_value())
    platform.reset_noise();

  return results;
}

namespace {
// Internal struct representing buffer to be filled asynchronously.
// When the `ready` future is set, the content of the buffer is filled.
struct async_run_result {
  std::future<void> ready;
  std::vector<py::object> *results;
  std::string *error;
};
} // namespace

/// @brief Run `cudaq::run_async` on the provided kernel.
static async_run_result run_async_impl(const std::string &shortName,
                                       MlirModule module, MlirType returnTy,
                                       std::size_t shots_count,
                                       std::optional<noise_model> noise_model,
                                       std::size_t qpu_id,
                                       py::args runtimeArgs) {
  if (!shots_count)
    return {};

  auto &platform = get_platform();
  auto numQPUs = platform.num_qpus();
  if (qpu_id >= numQPUs)
    throw std::runtime_error("qpu_id (" + std::to_string(qpu_id) +
                             ") exceeds the number of available QPUs (" +
                             std::to_string(numQPUs) + ")");

  auto mod = unwrap(module);
  // Set the `run` attribute on the module to indicate this is a run context
  // (for result handling).
  mod->setAttr(runtime::enableCudaqRun, mlir::UnitAttr::get(mod->getContext()));
  auto retTy = unwrap(returnTy);
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

  auto fnOp = getFuncOpAndCheckResult(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, runtimeArgs, fnOp);
  // Run the kernel and compute results span.
  {
    // Release GIL to allow c++ threads, all code inside the scope is c++, so
    // there is no need to re-acquire the GIL inside the thread.
    py::gil_scoped_release gil_release{};
    QuantumTask wrapped = detail::make_copyable_function(
        [sp = std::move(spanPromise), ep = std::move(errorPromise),
         noise_model = std::move(noise_model), qpu_id, name = shortName,
         opaques = std::move(opaques), shots_count, retTy,
         mod = mod.clone()]() mutable {
          auto &platform = get_platform();

          // Launch the kernel in the appropriate context.
          if (noise_model.has_value())
            platform.set_noise(&noise_model.value());
          try {
            auto span = pyRunTheKernel(name, platform, mod, retTy, shots_count,
                                       qpu_id, opaques);
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
                    errorPtr = result.error, resultsPtr = result.results, mod,
                    shots_count, shortName]() mutable {
                     auto error = ef.get();
                     std::swap(*errorPtr, error);
                     if (error.empty()) {
                       auto span = sf.get();
                       py::gil_scoped_acquire gil{};
                       auto results =
                           pyReadResults(span, mod, shots_count, shortName);
                       std::swap(*resultsPtr, results);
                     }
                   });
    result.ready = std::move(resultFuture);
  }

  return result;
}

/// @brief Bind the run cudaq function.
void cudaq::bindPyRun(py::module &mod) {
  mod.def("run_impl", pyRun,
          R"#(
Run the provided `kernel` with the given kernel arguments over the specified
number of circuit executions (`shots_count`).

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
  py::class_<async_run_result>(mod, "AsyncRunResultImpl", "")
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
          "FIXME: documentation goes here");

  mod.def("run_async_impl", run_async_impl,
          R"#(
Run the provided `kernel` with the given kernel arguments over the specified
number of circuit executions (`shots_count`) asynchronously on the specified
`qpu_id`.

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
