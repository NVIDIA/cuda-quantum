/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_run.h"
#include "common/RecordLogParser.h"
#include "cudaq_internal/compiler/LayoutInfo.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "cudaq/algorithms/run.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <exception>
#include <future>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

using namespace cudaq;

static std::vector<nanobind::object>
readRunResults(mlir::ModuleOp module, mlir::Type ty,
               detail::RunResultSpan &results, std::size_t count) {
  std::vector<nanobind::object> ret;
  if (count == 0)
    return ret;
  std::size_t byteSize = results.lengthInBytes / count;
  if (byteSize == 0)
    throw std::runtime_error("run: result buffer (" +
                             std::to_string(results.lengthInBytes) +
                             " bytes) is too small for " +
                             std::to_string(count) + " decoded results.");
  if (results.lengthInBytes % count != 0)
    throw std::runtime_error("run: result buffer (" +
                             std::to_string(results.lengthInBytes) +
                             " bytes) is not evenly divisible by " +
                             std::to_string(count) + " decoded results.");
  for (std::size_t i = 0; i < results.lengthInBytes; i += byteSize) {
    nanobind::object obj = convertResult(module, ty, results.data + i);
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

static std::unique_ptr<RecordLogParser> pyRunTheKernel(
    const std::string &name, quantum_platform &platform, mlir::ModuleOp mod,
    std::shared_ptr<detail::CompiledModuleCache> cache, std::size_t shots_count,
    std::size_t qpu_id, OpaqueArguments &opaques) {
  if (!name.ends_with(".run"))
    throw std::runtime_error("`cudaq.run` only supports runnable kernels.");
  // Set the `run` attribute on the module to indicate this is a run context
  // (for result handling).
  mod->setAttr(runtime::enableCudaqRun, mlir::UnitAttr::get(mod->getContext()));

  auto returnTy = recoverReturnType(mod, name);
  // Disallow returning nested vectors/vectors of structs from entry-point
  // kernels.
  if (auto vecTy = dyn_cast<cudaq::cc::SequenceType>(returnTy)) {
    auto elemTy = vecTy.getElementType();
    if (mlir::isa<cudaq::cc::SequenceType>(elemTy))
      throw std::runtime_error(
          "`cudaq.run` does not yet support returning nested `list` from "
          "entry-point kernels.");
    if (mlir::isa<cudaq::cc::StructType>(elemTy))
      throw std::runtime_error("`cudaq.run` does not yet support returning "
                               "`list` of `dataclass`/`tuple` from "
                               "entry-point kernels.");
  }
  auto layoutInfo =
      cudaq_internal::compiler::getLayoutInfo(name, mod.getOperation());
  cudaq::run_result runResult = detail::launchRun(
      [&]() mutable {
        [[maybe_unused]] auto result =
            clean_launch_module(name, mod, opaques, cache);
      },
      platform, name, shots_count, qpu_id);

  // Parse here rather than through `convertToRunResultSpan`: the parser keeps
  // owning the result storage and frees it once the results are read.
  auto parser = std::make_unique<RecordLogParser>(layoutInfo);
  parser->parse(runResult.outputLog);
  return parser;
}

static std::vector<nanobind::object>
pyReadResults(const RecordLogParser &parser, mlir::ModuleOp mod,
              const std::string &name) {
  auto returnTy = recoverReturnType(mod, name);
  detail::RunResultSpan results{static_cast<char *>(parser.getBufferPtr()),
                                parser.getBufferSize(),
                                parser.getResultCount()};
  return readRunResults(mod, returnTy, results, results.resultCount);
}

/// @brief Run `cudaq::run` on the provided kernel.
static std::vector<nanobind::object>
run_impl(const std::string &shortName, MlirModule module,
         std::shared_ptr<detail::CompiledModuleCache> cache,
         std::size_t shots_count, std::optional<noise_model> noise_model,
         std::size_t qpu_id, nanobind::args runtimeArgs) {
  if (shots_count == 0)
    return {};

  auto mod = unwrap(module);
  auto &platform = get_platform();
  if (noise_model.has_value() && platform.is_remote())
    throw std::runtime_error(
        "Noise model is not supported on remote platforms.");
  auto fnOp = getFuncOpAndCheckResult(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(
      mod, runtimeArgs, fnOp, launchSpecializesArguments(qpu_id));

  std::unique_ptr<RecordLogParser> parser;
  {
    nanobind::gil_scoped_release release;
    if (noise_model.has_value())
      // Launch the kernel in the appropriate context.
      platform.set_noise(&noise_model.value(), qpu_id);
    parser = detail::try_finally(
        [&] {
          return pyRunTheKernel(shortName, platform, mod, std::move(cache),
                                shots_count, qpu_id, opaques);
        },
        [&] {
          if (noise_model.has_value())
            platform.reset_noise(qpu_id);
        });
  }
  return pyReadResults(*parser, mod, shortName);
}

namespace {
// The parsed output log of a queued run. `get()` converts it on every call, so
// the handle holds no Python object. A Python error kept here would collect the
// frames of each `get()` caller, which can refer back to this handle through a
// reference the garbage collector cannot see.
struct PyAsyncRunResult {
  std::shared_future<std::unique_ptr<RecordLogParser>> parser;
  // Owned by the Python handle, `AsyncRunResult.mod`.
  mlir::ModuleOp mod;
  std::string shortName;
};
} // namespace

/// @brief Run `cudaq::run_async` on the provided kernel.
static PyAsyncRunResult
run_async_impl(const std::string &shortName, MlirModule module,
               std::shared_ptr<detail::CompiledModuleCache> cache,
               std::size_t shots_count, std::optional<noise_model> noise_model,
               std::size_t qpu_id, nanobind::args runtimeArgs) {
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
  if (noise_model.has_value() && platform.is_remote())
    throw std::runtime_error(
        "Noise model is not supported on remote platforms.");

  std::promise<std::unique_ptr<RecordLogParser>> parserPromise;
  PyAsyncRunResult result{parserPromise.get_future().share(), mod, shortName};

  if (shots_count == 0) {
    parserPromise.set_value(nullptr);
    return result;
  }

  auto fnOp = getFuncOpAndCheckResult(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(
      mod, runtimeArgs, fnOp, launchSpecializesArguments(qpu_id));
  // Run the kernel and parse its output log.
  {
    // Release GIL to allow c++ threads, all code inside the scope is c++, so
    // there is no need to re-acquire the GIL inside the thread.
    nanobind::gil_scoped_release gil_release{};
    QuantumTask wrapped = detail::make_copyable_function(
        [pp = std::move(parserPromise), noise_model = std::move(noise_model),
         qpu_id, name = shortName, opaques = std::move(opaques), shots_count,
         cache,
         clonedMod = std::shared_ptr<mlir::ModuleOp>(
             new mlir::ModuleOp(mod.clone()), [](mlir::ModuleOp *p) {
               p->erase();
               delete p;
             })]() mutable {
          auto &platform = get_platform();

          std::unique_ptr<RecordLogParser> parser;
          std::exception_ptr error;
          try {
            // Launch the kernel in the appropriate context.
            if (noise_model.has_value())
              platform.set_noise(&noise_model.value(), qpu_id);
            parser = pyRunTheKernel(name, platform, *clonedMod, cache,
                                    shots_count, qpu_id, opaques);
          } catch (...) {
            error = std::current_exception();
          }

          if (noise_model.has_value())
            detail::invoke_no_throw([&]() { platform.reset_noise(qpu_id); });

          // Erase the clone before signaling, so that it is gone by the time
          // `get()` returns. Otherwise whichever thread drops the last copy of
          // this task erases it, possibly after `get()`.
          clonedMod.reset();
          if (error)
            pp.set_exception(std::move(error));
          else
            pp.set_value(std::move(parser));
        });
    platform.enqueueAsyncTask(qpu_id, wrapped);
  }

  return result;
}

/// @brief Bind the run cudaq function.
void cudaq::bindPyRun(nanobind::module_ &mod) {
  mod.def("run_impl", run_impl, nanobind::arg(), nanobind::arg(),
          nanobind::arg(), nanobind::arg(), nanobind::arg().none(),
          nanobind::arg(), nanobind::arg(),
          R"#(
Run the provided `kernel` with the given kernel arguments over the specified
number of circuit executions (`shots_count`).

Args:
  kernel: The kernel to execute `shots_count` times on the QPU.
  *arguments: The concrete values to evaluate the kernel function at.
  shots_count: The number of kernel executions on the QPU. Defaults to 100.
  noise_model: The optional noise model to add noise to the kernel execution.

Returns:
  A list containing the return value from each successful execution. Its length
  may be less than `shots_count` if the backend reports failed shots.
)#");
}

/// @brief Bind the run_async cudaq function.
void cudaq::bindPyRunAsync(nanobind::module_ &mod) {
  nanobind::class_<PyAsyncRunResult>(mod, "AsyncRunResultImpl", "")
      .def(
          "get",
          [](PyAsyncRunResult &self) {
            {
              // Release the GIL so the async task's MLIR worker threads
              // can call PyGILState_Ensure without deadlocking on us.
              nanobind::gil_scoped_release release;
              self.parser.wait();
            }
            const RecordLogParser *parser = nullptr;
            try {
              // A failed run raises its error from every `get()`.
              parser = self.parser.get().get();
            } catch (const nanobind::python_error &e) {
              // A Python callback in the queued run, such as a noise channel,
              // raised it. Raising a `python_error` hands the error to Python
              // and empties `e`, so a later `get()` raises its text instead.
              if (!e.value())
                throw std::runtime_error(e.what());
              e.what(); // Keep the text before the error is handed over.
              throw;
            }
            // A null parser means `shots_count` was 0.
            if (!parser)
              return std::vector<nanobind::object>{};
            return pyReadResults(*parser, self.mod, self.shortName);
          },
          "FIXME: documentation goes here");

  mod.def("run_async_impl", run_async_impl, nanobind::arg(), nanobind::arg(),
          nanobind::arg(), nanobind::arg(), nanobind::arg().none(),
          nanobind::arg(), nanobind::arg(),
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
  AsyncRunResult: A handle whose `get()` method returns one value per successful
  execution. The result may contain fewer than `shots_count` values if the
  backend reports failed shots.
)#");
}
