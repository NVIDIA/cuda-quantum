/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_observe_async.h"
#include "common/Environment.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Todo.h"
#include "cudaq/algorithms/observe.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/NanobindAdaptors.h"
#include "utils/OpaqueArguments.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <fmt/core.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

using namespace cudaq;

namespace {
enum class PyParType { thread, mpi };
}

/// Analyze the MLIR Module for the kernel and check for CUDA-Q specification
/// adherence. Check that the kernel returns void and does not contain
/// measurements.
static std::tuple<bool, std::string>
isValidObserveKernel_impl(const std::string &kernelName, MlirModule kernelMod) {
  mlir::ModuleOp mod = unwrap(kernelMod);
  mlir::func::FuncOp kernelFunc = getKernelFuncOp(mod, kernelName);

  if (!kernelFunc) {
    std::string fullName = runtime::cudaqGenPrefixName + kernelName;
    mod.dump();
    throw std::runtime_error("kernel " + fullName + " must exist in module.");
  }

  // Do we have a return type?
  if (kernelFunc.getNumResults())
    return {false, "kernels passed to observe must have void return type."};

  // Are measurements specified?
  if (kernelFunc
          .walk([&](quake::MeasurementInterface measure) {
            // FIXME!! This is incorrect. If the kernel has calls, they are
            // completely ignored.
            return mlir::WalkResult::interrupt();
          })
          .wasInterrupted()) {
    return {false,
            "kernels passed to observe cannot have measurements specified."};
  }

  // Valid kernel...
  return {true, {}};
}

// The base `observe` launcher.
static async_observe_result pyObserveAsync(const std::string &shortName,
                                           mlir::ModuleOp mod,
                                           const spin_op &spin_operator,
                                           std::size_t qpu_id, int shots,
                                           nanobind::args args) {
  auto &platform = get_platform();
  args = simplifiedValidateInputArguments(args);
  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, args, fnOp);

  // Launch the asynchronous execution.
  nanobind::gil_scoped_release release;
  auto clonedMod = std::shared_ptr<mlir::ModuleOp>(
      new mlir::ModuleOp(mod.clone()), [](mlir::ModuleOp *p) {
        p->erase();
        delete p;
      });
  return details::runObservationAsync(
      detail::make_copyable_function(
          [opaques = std::move(opaques), shortName, clonedMod]() mutable {
            if (cudaq::getEnvBool("CUDAQ_DUMP_JIT_IR", false))
              clonedMod->dump();
            [[maybe_unused]] auto result =
                clean_launch_module(shortName, *clonedMod, opaques);
          }),
      spin_operator, platform, shots, shortName, qpu_id);
}

static async_observe_result
observe_async_impl(const std::string &shortName, MlirModule module,
                   nanobind::object &spin_operator_obj, std::size_t qpu_id,
                   int shots, nanobind::args args) {
  // FIXME(OperatorCpp): Remove this when the operator class is implemented in
  // C++
  spin_op spin_operator = [](nanobind::object &obj) -> spin_op {
    if (nanobind::hasattr(obj, "_to_spinop"))
      return nanobind::cast<spin_op>(obj.attr("_to_spinop")());
    return nanobind::cast<spin_op>(obj);
  }(spin_operator_obj);
  auto mod = unwrap(module);
  return pyObserveAsync(shortName, mod, spin_operator, qpu_id, shots, args);
}

/// @brief Run `cudaq::observe` on the provided kernel and spin operator.
static observe_result
pyObservePar(const PyParType &type, const std::string &shortName,
             mlir::ModuleOp module, spin_op &spin_operator, int shots,
             std::optional<noise_model> noise, nanobind::args args) {
  // Ensure the user input is correct.
  auto &platform = get_platform();
  if (!platform.supports_task_distribution())
    throw std::runtime_error(
        "The current quantum_platform does not support parallel distribution "
        "of observe() expectation value computations.");

  // FIXME Handle noise modeling with parallel distribution.
  if (noise)
    TODO("Handle Noise Models with python parallel distribution.");

  auto nQpus = platform.num_qpus();
  if (type == PyParType::thread) {
    // Does this platform expose more than 1 QPU
    // If so, let's distribute the work amongst the QPUs
    if (nQpus == 1)
      printf(
          "[cudaq::observe warning] distributed observe requested but only 1 "
          "QPU available. no speedup expected.\n");
    return details::distributeComputations(
        [&](std::size_t i, const spin_op &op) {
          return pyObserveAsync(shortName, module, op, i, shots, args);
        },
        spin_operator, nQpus);
  }

  if (!mpi::is_initialized())
    throw std::runtime_error("Cannot use mpi multi-node observe() without "
                             "MPI (did you initialize MPI?).");

  // Necessarily has to be MPI
  // Get the rank and the number of ranks
  auto rank = mpi::rank();
  auto nRanks = mpi::num_ranks();

  // Each rank gets a subset of the spin terms
  auto spins = spin_operator.distribute_terms(nRanks);

  // Get this rank's set of spins to compute
  auto localH = spins[rank];

  // Distribute locally, i.e. to the local nodes QPUs
  auto localRankResult = details::distributeComputations(
      [&](std::size_t i, const spin_op &op) {
        return pyObserveAsync(shortName, module, op, i, shots, args);
      },
      localH, nQpus);

  // combine all the data via an all_reduce
  auto exp_val = localRankResult.expectation();
  auto globalExpVal = mpi::all_reduce(exp_val, std::plus<double>());
  return observe_result{globalExpVal, spin_operator};
}

/// Observe can be a single observe call, a parallel observe call, or a observe
/// broadcast. All these variants are handled here.
static observe_result observe_parallel_impl(const std::string &shortName,
                                            MlirModule module,
                                            nanobind::type_object execution,
                                            spin_op &spin_operator, int shots,
                                            std::optional<noise_model> noise,
                                            nanobind::args arguments) {
  std::string applicatorKey =
      nanobind::cast<std::string>(execution.attr("__name__"));
  auto mod = unwrap(module);
  if (applicatorKey == "thread")
    return pyObservePar(PyParType::thread, shortName, mod, spin_operator, shots,
                        noise, arguments);
  if (applicatorKey == "mpi")
    return pyObservePar(PyParType::mpi, shortName, mod, spin_operator, shots,
                        noise, arguments);
  throw std::runtime_error("invalid parallel execution context");
}

void cudaq::bindObserveAsync(nanobind::module_ &mod) {
  auto parallelSubmodule = mod.def_submodule("parallel");
  nanobind::class_<parallel::mpi>(
      parallelSubmodule, "mpi",
      "Type indicating that the :func:`observe` function should distribute its "
      "expectation value computations across available MPI ranks and GPUs for "
      "each term.");
  nanobind::class_<parallel::thread>(
      parallelSubmodule, "thread",
      "Type indicating that the :func:`observe` function should distribute its "
      "term "
      "expectation value computations across available GPUs via standard C++ "
      "threads.");

  mod.def("observe_async_impl", observe_async_impl,
          "See the python documentation for `observe_async`.");

  mod.def("isValidObserveKernel_impl", isValidObserveKernel_impl,
          "Test to see if the kernel is suited for use with observe.");

  mod.def("observe_parallel_impl", observe_parallel_impl,
          "See the python documentation for observe_parallel.");
}
