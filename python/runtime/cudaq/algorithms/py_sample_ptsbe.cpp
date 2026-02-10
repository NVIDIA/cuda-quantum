/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_sample_ptsbe.h"
#include "common/DeviceCodeRegistry.h"
#include "cudaq/ptsbe/PTSBEOptions.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESampleResult.h"
#include "cudaq/ptsbe/PTSSamplingStrategy.h"
#include "cudaq/ptsbe/ShotAllocationStrategy.h"
#include "cudaq/ptsbe/strategies/ExhaustiveSamplingStrategy.h"
#include "cudaq/ptsbe/strategies/OrderedSamplingStrategy.h"
#include "cudaq/ptsbe/strategies/ProbabilisticSamplingStrategy.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace cudaq;

/// @brief Run PTSBE sampling from Python.
///
/// All PTSBE configuration is handled by the Python wrapper
/// (cudaq.ptsbe.sample) and passed here as positional parameters.
static ptsbe::sample_result pySamplePTSBE(
    const std::string &shortName, MlirModule module, MlirType returnTy,
    std::size_t shots_count, noise_model noiseModel,
    std::optional<std::size_t> max_trajectories, py::object sampling_strategy,
    py::object shot_allocation_obj, py::args runtimeArgs) {
  if (shots_count == 0)
    return ptsbe::sample_result();

  ptsbe::PTSBEOptions ptsbe_options;
  ptsbe_options.max_trajectories = max_trajectories;

  if (!sampling_strategy.is_none())
    ptsbe_options.strategy =
        sampling_strategy.cast<std::shared_ptr<ptsbe::PTSSamplingStrategy>>();

  if (!shot_allocation_obj.is_none())
    ptsbe_options.shot_allocation =
        shot_allocation_obj.cast<ptsbe::ShotAllocationStrategy>();

  auto mod = unwrap(module);
  runtimeArgs = simplifiedValidateInputArguments(runtimeArgs);
  auto retTy = unwrap(returnTy);
  auto &platform = get_platform();

  platform.set_noise(&noiseModel);

  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, runtimeArgs, fnOp);

  ptsbe::sample_result result;
  try {
    result = ptsbe::runSamplingPTSBE(
        [&]() mutable {
          [[maybe_unused]] auto res =
              clean_launch_module(shortName, mod, retTy, opaques);
        },
        platform, shortName, shots_count, ptsbe_options);
  } catch (const std::exception &e) {
    platform.reset_noise();
    throw std::runtime_error(std::string("cudaq.ptsbe.sample() failed: ") +
                             e.what());
  } catch (...) {
    platform.reset_noise();
    throw std::runtime_error(
        "cudaq.ptsbe.sample() failed with an unknown error.");
  }

  platform.reset_noise();
  return result;
}

/// @brief Async wrapper that holds the future for PTSBE sampling.
///
/// The future is a std::future<ptsbe::sample_result> which preserves the full
/// derived type (no slicing through KernelExecutionTask).
struct AsyncPTSBESampleResultImpl {
  ptsbe::async_sample_result future;

  explicit AsyncPTSBESampleResultImpl(ptsbe::async_sample_result &&f)
      : future(std::move(f)) {}

  ptsbe::sample_result get() { return future.get(); }
};

/// @brief Run PTSBE sampling asynchronously from Python.
///
/// Takes noise_model by reference so platform.set_noise() stores a pointer to
/// the pybind11-managed C++ object, not a stack-local copy. The Python wrapper
/// keeps the noise model alive until .get() is called.
static AsyncPTSBESampleResultImpl pySampleAsyncPTSBE(
    const std::string &shortName, MlirModule module, MlirType returnTy,
    std::size_t shots_count, noise_model &noiseModel,
    std::optional<std::size_t> max_trajectories, py::object sampling_strategy,
    py::object shot_allocation_obj, py::args runtimeArgs) {

  ptsbe::PTSBEOptions ptsbe_options;
  ptsbe_options.max_trajectories = max_trajectories;

  if (!sampling_strategy.is_none())
    ptsbe_options.strategy =
        sampling_strategy.cast<std::shared_ptr<ptsbe::PTSSamplingStrategy>>();

  if (!shot_allocation_obj.is_none())
    ptsbe_options.shot_allocation =
        shot_allocation_obj.cast<ptsbe::ShotAllocationStrategy>();

  auto mod = unwrap(module);
  runtimeArgs = simplifiedValidateInputArguments(runtimeArgs);
  auto retTy = unwrap(returnTy);
  auto &platform = get_platform();

  platform.set_noise(&noiseModel);

  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, runtimeArgs, fnOp);

  std::string kernelName = shortName;

  // Release GIL before launching async C++ work
  py::gil_scoped_release release;
  auto future = ptsbe::runSamplingAsyncPTSBE(
      [opaques = std::move(opaques), kernelName, retTy,
       mod = mod.clone()]() mutable {
        [[maybe_unused]] auto result =
            clean_launch_module(kernelName, mod, retTy, opaques);
      },
      platform, kernelName, shots_count, ptsbe_options);

  return AsyncPTSBESampleResultImpl(std::move(future));
}

void cudaq::bindSamplePTSBE(py::module &mod) {
  auto ptsbe = mod.def_submodule(
      "ptsbe", "PTSBE (Pre-Trajectory Sampling with Batch Execution)");

  // Base strategy class (abstract, not directly constructible)
  py::class_<ptsbe::PTSSamplingStrategy,
             std::shared_ptr<ptsbe::PTSSamplingStrategy>>(
      ptsbe, "PTSSamplingStrategy",
      "Base class for trajectory sampling strategies.")
      .def("name", &ptsbe::PTSSamplingStrategy::name,
           "Get the name of this strategy.");

  // Shot allocation strategy
  py::enum_<ptsbe::ShotAllocationStrategy::Type>(
      ptsbe, "ShotAllocationType",
      "Strategy type for allocating shots across trajectories.")
      .value("PROPORTIONAL", ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL,
             "Shots proportional to trajectory probability.")
      .value("UNIFORM", ptsbe::ShotAllocationStrategy::Type::UNIFORM,
             "Equal shots per trajectory.")
      .value("LOW_WEIGHT_BIAS",
             ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
             "Bias toward low-weight error trajectories.")
      .value("HIGH_WEIGHT_BIAS",
             ptsbe::ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS,
             "Bias toward high-weight error trajectories.");

  py::class_<ptsbe::ShotAllocationStrategy>(
      ptsbe, "ShotAllocationStrategy",
      "Strategy for allocating shots across selected trajectories.")
      .def(py::init<>(), "Create a default (PROPORTIONAL) strategy.")
      .def(py::init<ptsbe::ShotAllocationStrategy::Type, double>(),
           py::arg("type"), py::arg("bias_strength") = 2.0,
           "Create a strategy with specified type and optional bias strength.")
      .def_readwrite("type", &ptsbe::ShotAllocationStrategy::type,
                     "The allocation strategy type.")
      .def_readwrite("bias_strength",
                     &ptsbe::ShotAllocationStrategy::bias_strength,
                     "Bias factor for weighted strategies (default: 2.0).");

  // Concrete strategies
  py::class_<ptsbe::ProbabilisticSamplingStrategy, ptsbe::PTSSamplingStrategy,
             std::shared_ptr<ptsbe::ProbabilisticSamplingStrategy>>(
      ptsbe, "ProbabilisticSamplingStrategy",
      "Sample trajectories randomly based on their occurrence probabilities.")
      .def(py::init<std::uint64_t>(), py::arg("seed") = 0,
           "Create a probabilistic strategy with optional random seed.");

  py::class_<ptsbe::OrderedSamplingStrategy, ptsbe::PTSSamplingStrategy,
             std::shared_ptr<ptsbe::OrderedSamplingStrategy>>(
      ptsbe, "OrderedSamplingStrategy",
      "Sample trajectories sorted by probability in descending order.")
      .def(py::init<>(), "Create an ordered strategy.");

  py::class_<ptsbe::ExhaustiveSamplingStrategy, ptsbe::PTSSamplingStrategy,
             std::shared_ptr<ptsbe::ExhaustiveSamplingStrategy>>(
      ptsbe, "ExhaustiveSamplingStrategy",
      "Enumerate all possible trajectories in lexicographic order.")
      .def(py::init<>(), "Create an exhaustive strategy.");

  // PTSBE sample result (subclass of sample_result)
  py::class_<ptsbe::sample_result, sample_result>(ptsbe, "SampleResult",
                                                  "PTSBE sample result.");

  // Async PTSBE sample result wrapper
  py::class_<AsyncPTSBESampleResultImpl>(
      ptsbe, "AsyncSampleResultImpl",
      "Future-like wrapper for asynchronous PTSBE sampling.")
      .def("get", &AsyncPTSBESampleResultImpl::get,
           py::call_guard<py::gil_scoped_release>(),
           "Block until the PTSBE sampling result is available and return it.");

  // PTSBE sample implementation
  ptsbe.def("sample_impl", pySamplePTSBE,
            R"pbdoc(
Run PTSBE sampling on the provided kernel.

Args:
  kernel_name: The kernel name.
  module: The MLIR module.
  return_type: The MLIR return type.
  shots_count: The number of shots.
  noise_model: The noise model (required for PTSBE).
  max_trajectories: Maximum unique trajectories, or None to use shots.
  sampling_strategy: Sampling strategy or None for default (probabilistic).
  shot_allocation: Shot allocation strategy or None for default (proportional).
  *arguments: The kernel arguments.

Returns:
  SampleResult with PTSBE measurement counts.
)pbdoc");

  // PTSBE async sample implementation
  ptsbe.def("sample_async_impl", pySampleAsyncPTSBE,
            "Run PTSBE sampling asynchronously. Returns an "
            "AsyncSampleResultImpl.");
}
