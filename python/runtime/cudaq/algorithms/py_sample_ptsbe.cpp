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
#include "cudaq/ptsbe/PTSBESampleIntegration.h"
#include "cudaq/ptsbe/PTSSamplingStrategy.h"
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
static sample_result pySamplePTSBE(const std::string &shortName,
                                   MlirModule module, MlirType returnTy,
                                   std::size_t shots_count,
                                   noise_model noiseModel,
                                   std::optional<std::size_t> max_trajectories,
                                   py::object sampling_strategy,
                                   py::args runtimeArgs) {
  if (shots_count == 0)
    return sample_result();

  ptsbe::PTSBEOptions ptsbe_options;
  ptsbe_options.max_trajectories = max_trajectories;

  if (!sampling_strategy.is_none())
    ptsbe_options.strategy =
        sampling_strategy.cast<std::shared_ptr<ptsbe::PTSSamplingStrategy>>();

  auto mod = unwrap(module);
  runtimeArgs = simplifiedValidateInputArguments(runtimeArgs);
  auto retTy = unwrap(returnTy);
  auto &platform = get_platform();

  platform.set_noise(&noiseModel);

  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, runtimeArgs, fnOp);

  sample_result result;
  try {
    result = ptsbe::runSamplingPTSBE(
        [&]() mutable {
          [[maybe_unused]] auto res =
              clean_launch_module(shortName, mod, retTy, opaques);
        },
        platform, shortName, shots_count, ptsbe_options);
  } catch (...) {
    platform.reset_noise();
    throw;
  }

  platform.reset_noise();
  return result;
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
  *arguments: The kernel arguments.

Returns:
  SampleResult with measurement results.
)pbdoc");
}
