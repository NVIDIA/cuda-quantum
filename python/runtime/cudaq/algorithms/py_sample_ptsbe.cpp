/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_sample_ptsbe.h"
#include "common/DeviceCodeRegistry.h"
#include "cudaq/ptsbe/KrausSelection.h"
#include "cudaq/ptsbe/KrausTrajectory.h"
#include "cudaq/ptsbe/PTSBEExecutionData.h"
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
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

using namespace cudaq;

/// @brief Run PTSBE sampling from Python.
///
/// All PTSBE configuration is handled by the Python wrapper
/// (cudaq.ptsbe.sample) and passed here as positional parameters.
// nanobind 2.x cannot dispatch NB_TYPE_CASTER-based parameters (MlirModule)
// when nanobind::object appears in the same function signature. Use concrete
// std::optional types for all nullable parameters instead.
static ptsbe::sample_result
pySamplePTSBE(const std::string &shortName, MlirModule module,
              std::size_t shots_count, noise_model noiseModel,
              std::optional<std::size_t> max_trajectories,
              std::optional<std::shared_ptr<ptsbe::PTSSamplingStrategy>>
                  sampling_strategy,
              std::optional<ptsbe::ShotAllocationStrategy> shot_allocation,
              bool return_execution_data, bool include_sequential_data,
              nanobind::args runtimeArgs) {
  if (shots_count == 0)
    return ptsbe::sample_result();

  ptsbe::PTSBEOptions ptsbe_options;
  ptsbe_options.return_execution_data = return_execution_data;
  ptsbe_options.include_sequential_data = include_sequential_data;
  ptsbe_options.max_trajectories = max_trajectories;

  if (sampling_strategy)
    ptsbe_options.strategy = *sampling_strategy;

  if (shot_allocation)
    ptsbe_options.shot_allocation = *shot_allocation;

  auto mod = unwrap(module);
  runtimeArgs = simplifiedValidateInputArguments(runtimeArgs);
  auto &platform = get_platform();

  platform.set_noise(&noiseModel);

  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, runtimeArgs, fnOp);

  ptsbe::sample_result result;
  try {
    nanobind::gil_scoped_release release;
    result = ptsbe::detail::runSamplingPTSBE(
        [&]() mutable {
          [[maybe_unused]] auto res =
              clean_launch_module(shortName, mod, opaques);
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

namespace {
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
} // namespace

/// @brief Run PTSBE sampling asynchronously from Python.
static AsyncPTSBESampleResultImpl
pySampleAsyncPTSBE(const std::string &shortName, MlirModule module,
                   std::size_t shots_count, noise_model &noiseModel,
                   std::optional<std::size_t> max_trajectories,
                   std::optional<std::shared_ptr<ptsbe::PTSSamplingStrategy>>
                       sampling_strategy,
                   std::optional<ptsbe::ShotAllocationStrategy> shot_allocation,
                   bool return_execution_data, bool include_sequential_data,
                   nanobind::args runtimeArgs) {

  ptsbe::PTSBEOptions ptsbe_options;
  ptsbe_options.return_execution_data = return_execution_data;
  ptsbe_options.include_sequential_data = include_sequential_data;
  ptsbe_options.max_trajectories = max_trajectories;

  if (sampling_strategy)
    ptsbe_options.strategy = *sampling_strategy;

  if (shot_allocation)
    ptsbe_options.shot_allocation = *shot_allocation;

  auto mod = unwrap(module);
  runtimeArgs = simplifiedValidateInputArguments(runtimeArgs);
  auto &platform = get_platform();

  auto fnOp = getKernelFuncOp(mod, shortName);
  auto opaques = marshal_arguments_for_module_launch(mod, runtimeArgs, fnOp);

  std::string kernelName = shortName;

  // Release GIL before launching async C++ work
  nanobind::gil_scoped_release release;
  return AsyncPTSBESampleResultImpl(ptsbe::detail::runSamplingAsyncPTSBE(
      [opaques = std::move(opaques), kernelName, mod = mod.clone()]() mutable {
        [[maybe_unused]] auto result =
            clean_launch_module(kernelName, mod, opaques);
      },
      platform, kernelName, shots_count, ptsbe_options, /*qpu_id=*/0,
      noiseModel));
}

void cudaq::bindSamplePTSBE(nanobind::module_ &mod) {
  auto ptsbe = mod.def_submodule(
      "ptsbe", "PTSBE (Pre-Trajectory Sampling with Batch Execution)");

  // Base strategy class (abstract, not directly constructible)
  nanobind::class_<ptsbe::PTSSamplingStrategy>(
      ptsbe, "PTSSamplingStrategy",
      "Base class for trajectory sampling strategies.")
      .def("name", &ptsbe::PTSSamplingStrategy::name,
           "Get the name of this strategy.");

  // Shot allocation strategy
  nanobind::enum_<ptsbe::ShotAllocationStrategy::Type>(
      ptsbe, "ShotAllocationType",
      "Strategy type for allocating shots across trajectories.")
      .value("PROPORTIONAL", ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL,
             "Shots proportional to trajectory weight.")
      .value("UNIFORM", ptsbe::ShotAllocationStrategy::Type::UNIFORM,
             "Equal shots per trajectory.")
      .value("LOW_WEIGHT_BIAS",
             ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
             "Bias toward low-weight error trajectories.")
      .value("HIGH_WEIGHT_BIAS",
             ptsbe::ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS,
             "Bias toward high-weight error trajectories.");

  nanobind::class_<ptsbe::ShotAllocationStrategy>(
      ptsbe, "ShotAllocationStrategy",
      "Strategy for allocating shots across selected trajectories.")
      .def(nanobind::init<>(), "Create a default (PROPORTIONAL) strategy.")
      .def(
          "__init__",
          [](ptsbe::ShotAllocationStrategy *self,
             ptsbe::ShotAllocationStrategy::Type t, double bias,
             std::optional<std::uint64_t> seed) {
            new (self) ptsbe::ShotAllocationStrategy(t, bias, seed);
          },
          nanobind::arg("type"), nanobind::arg("bias_strength") = 2.0,
          nanobind::arg("seed") = nanobind::none(),
          "Create a strategy with specified type, optional bias strength, "
          "and optional random seed. When seed is None (default), uses "
          "CUDA-Q's global random seed.")
      .def_rw("type", &ptsbe::ShotAllocationStrategy::type,
              "The allocation strategy type.")
      .def_rw("bias_strength", &ptsbe::ShotAllocationStrategy::bias_strength,
              "Bias factor for weighted strategies. Default value is 2.0.");

  // Concrete strategies
  nanobind::class_<ptsbe::ProbabilisticSamplingStrategy,
                   ptsbe::PTSSamplingStrategy>(
      ptsbe, "ProbabilisticSamplingStrategy",
      "Sample trajectories randomly based on their occurrence probabilities.")
      .def(nanobind::init<std::optional<std::uint64_t>,
                          std::optional<std::size_t>>(),
           nanobind::arg("seed") = nanobind::none(),
           nanobind::arg("max_trajectory_samples") = nanobind::none(),
           "Create a probabilistic strategy with optional random seed and "
           "max trajectory sample count. When seed is None (default), uses "
           "CUDA-Q's global random seed. "
           "max_trajectory_samples sets a ceiling on Monte Carlo draws. "
           "The loop stops early once max_trajectories unique patterns are "
           "found. When None (default), a budget is auto-calculated.");

  nanobind::class_<ptsbe::OrderedSamplingStrategy, ptsbe::PTSSamplingStrategy>(
      ptsbe, "OrderedSamplingStrategy",
      "Sample trajectories sorted by probability in descending order.")
      .def(nanobind::init<>(), "Create an ordered strategy.");

  nanobind::class_<ptsbe::ExhaustiveSamplingStrategy,
                   ptsbe::PTSSamplingStrategy>(
      ptsbe, "ExhaustiveSamplingStrategy",
      "Enumerate all possible trajectories in lexicographic order.")
      .def(nanobind::init<>(), "Create an exhaustive strategy.");

  // Trace instruction type enum
  nanobind::enum_<ptsbe::TraceInstructionType>(
      ptsbe, "TraceInstructionType",
      "Type discriminator for trace instructions.")
      .value("Gate", ptsbe::TraceInstructionType::Gate)
      .value("Noise", ptsbe::TraceInstructionType::Noise)
      .value("Measurement", ptsbe::TraceInstructionType::Measurement)
      .export_values();

  // Trace instruction
  nanobind::class_<ptsbe::TraceInstruction>(
      ptsbe, "TraceInstruction", "Single operation in the execution trace.")
      .def_prop_ro(
          "type", [](const ptsbe::TraceInstruction &self) { return self.type; })
      .def_prop_ro(
          "name", [](const ptsbe::TraceInstruction &self) { return self.name; })
      .def_prop_ro("targets",
                   [](const ptsbe::TraceInstruction &self) {
                     return std::vector<std::size_t>(self.targets.begin(),
                                                     self.targets.end());
                   })
      .def_prop_ro("controls",
                   [](const ptsbe::TraceInstruction &self) {
                     return std::vector<std::size_t>(self.controls.begin(),
                                                     self.controls.end());
                   })
      .def_prop_ro("params",
                   [](const ptsbe::TraceInstruction &self) {
                     return std::vector<double>(self.params.begin(),
                                                self.params.end());
                   })
      .def_prop_ro("channel",
                   [](const ptsbe::TraceInstruction &self) -> nanobind::object {
                     if (!self.channel)
                       return nanobind::none();
                     return nanobind::cast(*self.channel);
                   })
      .def("__repr__", [](const ptsbe::TraceInstruction &self) {
        return "TraceInstruction(" + self.name + " on " +
               std::to_string(self.targets.size()) + " qubits)";
      });

  // Kraus selection (cudaq:: namespace)
  nanobind::class_<KrausSelection>(
      ptsbe, "KrausSelection",
      "Reference to a single Kraus operator selection.")
      .def_prop_ro(
          "circuit_location",
          [](const KrausSelection &self) { return self.circuit_location; })
      .def_prop_ro(
          "kraus_operator_index",
          [](const KrausSelection &self) { return self.kraus_operator_index; })
      .def_prop_ro("is_error",
                   [](const KrausSelection &self) { return self.is_error; })
      .def_prop_ro("qubits",
                   [](const KrausSelection &self) { return self.qubits; })
      .def_prop_ro("op_name",
                   [](const KrausSelection &self) { return self.op_name; })
      .def("__repr__", [](const KrausSelection &self) {
        return "KrausSelection(loc=" + std::to_string(self.circuit_location) +
               ", idx=" + std::to_string(self.kraus_operator_index) +
               ", error=" + (self.is_error ? "true" : "false") + ")";
      });

  // Kraus trajectory (cudaq:: namespace)
  nanobind::class_<KrausTrajectory>(
      ptsbe, "KrausTrajectory",
      "Complete specification of one noise trajectory with outcomes.")
      .def_prop_ro(
          "trajectory_id",
          [](const KrausTrajectory &self) { return self.trajectory_id; })
      .def_prop_ro("probability",
                   [](const KrausTrajectory &self) { return self.probability; })
      .def_prop_ro("num_shots",
                   [](const KrausTrajectory &self) { return self.num_shots; })
      .def_ro("multiplicity", &KrausTrajectory::multiplicity,
              "Number of times this trajectory was sampled.")
      .def_ro("weight", &KrausTrajectory::weight,
              "Allocation weight for shot distribution.")
      .def_prop_ro(
          "kraus_selections",
          [](const KrausTrajectory &self) { return self.kraus_selections; },
          nanobind::rv_policy::reference_internal)
      .def_prop_ro(
          "measurement_counts",
          [](const KrausTrajectory &self) { return self.measurement_counts; })
      .def("__repr__", [](const KrausTrajectory &self) {
        return "KrausTrajectory(id=" + std::to_string(self.trajectory_id) +
               ", p=" + std::to_string(self.probability) +
               ", shots=" + std::to_string(self.num_shots) + ")";
      });

  // PTSBE execution data container
  nanobind::class_<ptsbe::PTSBEExecutionData>(
      ptsbe, "PTSBEExecutionData",
      "Container for PTSBE execution data including circuit structure, "
      "trajectory specifications, and per-trajectory measurement outcomes.")
      .def_prop_ro(
          "instructions",
          [](const ptsbe::PTSBEExecutionData &self)
              -> const std::vector<ptsbe::TraceInstruction> & {
            return self.instructions;
          },
          nanobind::rv_policy::reference_internal)
      .def_prop_ro(
          "trajectories",
          [](const ptsbe::PTSBEExecutionData &self)
              -> const std::vector<cudaq::KrausTrajectory> & {
            return self.trajectories;
          },
          nanobind::rv_policy::reference_internal)
      .def(
          "count_instructions",
          [](const ptsbe::PTSBEExecutionData &self,
             ptsbe::TraceInstructionType type,
             nanobind::object name) -> std::size_t {
            std::optional<std::string> nameOpt;
            if (!name.is_none())
              nameOpt = nanobind::cast<std::string>(name);
            return self.count_instructions(type, nameOpt);
          },
          nanobind::arg("type"), nanobind::arg("name") = nanobind::none(),
          "Count instructions of a given type.")
      .def(
          "get_trajectory",
          [](const ptsbe::PTSBEExecutionData &self,
             std::size_t trajectory_id) -> const cudaq::KrausTrajectory * {
            auto result = self.get_trajectory(trajectory_id);
            if (!result.has_value())
              return nullptr;
            return &result.value().get();
          },
          nanobind::rv_policy::reference_internal,
          nanobind::arg("trajectory_id"),
          "Look up a trajectory by its ID. Returns None if not found.")
      .def("__repr__",
           [](const ptsbe::PTSBEExecutionData &self) {
             return "PTSBEExecutionData(" +
                    std::to_string(self.instructions.size()) +
                    " instructions, " +
                    std::to_string(self.trajectories.size()) + " trajectories)";
           })
      .def("__len__", [](const ptsbe::PTSBEExecutionData &self) {
        return self.instructions.size();
      });

  // PTSBE sample result (subclass of sample_result)
  nanobind::class_<ptsbe::sample_result, sample_result>(
      ptsbe, "PTSBESampleResult",
      "PTSBE sample result with optional execution data.")
      .def_prop_ro(
          "ptsbe_execution_data",
          [](const ptsbe::sample_result &self)
              -> const ptsbe::PTSBEExecutionData * {
            if (self.has_execution_data())
              return &self.execution_data();
            return nullptr;
          },
          // reference_internal ties the returned object's lifetime to self,
          // so the pointer into internal data stays valid.
          nanobind::rv_policy::reference_internal,
          "PTSBE execution data if return_execution_data was True, None "
          "otherwise.")
      .def("has_execution_data", &ptsbe::sample_result::has_execution_data,
           "Check if execution data is available.");

  // Async PTSBE sample result wrapper
  nanobind::class_<AsyncPTSBESampleResultImpl>(
      ptsbe, "AsyncSampleResultImpl",
      "Future-like wrapper for asynchronous PTSBE sampling.")
      .def("get", &AsyncPTSBESampleResultImpl::get,
           nanobind::call_guard<nanobind::gil_scoped_release>(),
           "Block until the PTSBE sampling result is available and return it.");

  // PTSBE sample implementation
  ptsbe.def(
      "sample_impl", pySamplePTSBE, nanobind::arg("kernel_name"),
      nanobind::arg("module"), nanobind::arg("shots_count"),
      nanobind::arg("noise_model"), nanobind::arg("max_trajectories").none(),
      nanobind::arg("sampling_strategy").none(),
      nanobind::arg("shot_allocation").none(),
      nanobind::arg("return_execution_data"),
      nanobind::arg("include_sequential_data"), nanobind::arg("arguments"),
      R"pbdoc(
Run PTSBE sampling on the provided kernel.

Args:
  kernel_name: The kernel name.
  module: The MLIR module.
  shots_count: The number of shots.
  noise_model: The noise model.
  max_trajectories: Maximum unique trajectories, or None to use shots.
  sampling_strategy: Sampling strategy or None for default (probabilistic).
  shot_allocation: Shot allocation strategy or None for default (proportional).
  return_execution_data: Whether to include execution data in the result.
  include_sequential_data: Whether to populate per-shot sequential data.
  *arguments: The kernel arguments.

Returns:
  PTSBESampleResult with optional PTSBE execution data.
)pbdoc");

  // PTSBE async sample implementation
  ptsbe.def(
      "sample_async_impl", pySampleAsyncPTSBE, nanobind::arg("kernel_name"),
      nanobind::arg("module"), nanobind::arg("shots_count"),
      nanobind::arg("noise_model"), nanobind::arg("max_trajectories").none(),
      nanobind::arg("sampling_strategy").none(),
      nanobind::arg("shot_allocation").none(),
      nanobind::arg("return_execution_data"),
      nanobind::arg("include_sequential_data"), nanobind::arg("arguments"),
      "Run PTSBE sampling asynchronously. Returns an "
      "AsyncSampleResultImpl.");
}
