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
static ptsbe::sample_result
pySamplePTSBE(const std::string &shortName, MlirModule module,
              MlirType returnTy, std::size_t shots_count,
              noise_model noiseModel,
              std::optional<std::size_t> max_trajectories,
              py::object sampling_strategy, py::object shot_allocation_obj,
              bool return_execution_data, py::args runtimeArgs) {
  if (shots_count == 0)
    return ptsbe::sample_result();

  ptsbe::PTSBEOptions ptsbe_options;
  ptsbe_options.return_execution_data = return_execution_data;
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
static AsyncPTSBESampleResultImpl
pySampleAsyncPTSBE(const std::string &shortName, MlirModule module,
                   MlirType returnTy, std::size_t shots_count,
                   noise_model &noiseModel,
                   std::optional<std::size_t> max_trajectories,
                   py::object sampling_strategy, py::object shot_allocation_obj,
                   bool return_execution_data, py::args runtimeArgs) {

  ptsbe::PTSBEOptions ptsbe_options;
  ptsbe_options.return_execution_data = return_execution_data;
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

  // Safe to reset now: runSamplingAsyncPTSBE copied the noise model into the
  // async lambda.
  platform.reset_noise();
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
             "Shots proportional to trajectory weight.")
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
      .def(py::init([](ptsbe::ShotAllocationStrategy::Type t, double bias,
                       std::optional<std::uint64_t> seed) {
             return ptsbe::ShotAllocationStrategy(t, bias, seed);
           }),
           py::arg("type"), py::arg("bias_strength") = 2.0,
           py::arg("seed") = py::none(),
           "Create a strategy with specified type, optional bias strength, "
           "and optional random seed. When seed is None (default), uses "
           "CUDA-Q's global random seed.")
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
      .def(py::init<std::optional<std::uint64_t>, std::optional<std::size_t>>(),
           py::arg("seed") = py::none(),
           py::arg("max_trajectory_samples") = py::none(),
           "Create a probabilistic strategy with optional random seed and "
           "max trajectory sample count. When seed is None (default), uses "
           "CUDA-Q's global random seed. "
           "max_trajectory_samples sets a ceiling on Monte Carlo draws. "
           "The loop stops early once max_trajectories unique patterns are "
           "found. When None (default), a budget is auto-calculated.");

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

  // Trace instruction type enum
  py::enum_<ptsbe::TraceInstructionType>(
      ptsbe, "TraceInstructionType",
      "Type discriminator for trace instructions.")
      .value("Gate", ptsbe::TraceInstructionType::Gate)
      .value("Noise", ptsbe::TraceInstructionType::Noise)
      .value("Measurement", ptsbe::TraceInstructionType::Measurement)
      .export_values();

  // Trace instruction
  py::class_<ptsbe::TraceInstruction>(
      ptsbe, "TraceInstruction", "Single operation in the execution trace.")
      .def_property_readonly(
          "type", [](const ptsbe::TraceInstruction &self) { return self.type; })
      .def_property_readonly(
          "name", [](const ptsbe::TraceInstruction &self) { return self.name; })
      .def_property_readonly("targets",
                             [](const ptsbe::TraceInstruction &self) {
                               return std::vector<std::size_t>(
                                   self.targets.begin(), self.targets.end());
                             })
      .def_property_readonly("controls",
                             [](const ptsbe::TraceInstruction &self) {
                               return std::vector<std::size_t>(
                                   self.controls.begin(), self.controls.end());
                             })
      .def_property_readonly("params",
                             [](const ptsbe::TraceInstruction &self) {
                               return std::vector<double>(self.params.begin(),
                                                          self.params.end());
                             })
      .def("__repr__", [](const ptsbe::TraceInstruction &self) {
        return "TraceInstruction(" + self.name + " on " +
               std::to_string(self.targets.size()) + " qubits)";
      });

  // Kraus selection (cudaq:: namespace)
  py::class_<KrausSelection>(ptsbe, "KrausSelection",
                             "Reference to a single Kraus operator selection.")
      .def_property_readonly(
          "circuit_location",
          [](const KrausSelection &self) { return self.circuit_location; })
      .def_property_readonly(
          "kraus_operator_index",
          [](const KrausSelection &self) { return self.kraus_operator_index; })
      .def_property_readonly(
          "is_error", [](const KrausSelection &self) { return self.is_error; })
      .def_property_readonly(
          "qubits", [](const KrausSelection &self) { return self.qubits; })
      .def_property_readonly(
          "op_name", [](const KrausSelection &self) { return self.op_name; })
      .def("__repr__", [](const KrausSelection &self) {
        return "KrausSelection(loc=" + std::to_string(self.circuit_location) +
               ", idx=" + std::to_string(self.kraus_operator_index) +
               ", error=" + (self.is_error ? "true" : "false") + ")";
      });

  // Kraus trajectory (cudaq:: namespace)
  py::class_<KrausTrajectory>(
      ptsbe, "KrausTrajectory",
      "Complete specification of one noise trajectory with outcomes.")
      .def_property_readonly(
          "trajectory_id",
          [](const KrausTrajectory &self) { return self.trajectory_id; })
      .def_property_readonly(
          "probability",
          [](const KrausTrajectory &self) { return self.probability; })
      .def_property_readonly(
          "num_shots",
          [](const KrausTrajectory &self) { return self.num_shots; })
      .def_readonly("multiplicity", &KrausTrajectory::multiplicity,
                    "Number of times this trajectory was sampled.")
      .def_readonly("weight", &KrausTrajectory::weight,
                    "Allocation weight for shot distribution.")
      .def_property_readonly(
          "kraus_selections",
          [](const KrausTrajectory &self) { return self.kraus_selections; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "measurement_counts",
          [](const KrausTrajectory &self) { return self.measurement_counts; })
      .def("__repr__", [](const KrausTrajectory &self) {
        return "KrausTrajectory(id=" + std::to_string(self.trajectory_id) +
               ", p=" + std::to_string(self.probability) +
               ", shots=" + std::to_string(self.num_shots) + ")";
      });

  // PTSBE execution data container
  py::class_<ptsbe::PTSBEExecutionData>(
      ptsbe, "PTSBEExecutionData",
      "Container for PTSBE execution data including circuit structure, "
      "trajectory specifications, and per-trajectory measurement outcomes.")
      .def_property_readonly(
          "instructions",
          [](const ptsbe::PTSBEExecutionData &self)
              -> const std::vector<ptsbe::TraceInstruction> & {
            return self.instructions;
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "trajectories",
          [](const ptsbe::PTSBEExecutionData &self)
              -> const std::vector<cudaq::KrausTrajectory> & {
            return self.trajectories;
          },
          py::return_value_policy::reference_internal)
      .def(
          "count_instructions",
          [](const ptsbe::PTSBEExecutionData &self,
             ptsbe::TraceInstructionType type, py::object name) -> std::size_t {
            std::optional<std::string> nameOpt;
            if (!name.is_none())
              nameOpt = name.cast<std::string>();
            return self.count_instructions(type, nameOpt);
          },
          py::arg("type"), py::arg("name") = py::none(),
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
          py::return_value_policy::reference_internal, py::arg("trajectory_id"),
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
  py::class_<ptsbe::sample_result, sample_result>(
      ptsbe, "SampleResult",
      "PTSBE sample result with optional execution data.")
      .def_property_readonly(
          "ptsbe_execution_data",
          [](const ptsbe::sample_result &self)
              -> const ptsbe::PTSBEExecutionData * {
            if (!self.has_execution_data())
              return nullptr;
            return &self.execution_data();
          },
          py::return_value_policy::reference_internal,
          "PTSBE execution data if return_execution_data was True, None "
          "otherwise.")
      .def("has_execution_data", &ptsbe::sample_result::has_execution_data,
           "Check if execution data is available.");

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
  noise_model: Optional noise model for gate-based noise; may be None.
  max_trajectories: Maximum unique trajectories, or None to use shots.
  sampling_strategy: Sampling strategy or None for default (probabilistic).
  shot_allocation: Shot allocation strategy or None for default (proportional).
  return_execution_data: Whether to include execution data in the result.
  *arguments: The kernel arguments.

Returns:
  SampleResult with optional PTSBE execution data.
)pbdoc");

  // PTSBE async sample implementation
  ptsbe.def("sample_async_impl", pySampleAsyncPTSBE,
            "Run PTSBE sampling asynchronously. Returns an "
            "AsyncSampleResultImpl.");
}
