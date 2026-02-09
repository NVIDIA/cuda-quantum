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
#include "cudaq/ptsbe/PTSBEOptions.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESampleResult.h"
#include "cudaq/ptsbe/PTSBETrace.h"
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
static ptsbe::sample_result pySamplePTSBE(
    const std::string &shortName, MlirModule module, MlirType returnTy,
    std::size_t shots_count, noise_model noiseModel,
    std::optional<std::size_t> max_trajectories, py::object sampling_strategy,
    bool return_trace, py::args runtimeArgs) {
  if (shots_count == 0)
    return ptsbe::sample_result();

  ptsbe::PTSBEOptions ptsbe_options;
  ptsbe_options.return_trace = return_trace;
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
      .def_property_readonly("kraus_operator_index",
                             [](const KrausSelection &self) {
                               return static_cast<std::size_t>(
                                   self.kraus_operator_index);
                             })
      .def("__repr__", [](const KrausSelection &self) {
        return "KrausSelection(loc=" + std::to_string(self.circuit_location) +
               ", idx=" +
               std::to_string(
                   static_cast<std::size_t>(self.kraus_operator_index)) +
               ")";
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
      .def_property_readonly(
          "kraus_selections",
          [](const KrausTrajectory &self) { return self.kraus_selections; },
          py::return_value_policy::reference_internal)
      .def_property_readonly("measurement_counts",
                             [](const KrausTrajectory &self) -> py::object {
                               if (!self.measurement_counts.has_value())
                                 return py::none();
                               return py::cast(self.measurement_counts.value());
                             })
      .def("__repr__", [](const KrausTrajectory &self) {
        return "KrausTrajectory(id=" + std::to_string(self.trajectory_id) +
               ", p=" + std::to_string(self.probability) +
               ", shots=" + std::to_string(self.num_shots) + ")";
      });

  // PTSBE trace container
  py::class_<ptsbe::PTSBETrace>(
      ptsbe, "PTSBETrace",
      "Container for PTSBE trace data including circuit and trajectories.")
      .def_property_readonly(
          "instructions",
          [](const ptsbe::PTSBETrace &self) { return self.instructions; },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "trajectories",
          [](const ptsbe::PTSBETrace &self) { return self.trajectories; },
          py::return_value_policy::reference_internal)
      .def(
          "count_instructions",
          [](const ptsbe::PTSBETrace &self, ptsbe::TraceInstructionType type,
             py::object name) -> std::size_t {
            std::optional<std::string> nameOpt;
            if (!name.is_none())
              nameOpt = name.cast<std::string>();
            return self.count_instructions(type, nameOpt);
          },
          py::arg("type"), py::arg("name") = py::none(),
          "Count instructions of a given type.")
      .def(
          "get_trajectory",
          [](const ptsbe::PTSBETrace &self,
             std::size_t trajectory_id) -> const cudaq::KrausTrajectory * {
            auto result = self.get_trajectory(trajectory_id);
            if (!result.has_value())
              return nullptr;
            return &result.value().get();
          },
          py::return_value_policy::reference_internal, py::arg("trajectory_id"),
          "Look up a trajectory by its ID. Returns None if not found.")
      .def("__repr__",
           [](const ptsbe::PTSBETrace &self) {
             return "PTSBETrace(" + std::to_string(self.instructions.size()) +
                    " instructions, " +
                    std::to_string(self.trajectories.size()) + " trajectories)";
           })
      .def("__len__", [](const ptsbe::PTSBETrace &self) {
        return self.instructions.size();
      });

  // PTSBE sample result (subclass of sample_result)
  py::class_<ptsbe::sample_result, sample_result>(
      ptsbe, "SampleResult", "PTSBE sample result with optional trace data.")
      .def_property_readonly(
          "ptsbe_trace",
          [](const ptsbe::sample_result &self) -> const ptsbe::PTSBETrace * {
            if (!self.has_trace())
              return nullptr;
            return &self.trace();
          },
          py::return_value_policy::reference_internal,
          "PTSBE trace data if return_trace was True, None otherwise.")
      .def("has_trace", &ptsbe::sample_result::has_trace,
           "Check if trace data is available.");

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
  return_trace: Whether to include trace and trajectory data in the result.
  *arguments: The kernel arguments.

Returns:
  SampleResult with optional PTSBE trace data.
)pbdoc");
}
