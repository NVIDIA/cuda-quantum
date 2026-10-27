/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_compile_target.h"
#include "cudaq/Target/CompileTarget.h"
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <utility>
#include <variant>

namespace nb = nanobind;
using namespace nb::literals;
using namespace cudaq::config;

namespace {
void initCompileTarget(cudaq::CompileTarget *self,
                       std::vector<CompilationStage> stages, Codegen codegen) {
  new (self) cudaq::CompileTarget{};
  self->stages = std::move(stages);
  self->codegen = std::move(codegen);
  self->fullySpecialize = false;
  self->argumentSynthChangeSemantics = false;
}
} // namespace

void cudaq::bindCompileTarget(nb::module_ &m) {
  nb::class_<cudaq::CompileTarget>(m, "CompileTarget")
      .def("__init__", &initCompileTarget,
           "stages"_a = std::vector<CompilationStage>{}, nb::kw_only(),
           "codegen"_a = Codegen{"qir:", ""})
      .def(
          "__init__",
          [](cudaq::CompileTarget *self, Profile profile, Codegen codegen) {
            initCompileTarget(
                self,
                {CompilationStage{std::move(profile), InferredLowering{}}},
                std::move(codegen));
          },
          nb::kw_only(), "ir_output_profile"_a,
          "codegen"_a = Codegen{"qir:", ""})
      .def(
          "__init__",
          [](cudaq::CompileTarget *self,
             std::variant<CustomPipelineLowering, std::string> pipeline,
             Codegen codegen) {
            auto *str = std::get_if<std::string>(&pipeline);
            auto lowering =
                str ? CustomPipelineLowering{std::move(*str)}
                    : std::get<CustomPipelineLowering>(std::move(pipeline));
            initCompileTarget(
                self,
                {CompilationStage{UncheckedProfile{}, std::move(lowering)}},
                std::move(codegen));
          },
          nb::kw_only(), "custom_pipeline"_a, "codegen"_a = Codegen{"qir:", ""})
      .def_prop_rw(
          "stages", [](const cudaq::CompileTarget &t) { return t.stages; },
          [](cudaq::CompileTarget &t, std::vector<CompilationStage> stages) {
            t.stages = std::move(stages);
          })
      .def_rw("codegen", &cudaq::CompileTarget::codegen,
              nb::rv_policy::reference_internal)
      .def_rw("override_aot_compilation",
              &cudaq::CompileTarget::overrideAOTCompilation)
      .def_rw("support_conditionals_on_measure_results",
              &cudaq::CompileTarget::supportConditionalsOnMeasureResults)
      .def_rw("support_device_calls", &cudaq::CompileTarget::supportDeviceCalls)
      .def_rw("support_resource_counts",
              &cudaq::CompileTarget::supportResourceCounts)
      .def_rw("support_explicit_measurements",
              &cudaq::CompileTarget::supportExplicitMeasurements)
      .def_rw("support_observable_measurements",
              &cudaq::CompileTarget::supportObservableMeasurements)
      .def_rw("support_sample_without_measurements",
              &cudaq::CompileTarget::supportSampleWithoutMeasurements)
      .def_rw("fully_specialize", &cudaq::CompileTarget::fullySpecialize)
      .def_rw("argument_synth_change_semantics",
              &cudaq::CompileTarget::argumentSynthChangeSemantics)
      .def_rw("disable_qubit_mapping",
              &cudaq::CompileTarget::disableQubitMapping)
      .def_rw("replace_state_with_kernel",
              &cudaq::CompileTarget::replaceStateWithKernel)
      .def(nb::self == nb::self)
      .def("__hash__", std::hash<cudaq::CompileTarget>{})
      .def("__repr__", [](const cudaq::CompileTarget &t) {
        return "CompileTarget(stages=" +
               std::string(nb::repr(nb::cast(t.stages)).c_str()) +
               ", codegen=" +
               std::string(nb::repr(nb::cast(t.codegen)).c_str()) + ")";
      });
}
