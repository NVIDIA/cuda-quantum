/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_compile_target.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/runtime/logger/logger.h"
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <sstream>

using namespace nanobind::literals;

static std::string reprStr(const std::string &s) { return "'" + s + "'"; }

static std::string customPipelineLoweringRepr(
    const cudaq::config::CustomPipelineLowering &lowering) {
  std::ostringstream os;
  os << "CustomPipelineLowering(";
  if (!lowering.overridePassPipeline.empty())
    os << "pipeline=" << reprStr(lowering.overridePassPipeline);
  else {
    bool first = true;
    auto keyword = [&](const char *name, const std::string &value) {
      if (value.empty())
        return;
      os << (first ? "" : ", ") << name << "=" << reprStr(value);
      first = false;
    };
    keyword("high_level", lowering.highLevelPipeline);
    keyword("mid_level", lowering.midLevelPipeline);
    keyword("low_level", lowering.lowLevelPipeline);
  }
  os << ")";
  return os.str();
}

static std::string codegenRepr(const cudaq::config::Codegen &codegen) {
  std::ostringstream os;
  os << "Codegen(" << reprStr(codegen.emissionFormat);
  if (!codegen.postCodegenPasses.empty())
    os << ", post_codegen_passes=" << reprStr(codegen.postCodegenPasses);
  os << ")";
  return os.str();
}

static void reportUnsupportedStage() {
  CUDAQ_ERROR("Unsupported compilation stage: only a single stage with a "
              "custom-pipeline lowering is currently supported\n");
}

/// Access the custom pipeline lowering of a compile target with a single
/// compilation stage.
// TODO: Remove this once the Python bindings have been updated to handle
// multiple stages.
static const cudaq::config::CustomPipelineLowering &
getCustomPipelineLowering(const cudaq::CompileTarget &target) {
  if (target.stages.size() != 1)
    reportUnsupportedStage();
  const auto *lowering = std::get_if<cudaq::config::CustomPipelineLowering>(
      &target.stages[0].lowering);
  if (!lowering)
    reportUnsupportedStage();
  return *lowering;
}
static cudaq::config::CustomPipelineLowering &
getCustomPipelineLowering(cudaq::CompileTarget &target) {
  if (target.stages.size() != 1)
    reportUnsupportedStage();
  auto *lowering = std::get_if<cudaq::config::CustomPipelineLowering>(
      &target.stages[0].lowering);
  if (!lowering)
    reportUnsupportedStage();
  return *lowering;
}

static std::string compileTargetRepr(const cudaq::CompileTarget &ct) {
  std::ostringstream os;
  os << "CompileTarget(lowering=";
  if (ct.stages.empty())
    os << "None";
  else
    os << customPipelineLoweringRepr(getCustomPipelineLowering(ct));
  os << ", codegen=" << codegenRepr(ct.codegen) << ")";
  return os.str();
}

void cudaq::bindCompileTarget(nanobind::module_ &mod) {
  using Codegen = cudaq::config::Codegen;
  using CustomPipelineLowering = cudaq::config::CustomPipelineLowering;

  nanobind::class_<Codegen>(
      mod, "Codegen",
      "Code generation selected by a `CompileTarget`: the emission format "
      "(e.g. 'qir:', 'qir-base', 'qasm2') and an optional MLIR pass pipeline "
      "to run after code generation.")
      .def(nanobind::init<>())
      .def(
          "__init__",
          [](Codegen *self, const std::string &emissionFormat,
             const std::string &postCodegenPasses) {
            new (self) Codegen{.emissionFormat = emissionFormat,
                               .postCodegenPasses = postCodegenPasses};
          },
          "emission_format"_a = "", "post_codegen_passes"_a = "")
      .def_rw("emission_format", &Codegen::emissionFormat)
      .def_rw("post_codegen_passes", &Codegen::postCodegenPasses)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<Codegen>())
      .def("__repr__", codegenRepr);

  // A plain emission-format string implicitly converts to a `Codegen`, like
  // `codegen: qir-base` in the target configuration schema.
  nanobind::implicitly_convertible<std::string, Codegen>();

  nanobind::class_<CustomPipelineLowering>(
      mod, "CustomPipelineLowering",
      "A custom MLIR pass pipeline for a compilation stage. Either a single "
      "pipeline string overriding the whole compilation pipeline, or "
      "`high_level`/`mid_level`/`low_level` pipelines inserted around the "
      "built-in compilation stages.")
      .def(
          "__init__",
          [](CustomPipelineLowering *self, const std::string &pipeline,
             const std::string &highLevel, const std::string &midLevel,
             const std::string &lowLevel) {
            bool hasStaged =
                !highLevel.empty() || !midLevel.empty() || !lowLevel.empty();
            if (!pipeline.empty() && hasStaged)
              throw std::invalid_argument(
                  "provide either an override pipeline or staged pipelines, "
                  "not both");
            new (self) CustomPipelineLowering{.overridePassPipeline = pipeline,
                                              .highLevelPipeline = highLevel,
                                              .midLevelPipeline = midLevel,
                                              .lowLevelPipeline = lowLevel};
          },
          "pipeline"_a = "", "high_level"_a = "", "mid_level"_a = "",
          "low_level"_a = "")
      .def_rw("override_pass_pipeline",
              &CustomPipelineLowering::overridePassPipeline)
      .def_rw("high_level", &CustomPipelineLowering::highLevelPipeline)
      .def_rw("mid_level", &CustomPipelineLowering::midLevelPipeline)
      .def_rw("low_level", &CustomPipelineLowering::lowLevelPipeline)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<CustomPipelineLowering>())
      .def("__repr__", customPipelineLoweringRepr);

  // A plain pipeline string implicitly converts to an override
  // `CustomPipelineLowering`, like a plain string `custom-pipeline` in the
  // target configuration schema.
  nanobind::implicitly_convertible<std::string, CustomPipelineLowering>();

  nanobind::class_<CompileTarget>(
      mod, "CompileTarget",
      "A machine model that determines how CUDA-Q compiles kernels. Includes "
      "device capabilities, supported operations, codegen format and may even "
      "specify MLIR pass pipelines explicitly.")
      .def(
          "__init__",
          [](CompileTarget *target,
             std::optional<CustomPipelineLowering> lowering,
             const Codegen &codegen) {
            new (target) CompileTarget();
            if (lowering)
              target->stages.push_back(cudaq::config::CompilationStage{
                  .irOutputProfile = cudaq::config::UncheckedProfile{},
                  .lowering = std::move(*lowering)});
            target->codegen = codegen;

            // Some good defaults for Python simulators.
            // TODO: refine this and unify with `createDefaultCompileTarget`.
            target->fullySpecialize = false;
            target->argumentSynthChangeSemantics = false;
          },
          "lowering"_a = nanobind::none(), "codegen"_a = Codegen{"qir:"})
      .def_prop_rw(
          "lowering",
          [](CompileTarget &target) -> CustomPipelineLowering * {
            if (target.stages.empty())
              return nullptr;
            return &getCustomPipelineLowering(target);
          },
          [](CompileTarget &target,
             std::optional<CustomPipelineLowering> lowering) {
            if (!lowering) {
              target.stages.clear();
              return;
            }
            if (target.stages.empty()) {
              target.stages.push_back(cudaq::config::CompilationStage{
                  .irOutputProfile = cudaq::config::UncheckedProfile{},
                  .lowering = std::move(*lowering)});
              return;
            }
            auto &target_lowering = getCustomPipelineLowering(target);
            target_lowering = std::move(*lowering);
          },
          nanobind::rv_policy::reference_internal)
      .def_rw("codegen", &CompileTarget::codegen,
              nanobind::rv_policy::reference_internal)
      .def_rw("support_conditionals_on_measure_results",
              &CompileTarget::supportConditionalsOnMeasureResults)
      .def_rw("support_device_calls", &CompileTarget::supportDeviceCalls)
      .def_rw("support_resource_counts", &CompileTarget::supportResourceCounts)
      .def_rw("support_explicit_measurements",
              &CompileTarget::supportExplicitMeasurements)
      .def_rw("support_observable_measurements",
              &CompileTarget::supportObservableMeasurements)
      .def_rw("support_sample_without_measurements",
              &CompileTarget::supportSampleWithoutMeasurements)
      .def_rw("fully_specialize", &CompileTarget::fullySpecialize)
      .def_rw("disable_qubit_mapping", &CompileTarget::disableQubitMapping)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<CompileTarget>())
      .def("__repr__", compileTargetRepr);
}
