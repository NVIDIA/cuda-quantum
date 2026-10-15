/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_compile_target.h"
#include "cudaq/Target/CompileTarget.h"
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>

using namespace nanobind::literals;

static std::string reprStr(const std::string &s) { return "'" + s + "'"; }

static std::string customPipelineLoweringRepr(
    const cudaq::config::CustomPipelineLowering &lowering) {
  std::ostringstream os;
  os << "CustomPipelineLowering(";
  if (!lowering.overridePassPipeline.empty())
    os << "override_pass_pipeline=" << reprStr(lowering.overridePassPipeline);
  else {
    os << "high_level_pipeline=" << reprStr(lowering.highLevelPipeline)
       << ", mid_level_pipeline=" << reprStr(lowering.midLevelPipeline)
       << ", low_level_pipeline=" << reprStr(lowering.lowLevelPipeline);
  }
  os << ")";
  return os.str();
}

static std::string compileTargetRepr(const cudaq::CompileTarget &ct) {
  std::string loweringRepr = "[]";
  if (!ct.stages.empty())
    loweringRepr =
        "[" + customPipelineLoweringRepr(ct.getCustomPipelineLowering()) + "]";
  return "CompileTarget(stages=" + loweringRepr +
         ", codegen_translation=" + reprStr(ct.codegenTranslation) + ")";
}

void cudaq::bindCompileTarget(nanobind::module_ &mod) {
  using CustomPipelineLowering = cudaq::config::CustomPipelineLowering;

  nanobind::class_<CustomPipelineLowering>(
      mod, "CustomPipelineLowering",
      "The MLIR pass pipelines that a `CompileTarget` compiles kernels with.")
      .def(nanobind::init<>())
      .def_rw("override_pass_pipeline",
              &CustomPipelineLowering::overridePassPipeline)
      .def_rw("high_level_pipeline", &CustomPipelineLowering::highLevelPipeline)
      .def_rw("mid_level_pipeline", &CustomPipelineLowering::midLevelPipeline)
      .def_rw("low_level_pipeline", &CustomPipelineLowering::lowLevelPipeline)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<CustomPipelineLowering>())
      .def("__repr__", customPipelineLoweringRepr);

  nanobind::class_<CompileTarget>(
      mod, "CompileTarget",
      "A machine model that determines how CUDA-Q compiles kernels. Includes "
      "device capabilities, supported operations, codegen format and may even "
      "specify MLIR pass pipelines explicitly.")
      .def(
          "__init__",
          [](CompileTarget *target, CustomPipelineLowering *pipelineConfig) {
            new (target) CompileTarget();

            if (pipelineConfig)
              target->stages.push_back(cudaq::config::CompilationStage{
                  .irOutputProfile = cudaq::config::UncheckedProfile{},
                  .lowering = *pipelineConfig});

            // Some good defaults for Python simulators.
            // TODO: refine this and unify with `createDefaultCompileTarget`.
            target->fullySpecialize = false;
            target->argumentSynthChangeSemantics = false;
            if (target->codegenTranslation.empty())
              target->codegenTranslation = "qir:";
          },
          "pipeline_config"_a = nanobind::none())
      .def_prop_rw(
          "pipeline_config",
          [](CompileTarget &target) -> CustomPipelineLowering & {
            if (target.stages.empty())
              target.stages.push_back(cudaq::config::CompilationStage{
                  .irOutputProfile = cudaq::config::UncheckedProfile{},
                  .lowering = CustomPipelineLowering{}});
            return target.getCustomPipelineLowering();
          },
          [](CompileTarget &target, const CustomPipelineLowering &lowering) {
            if (target.stages.empty()) {
              target.stages.push_back(cudaq::config::CompilationStage{
                  .irOutputProfile = cudaq::config::UncheckedProfile{},
                  .lowering = lowering});
              return;
            }
            auto &existing = target.getCustomPipelineLowering();
            existing = lowering;
          },
          nanobind::rv_policy::reference_internal)
      .def_rw("codegen_translation", &CompileTarget::codegenTranslation)
      .def_rw("post_code_gen_passes", &CompileTarget::postCodeGenPasses)
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
