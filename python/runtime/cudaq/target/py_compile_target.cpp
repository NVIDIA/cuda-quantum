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

static std::string explicitPipelineProfileRepr(
    const cudaq::config::ExplicitPipelineProfile &profile) {
  std::ostringstream os;
  os << "ExplicitPipelineProfile(";
  if (!profile.overridePassPipeline.empty())
    os << "override_pass_pipeline=" << reprStr(profile.overridePassPipeline);
  else {
    os << "high_level_pipeline=" << reprStr(profile.highLevelPipeline)
       << ", mid_level_pipeline=" << reprStr(profile.midLevelPipeline)
       << ", low_level_pipeline=" << reprStr(profile.lowLevelPipeline)
       << ", post_code_gen_passes=" << reprStr(profile.postCodeGenPasses);
  }
  os << ")";
  return os.str();
}

static std::string compileTargetRepr(const cudaq::CompileTarget &ct) {
  std::string profileRepr =
      explicitPipelineProfileRepr(ct.getExplicitPipelineProfile());
  return "CompileTarget(stages=[" + profileRepr +
         "], codegen_translation=" + reprStr(ct.codegenTranslation) + ")";
}

void cudaq::bindCompileTarget(nanobind::module_ &mod) {
  using ExplicitPipelineProfile = cudaq::config::ExplicitPipelineProfile;

  nanobind::class_<ExplicitPipelineProfile>(
      mod, "ExplicitPipelineProfile",
      "The MLIR pass pipelines and code generation settings that a "
      "`CompileTarget` compiles kernels with.")
      .def(nanobind::init<>())
      .def_rw("override_pass_pipeline",
              &ExplicitPipelineProfile::overridePassPipeline)
      .def_rw("high_level_pipeline",
              &ExplicitPipelineProfile::highLevelPipeline)
      .def_rw("mid_level_pipeline", &ExplicitPipelineProfile::midLevelPipeline)
      .def_rw("low_level_pipeline", &ExplicitPipelineProfile::lowLevelPipeline)
      .def_rw("post_code_gen_passes",
              &ExplicitPipelineProfile::postCodeGenPasses)
      .def_rw("disable_qubit_mapping",
              &ExplicitPipelineProfile::disableQubitMapping)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<ExplicitPipelineProfile>())
      .def("__repr__", explicitPipelineProfileRepr);

  nanobind::class_<CompileTarget>(
      mod, "CompileTarget",
      "A machine model that determines how CUDA-Q compiles kernels. Includes "
      "device capabilities, supported operations, codegen format and may even "
      "specify MLIR pass pipelines explicitly.")
      .def(
          "__init__",
          [](CompileTarget *target, ExplicitPipelineProfile *pipelineConfig) {
            new (target) CompileTarget();

            if (pipelineConfig)
              target->profiles.push_back(*pipelineConfig);

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
          [](CompileTarget &target) -> ExplicitPipelineProfile & {
            if (target.profiles.empty())
              target.profiles.push_back(ExplicitPipelineProfile{});
            return target.getExplicitPipelineProfile();
          },
          [](CompileTarget &target, const ExplicitPipelineProfile &profile) {
            if (target.profiles.empty()) {
              target.profiles.push_back(profile);
              return;
            }
            auto &existing = target.getExplicitPipelineProfile();
            existing = profile;
          },
          nanobind::rv_policy::reference_internal)
      .def_rw("codegen_translation", &CompileTarget::codegenTranslation)
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
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<CompileTarget>())
      .def("__repr__", compileTargetRepr);
}
