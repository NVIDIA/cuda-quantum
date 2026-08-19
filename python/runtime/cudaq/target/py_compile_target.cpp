/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_compile_target.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/platform.h"
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <sstream>

using namespace nanobind::literals;

static std::string reprStr(const std::string &s) { return "'" + s + "'"; }

static std::string
pipelineConfigRepr(const cudaq::CompileTarget::PipelineConfig &pc) {
  std::ostringstream os;
  os << "PipelineConfig(";
  if (!pc.overridePassPipeline.empty())
    os << "override_pass_pipeline=" << reprStr(pc.overridePassPipeline);
  else {
    os << "high_level_pipeline=" << reprStr(pc.highLevelPipeline)
       << ", mid_level_pipeline=" << reprStr(pc.midLevelPipeline)
       << ", low_level_pipeline=" << reprStr(pc.lowLevelPipeline)
       << ", codegen_translation=" << reprStr(pc.codegenTranslation)
       << ", post_code_gen_passes=" << reprStr(pc.postCodeGenPasses);
  }
  os << ")";
  return os.str();
}

static std::string compileTargetRepr(const cudaq::CompileTarget &ct) {
  return "CompileTarget(pipeline_config=" +
         pipelineConfigRepr(ct.pipelineConfig) + ")";
}

void cudaq::bindCompileTarget(nanobind::module_ &mod) {
  using PipelineConfig = cudaq::CompileTarget::PipelineConfig;

  nanobind::class_<PipelineConfig>(
      mod, "PipelineConfig",
      "The MLIR pass pipelines and code generation settings that a "
      "`CompileTarget` compiles kernels with.")
      .def(nanobind::init<>())
      .def_rw("override_pass_pipeline", &PipelineConfig::overridePassPipeline)
      .def_rw("high_level_pipeline", &PipelineConfig::highLevelPipeline)
      .def_rw("mid_level_pipeline", &PipelineConfig::midLevelPipeline)
      .def_rw("low_level_pipeline", &PipelineConfig::lowLevelPipeline)
      .def_rw("codegen_translation", &PipelineConfig::codegenTranslation)
      .def_rw("post_code_gen_passes", &PipelineConfig::postCodeGenPasses)
      .def_rw("disable_qubit_mapping", &PipelineConfig::disableQubitMapping)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<PipelineConfig>())
      .def("__repr__", pipelineConfigRepr);

  nanobind::class_<CompileTarget>(
      mod, "CompileTarget",
      "A machine model that determines how CUDA-Q compiles kernels. Includes "
      "device capabilities, supported operations, codegen format and may even "
      "specify MLIR pass pipelines explicitly.")
      .def(
          "__init__",
          [](CompileTarget *target, PipelineConfig *pipelineConfig) {
            new (target) CompileTarget();

            if (pipelineConfig) {
              target->pipelineConfig = *pipelineConfig;
            }

            // Some good defaults for Python simulators.
            // TODO: refine this and unify with `createDefaultCompileTarget`.
            target->fullySpecialize = false;
            target->isLocalSimulator = true;
            target->argumentSynthChangeSemantics = false;
            if (target->pipelineConfig.codegenTranslation.empty()) {
              target->pipelineConfig.codegenTranslation = "qir:";
            }
          },
          "pipeline_config"_a = nanobind::none())
      .def_rw("pipeline_config", &CompileTarget::pipelineConfig)
      .def_rw("support_conditionals_on_measure_results",
              &CompileTarget::supportConditionalsOnMeasureResults)
      .def_rw("support_device_calls", &CompileTarget::supportDeviceCalls)
      .def_rw("fully_specialize", &CompileTarget::fullySpecialize)
      .def_rw("is_local_simulator", &CompileTarget::isLocalSimulator)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<CompileTarget>())
      .def("__repr__", compileTargetRepr);

  mod.def(
      "set_compile_target",
      [](CompileTarget target) {
        get_platform().setCompileTarget(std::move(target));
      },
      nanobind::arg("target"),
      "Compile kernels with the given `CompileTarget` instead of the one the "
      "active target's QPU provides.");
}
