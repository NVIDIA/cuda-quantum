/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_compile_target.h"
#include "py_runtime_target.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/platform.h"
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

using namespace nanobind::literals;

static std::string reprStr(const std::string &s) { return "'" + s + "'"; }

static std::string reprDict(const std::map<std::string, std::string> &dict) {
  std::ostringstream os;
  os << "{";
  for (const auto &[key, value] : dict) {
    os << reprStr(key) << ": " << reprStr(value) << ", ";
  }
  os << "}";
  return os.str();
}

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

static std::string
runtimeEndpointRepr(const cudaq::CompileTarget::RuntimeEndpoint &re) {
  return "RuntimeEndpoint(name=" + reprStr(re.name) +
         ", options=" + reprDict(re.options) + ")";
}

static std::string compileTargetRepr(const cudaq::CompileTarget &ct) {
  std::ostringstream os;
  os << "CompileTarget(pipeline_config="
     << pipelineConfigRepr(ct.pipelineConfig)
     << ", runtime_endpoint=" << runtimeEndpointRepr(ct.runtimeEndpoint) << ")";
  return os.str();
}

void cudaq::bindCompileTarget(nanobind::module_ &mod) {
  using PipelineConfig = cudaq::CompileTarget::PipelineConfig;
  using RuntimeEndpoint = cudaq::CompileTarget::RuntimeEndpoint;

  nanobind::class_<PipelineConfig>(mod, "PipelineConfig")
      .def(nanobind::init<>())
      .def_rw("override_pass_pipeline", &PipelineConfig::overridePassPipeline)
      .def_rw("high_level_pipeline", &PipelineConfig::highLevelPipeline)
      .def_rw("mid_level_pipeline", &PipelineConfig::midLevelPipeline)
      .def_rw("low_level_pipeline", &PipelineConfig::lowLevelPipeline)
      .def_rw("codegen_translation", &PipelineConfig::codegenTranslation)
      .def_rw("post_code_gen_passes", &PipelineConfig::postCodeGenPasses)
      .def_rw("skip_target_lowering_pipeline",
              &PipelineConfig::skipTargetLoweringPipeline)
      .def_rw("disable_qubit_mapping", &PipelineConfig::disableQubitMapping)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<PipelineConfig>())
      .def("__repr__", pipelineConfigRepr);

  nanobind::class_<RuntimeEndpoint>(mod, "RuntimeEndpoint")
      .def(
          "__init__",
          [](RuntimeEndpoint *self, const std::string &name,
             const nanobind::dict &options) {
            new (self) RuntimeEndpoint(name, parseTargetKwArgs(options));
          },
          "name"_a, "options"_a = nanobind::dict{})
      .def(nanobind::init_implicit<std::string>())
      .def_rw("name", &RuntimeEndpoint::name)
      .def_prop_rw(
          "options", [](const RuntimeEndpoint &self) { return self.options; },
          [](RuntimeEndpoint &self, const nanobind::dict &options) {
            self.options = parseTargetKwArgs(options);
          })
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<RuntimeEndpoint>())
      .def("__repr__", runtimeEndpointRepr);

  nanobind::class_<CompileTarget>(mod, "CompileTarget")
      .def(
          "__init__",
          [](CompileTarget *target, PipelineConfig *pipelineConfig,
             RuntimeEndpoint *runtimeEndpoint) {
            new (target) CompileTarget();

            if (pipelineConfig) {
              target->pipelineConfig = *pipelineConfig;
            }
            if (runtimeEndpoint) {
              target->runtimeEndpoint = *runtimeEndpoint;
            }

            // Some good defaults for Python simulators.
            // TODO: refine this and unify with CompileTarget constructor.
            target->fullySpecialize = false;
            target->isLocalSimulator = true;
            target->argumentSynthChangeSemantics = false;
            target->emitJit = true;
            if (target->pipelineConfig.codegenTranslation.empty()) {
              target->pipelineConfig.codegenTranslation = "qir:";
            }
          },
          "pipeline_config"_a = nanobind::none(),
          "runtime_endpoint"_a = nanobind::none())
      .def_rw("pipeline_config", &CompileTarget::pipelineConfig)
      .def_rw("runtime_endpoint", &CompileTarget::runtimeEndpoint)
      .def_rw("support_conditionals_on_measure_results",
              &CompileTarget::supportConditionalsOnMeasureResults)
      .def_rw("support_device_calls", &CompileTarget::supportDeviceCalls)
      .def_rw("fully_specialize", &CompileTarget::fullySpecialize)
      .def_rw("is_local_simulator", &CompileTarget::isLocalSimulator)
      .def(nanobind::self == nanobind::self)
      .def("__hash__", std::hash<CompileTarget>())
      .def("__repr__", compileTargetRepr);
}
