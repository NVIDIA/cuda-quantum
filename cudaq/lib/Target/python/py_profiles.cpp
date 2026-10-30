/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_profiles.h"
#include "cudaq/Target/CompileTarget.h"
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace cudaq::config;

namespace {
MeasurementForm measurementForm(const std::string &value) {
  if (value == "any")
    return MeasurementForm::any;
  if (value == "single_qubit")
    return MeasurementForm::single_qubit;
  throw std::invalid_argument(
      "measurement_type must be 'any' or 'single_qubit'");
}

std::string measurementFormName(MeasurementForm form) {
  return form == MeasurementForm::any ? "any" : "single_qubit";
}

/// Accepts a `CostModel`, its name as a string such as `"t_count"`, or None.
std::optional<CostModel> costModel(nb::handle value) {
  if (value.is_none())
    return std::nullopt;
  if (!nb::isinstance<nb::str>(value))
    return nb::cast<CostModel>(value);
  try {
    return nb::cast<CostModel>(nb::type<CostModel>().attr(value));
  } catch (const std::exception &) {
    throw std::invalid_argument(
        "cost_model must be a CostModel, the name of one, or None");
  }
}

Semantics semantics(const std::string &value) {
  if (value == "any")
    return Semantics::any;
  if (value == "value")
    return Semantics::value;
  throw std::invalid_argument("semantics must be 'any' or 'value'");
}

std::string semanticsName(Semantics semantics) {
  switch (semantics) {
  case Semantics::any:
    return "any";
  case Semantics::value:
    return "value";
  }
  throw std::invalid_argument("unknown semantics");
}

LoopUnrolling loopUnrolling(const std::string &value) {
  if (value == "all")
    return LoopUnrolling::all;
  if (value == "index_use")
    return LoopUnrolling::index_use;
  throw std::invalid_argument("unroll_loops must be 'all' or 'index_use'");
}

std::string loopUnrollingName(LoopUnrolling unrolling) {
  switch (unrolling) {
  case LoopUnrolling::all:
    return "all";
  case LoopUnrolling::index_use:
    return "index_use";
  }
  throw std::invalid_argument("unknown unroll_loops value");
}

template <typename T>
std::string repr(const T &value) {
  return nb::repr(nb::cast(value)).c_str();
}

/// The cost model as Python reads it back, such as `CostModel.t_count`.
std::string costModelRepr(const std::optional<CostModel> &costModel) {
  if (!costModel)
    return "None";
  return "CostModel." +
         std::string(nb::str(nb::cast(*costModel).attr("name")).c_str());
}

template <typename T>
nb::class_<T> valueClass(nb::module_ &m, const char *name) {
  return nb::class_<T>(m, name)
      .def(nb::self == nb::self)
      .def("__hash__", std::hash<T>{});
}
} // namespace

void cudaq::bindTargetProfiles(nb::module_ &m) {
  valueClass<UncheckedProfile>(m, "UncheckedProfile")
      .def(nb::init<>())
      .def("__repr__",
           [](const UncheckedProfile &) { return "UncheckedProfile()"; });
  valueClass<QuakeReferenceSemanticsProfile>(m,
                                             "QuakeReferenceSemanticsProfile")
      .def(
          "__init__",
          [](QuakeReferenceSemanticsProfile *self,
             std::vector<std::string> basis, const std::string &measurementType,
             bool reset) {
            auto form = measurementForm(measurementType);
            new (self)
                QuakeReferenceSemanticsProfile{std::move(basis), form, reset};
          },
          nb::kw_only(), "basis_gates"_a = std::vector<std::string>{},
          "measurement_type"_a = "any", "reset_after_measure"_a = false)
      .def_rw("basis_gates", &QuakeReferenceSemanticsProfile::basisGates)
      .def_prop_rw(
          "measurement_type",
          [](const QuakeReferenceSemanticsProfile &p) {
            return measurementFormName(p.measurementType);
          },
          [](QuakeReferenceSemanticsProfile &p, const std::string &value) {
            p.measurementType = measurementForm(value);
          })
      .def_rw("reset_after_measure",
              &QuakeReferenceSemanticsProfile::resetAfterMeasure)
      .def("__repr__", [](const QuakeReferenceSemanticsProfile &p) {
        return "QuakeReferenceSemanticsProfile(basis_gates=" +
               repr(p.basisGates) + ", measurement_type=" +
               repr(measurementFormName(p.measurementType)) +
               ", reset_after_measure=" +
               (p.resetAfterMeasure ? "True" : "False") + ")";
      });
  nb::enum_<CostModel>(m, "CostModel")
      .value("t_count", CostModel::t_count)
      .value("two_qubit_gate_count", CostModel::two_qubit_gate_count);
  valueClass<QuakeUnmappedProfile>(m, "QuakeUnmappedProfile")
      .def(
          "__init__",
          [](QuakeUnmappedProfile *self, std::vector<std::string> basis,
             double tolerance, nb::handle model,
             const std::string &semanticsValue, const std::string &unrollLoops,
             unsigned maxUnrollIterations) {
            new (self) QuakeUnmappedProfile{
                .basisGates = std::move(basis),
                .approximationTolerance = tolerance,
                .costModel = costModel(model),
                .semantics = semantics(semanticsValue),
                .unrollLoops = loopUnrolling(unrollLoops),
                .maxUnrollIterations = maxUnrollIterations};
          },
          nb::kw_only(), "basis_gates"_a = std::vector<std::string>{},
          "approximation_tolerance"_a = 0.0, "cost_model"_a = nb::none(),
          "semantics"_a = "any", "unroll_loops"_a = "all",
          "max_unroll_iterations"_a =
              QuakeUnmappedProfile{}.maxUnrollIterations)
      .def_rw("basis_gates", &QuakeUnmappedProfile::basisGates)
      .def_rw("approximation_tolerance",
              &QuakeUnmappedProfile::approximationTolerance)
      .def_prop_rw(
          "cost_model",
          [](const QuakeUnmappedProfile &p) { return p.costModel; },
          [](QuakeUnmappedProfile &p, nb::handle model) {
            p.costModel = costModel(model);
          },
          nb::for_setter(nb::arg("value").none()))
      .def_prop_rw(
          "semantics",
          [](const QuakeUnmappedProfile &p) {
            return semanticsName(p.semantics);
          },
          [](QuakeUnmappedProfile &p, const std::string &value) {
            p.semantics = semantics(value);
          })
      .def_prop_rw(
          "unroll_loops",
          [](const QuakeUnmappedProfile &p) {
            return loopUnrollingName(p.unrollLoops);
          },
          [](QuakeUnmappedProfile &p, const std::string &value) {
            p.unrollLoops = loopUnrolling(value);
          })
      .def_rw("max_unroll_iterations",
              &QuakeUnmappedProfile::maxUnrollIterations)
      .def("__repr__", [](const QuakeUnmappedProfile &p) {
        return "QuakeUnmappedProfile(basis_gates=" + repr(p.basisGates) +
               ", approximation_tolerance=" + repr(p.approximationTolerance) +
               ", cost_model=" + costModelRepr(p.costModel) +
               ", semantics=" + repr(semanticsName(p.semantics)) +
               ", unroll_loops=" + repr(loopUnrollingName(p.unrollLoops)) +
               ", max_unroll_iterations=" + repr(p.maxUnrollIterations) + ")";
      });
  valueClass<QuakeMappedProfile>(m, "QuakeMappedProfile")
      .def(
          "__init__",
          [](QuakeMappedProfile *self, std::vector<std::string> basis,
             std::optional<std::string> device) {
            new (self) QuakeMappedProfile{std::move(basis), std::move(device)};
          },
          nb::kw_only(), "basis_gates"_a = std::vector<std::string>{},
          "device"_a = nb::none())
      .def_rw("basis_gates", &QuakeMappedProfile::basisGates)
      .def_rw("device", &QuakeMappedProfile::device,
              nb::for_setter(nb::arg("value").none()))
      .def("__repr__", [](const QuakeMappedProfile &p) {
        return "QuakeMappedProfile(basis_gates=" + repr(p.basisGates) +
               ", device=" + repr(p.device) + ")";
      });
  valueClass<CustomPipelineLowering>(m, "CustomPipelineLowering")
      .def(
          "__init__",
          [](CustomPipelineLowering *self, std::string pipeline,
             std::string high, std::string mid, std::string low) {
            if (!pipeline.empty() &&
                (!high.empty() || !mid.empty() || !low.empty()))
              throw std::invalid_argument(
                  "pipeline cannot be combined with fragments");
            new (self)
                CustomPipelineLowering{std::move(pipeline), std::move(high),
                                       std::move(mid), std::move(low)};
          },
          "pipeline"_a = "", nb::kw_only(), "high_level"_a = "",
          "mid_level"_a = "", "low_level"_a = "")
      .def_rw("override_pass_pipeline",
              &CustomPipelineLowering::overridePassPipeline)
      .def_rw("high_level", &CustomPipelineLowering::highLevelPipeline)
      .def_rw("mid_level", &CustomPipelineLowering::midLevelPipeline)
      .def_rw("low_level", &CustomPipelineLowering::lowLevelPipeline)
      .def("__repr__", [](const CustomPipelineLowering &p) {
        return "CustomPipelineLowering(pipeline=" +
               repr(p.overridePassPipeline) +
               ", high_level=" + repr(p.highLevelPipeline) +
               ", mid_level=" + repr(p.midLevelPipeline) +
               ", low_level=" + repr(p.lowLevelPipeline) + ")";
      });
  valueClass<Codegen>(m, "Codegen")
      .def(
          "__init__",
          [](Codegen *self, std::string format, std::string passes) {
            new (self) Codegen{std::move(format), std::move(passes)};
          },
          "emission_format"_a = "", "post_codegen_passes"_a = "")
      .def_rw("emission_format", &Codegen::emissionFormat)
      .def_rw("post_codegen_passes", &Codegen::postCodegenPasses)
      .def("__repr__", [](const Codegen &c) {
        return "Codegen(" + repr(c.emissionFormat) +
               ", post_codegen_passes=" + repr(c.postCodegenPasses) + ")";
      });
  nb::implicitly_convertible<std::string, Codegen>();
  valueClass<CompilationStage>(m, "CompilationStage")
      .def(
          "__init__",
          [](CompilationStage *self, Profile profile,
             std::optional<CustomPipelineLowering> lowering) {
            new (self) CompilationStage{
                std::move(profile), lowering ? Lowering{std::move(*lowering)}
                                             : Lowering{InferredLowering{}}};
          },
          "ir_output_profile"_a, nb::kw_only(), "lowering"_a = nb::none())
      .def_prop_rw(
          "ir_output_profile",
          [](const CompilationStage &s) { return s.irOutputProfile; },
          [](CompilationStage &s, Profile profile) {
            s.irOutputProfile = std::move(profile);
          })
      .def_prop_rw(
          "lowering",
          [](const CompilationStage &s)
              -> std::optional<CustomPipelineLowering> {
            if (auto *p = std::get_if<CustomPipelineLowering>(&s.lowering))
              return *p;
            return std::nullopt;
          },
          [](CompilationStage &s,
             std::optional<CustomPipelineLowering> lowering) {
            s.lowering = lowering ? Lowering{std::move(*lowering)}
                                  : Lowering{InferredLowering{}};
          })
      .def("__repr__", [](const CompilationStage &s) {
        auto kind = std::visit(
            [](const auto &lowering) { return std::string(lowering.name); },
            s.lowering);
        auto detail = std::get_if<CustomPipelineLowering>(&s.lowering);
        return "CompilationStage(ir_output_profile=" + repr(s.irOutputProfile) +
               ", lowering=" + kind + (detail ? ": " + repr(*detail) : "") +
               ")";
      });
}
