// Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

template <typename T>
std::string repr(const T &value) {
  return nb::repr(nb::cast(value)).c_str();
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
      .def_prop_rw(
          "basis_gates",
          [](const QuakeReferenceSemanticsProfile &p) { return p.basisGates; },
          [](QuakeReferenceSemanticsProfile &p,
             std::vector<std::string> gates) {
            p.basisGates = std::move(gates);
          })
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
