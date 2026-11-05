/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Nanobind type caster between `cudaq::CompileTarget` (and related types) and
/// the dataclasses in `cudaq.core._experimental.target`. Types are deep-copied
/// when converted.
///
/// Include this header in every translation unit whose nanobind bindings
/// mention `cudaq::CompileTarget`.

#include "cudaq/Target/CompileTarget.h"
#include "cudaq/Target/python/DataclassCaster.h"

namespace cudaq::python {

/// The Python module `cudaq.core._experimental.target`.
class CudaqTargetPyModule {
public:
  /// Singleton, imports the module on first use.
  static const CudaqTargetPyModule &get() {
    static CudaqTargetPyModule *instance = nullptr;
    if (!instance)
      instance = new CudaqTargetPyModule();
    return *instance;
  }

  bool isInstance(nb::handle obj, const char *type) const {
    return nb::isinstance(obj, module.attr(type));
  }

  /// Calls the dataclass `type` with keyword arguments.
  nb::object create(const char *type, const nb::dict &kwargs) const {
    return module.attr(type)(**kwargs);
  }

  /// The name of `obj` if it is a member of the enum `type`.
  std::optional<std::string> enumMemberName(nb::handle obj,
                                            const char *type) const {
    if (!isInstance(obj, type))
      return std::nullopt;
    return nb::cast<std::string>(obj.attr("name"));
  }

  nb::object enumMember(const char *type, const char *name) const {
    return module.attr(type).attr(name);
  }

private:
  CudaqTargetPyModule()
      : module(nb::module_::import_("cudaq.core._experimental.target")) {}

  nb::module_ module;
};

template <>
struct Enum<NoiseSupport> {
  static constexpr auto name = nb::detail::const_name("NoiseSupport");
  static constexpr EnumMember<NoiseSupport> members[] = {
      {"erase", NoiseSupport::erase},
      {"supported", NoiseSupport::supported},
      {"unsupported", NoiseSupport::unsupported}};
};

template <>
struct Enum<config::CostModel> {
  static constexpr auto name = nb::detail::const_name("CostModel");
  static constexpr EnumMember<config::CostModel> members[] = {
      {"t_count", config::CostModel::t_count},
      {"two_qubit_gate_count", config::CostModel::two_qubit_gate_count}};
};

template <>
struct Enum<config::LoopUnrolling> {
  static constexpr auto name = nb::detail::const_name("LoopUnrolling");
  static constexpr EnumMember<config::LoopUnrolling> members[] = {
      {"all", config::LoopUnrolling::all},
      {"index_use", config::LoopUnrolling::index_use}};
};

template <>
struct Enum<config::MeasurementForm> {
  static constexpr auto name = nb::detail::const_name("str");
  static constexpr EnumMember<config::MeasurementForm> members[] = {
      {"any", config::MeasurementForm::any},
      {"single_qubit", config::MeasurementForm::single_qubit}};
};

template <>
struct Dataclass<config::UncheckedProfile> {
  static constexpr auto name = nb::detail::const_name("UncheckedProfile");
  static constexpr std::tuple<> fields{};
};

template <>
struct Dataclass<config::QuakeReferenceSemanticsProfile> {
  using T = config::QuakeReferenceSemanticsProfile;
  static constexpr auto name =
      nb::detail::const_name("QuakeReferenceSemanticsProfile");
  static constexpr std::tuple fields{
      Field{"basis_gates", &T::basisGates},
      Field{"measurement_type", &T::measurementType},
      Field{"reset_after_measure", &T::resetAfterMeasure},
  };
};

template <>
struct Dataclass<config::QuakeUnmappedProfile> {
  using T = config::QuakeUnmappedProfile;
  static constexpr auto name = nb::detail::const_name("QuakeUnmappedProfile");
  static constexpr std::tuple fields{
      Field{"basis_gates", &T::basisGates},
      Field{"approximation_tolerance", &T::approximationTolerance},
      Field{"cost_model", &T::costModel},
      Field{"unroll_loops", &T::unrollLoops},
      Field{"max_unroll_iterations", &T::maxUnrollIterations},
  };
};

template <>
struct Dataclass<config::QuakeMappedProfile> {
  using T = config::QuakeMappedProfile;
  static constexpr auto name = nb::detail::const_name("QuakeMappedProfile");
  static constexpr std::tuple fields{
      Field{"basis_gates", &T::basisGates},
      Field{"device", &T::device},
  };
};

template <>
struct Dataclass<config::CustomPipelineLowering> {
  using T = config::CustomPipelineLowering;
  static constexpr auto name = nb::detail::const_name("CustomPipelineLowering");
  static constexpr std::tuple fields{
      Field{"override_pass_pipeline", &T::overridePassPipeline},
      Field{"high_level", &T::highLevelPipeline},
      Field{"mid_level", &T::midLevelPipeline},
      Field{"low_level", &T::lowLevelPipeline},
  };
};

template <>
struct Dataclass<config::CompilationStage> {
  using T = config::CompilationStage;
  static constexpr auto name = nb::detail::const_name("CompilationStage");
  static constexpr std::tuple fields{
      Field{"ir_output_profile", &T::irOutputProfile},
      Field{"lowering", &T::lowering},
  };
};

template <>
struct Dataclass<config::Codegen> {
  using T = config::Codegen;
  static constexpr auto name = nb::detail::const_name("Codegen");
  static constexpr std::tuple fields{
      Field{"emission_format", &T::emissionFormat},
      Field{"post_codegen_passes", &T::postCodegenPasses},
  };
};

template <>
struct Dataclass<CompileTarget> {
  using T = CompileTarget;
  static constexpr auto name = nb::detail::const_name("CompileTarget");
  static constexpr std::tuple fields{
      Field{"stages", &T::stages},
      Field{"codegen", &T::codegen},
      Field{"override_aot_compilation", &T::overrideAOTCompilation},
      Field{"support_conditionals_on_measure_results",
            &T::supportConditionalsOnMeasureResults},
      Field{"support_device_calls", &T::supportDeviceCalls},
      Field{"support_resource_counts", &T::supportResourceCounts},
      Field{"support_explicit_measurements", &T::supportExplicitMeasurements},
      Field{"support_observable_measurements",
            &T::supportObservableMeasurements},
      Field{"support_sample_without_measurements",
            &T::supportSampleWithoutMeasurements},
      Field{"fully_specialize", &T::fullySpecialize},
      Field{"preserve_loops", &T::preserveLoops},
      Field{"noise_support", &T::noiseSupport},
      Field{"erase_qec", &T::eraseQEC},
      Field{"argument_synth_change_semantics",
            &T::argumentSynthChangeSemantics},
      Field{"disable_qubit_mapping", &T::disableQubitMapping},
      Field{"replace_state_with_kernel", &T::replaceStateWithKernel},
  };
};

} // namespace cudaq::python

namespace nanobind::detail {

template <typename T>
using target_dataclass_caster =
    dataclass_caster<T, cudaq::python::CudaqTargetPyModule>;
template <typename E>
using target_enum_caster = enum_caster<E, cudaq::python::CudaqTargetPyModule>;

template <typename T>
struct type_caster<T, enable_if_t<cudaq::python::isDataclass<T>>>
    : target_dataclass_caster<T> {};

template <>
struct type_caster<cudaq::NoiseSupport>
    : target_enum_caster<cudaq::NoiseSupport> {};
template <>
struct type_caster<cudaq::config::CostModel>
    : target_enum_caster<cudaq::config::CostModel> {};
template <>
struct type_caster<cudaq::config::LoopUnrolling>
    : target_enum_caster<cudaq::config::LoopUnrolling> {};
template <>
struct type_caster<cudaq::config::MeasurementForm>
    : target_enum_caster<cudaq::config::MeasurementForm> {};

/// `InferredLowering` is `None` in Python.
template <>
struct type_caster<cudaq::config::InferredLowering>
    : none_caster<cudaq::config::InferredLowering> {};

/// Also accepts the emission format as a string.
template <>
struct type_caster<cudaq::config::Codegen>
    : target_dataclass_caster<cudaq::config::Codegen> {
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (std::string format; try_cast(src, format)) {
      value = {std::move(format), ""};
      return true;
    }
    return dataclass_caster::from_python(src, flags, cleanup);
  }
};

template <>
struct type_caster<cudaq::config::CustomPipelineLowering>
    : target_dataclass_caster<cudaq::config::CustomPipelineLowering> {
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (!dataclass_caster::from_python(src, flags, cleanup))
      return false;
    if (value.overridePassPipeline.empty() ||
        (value.highLevelPipeline.empty() && value.midLevelPipeline.empty() &&
         value.lowLevelPipeline.empty()))
      return true;
    cudaq::python::conversionError() =
        ": override_pass_pipeline cannot be combined with pipeline fragments";
    return false;
  }
};

/// Throws a `TypeError` naming the offending field if `src` is an invalid
/// `CompileTarget`.
template <>
struct type_caster<cudaq::CompileTarget>
    : target_dataclass_caster<cudaq::CompileTarget> {
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    cudaq::python::conversionError().clear();
    if (dataclass_caster::from_python(src, flags, cleanup))
      return true;
    if (cudaq::python::conversionError().empty())
      return false;
    throw type_error(
        ("CompileTarget" + cudaq::python::conversionError()).c_str());
  }
};

/// Nanobind's caster for `std::optional` is `noexcept`, so it cannot hold the
/// throwing `CompileTarget` caster.
template <>
struct type_caster<std::optional<cudaq::CompileTarget>> {
  NB_TYPE_CASTER(std::optional<cudaq::CompileTarget>,
                 const_name("CompileTarget | None"))

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    if (src.is_none()) {
      value.reset();
      return true;
    }
    make_caster<cudaq::CompileTarget> caster;
    if (!caster.from_python(src, flags, cleanup))
      return false;
    value = std::move(caster.value);
    return true;
  }

  static handle from_cpp(const std::optional<cudaq::CompileTarget> &target,
                         rv_policy policy, cleanup_list *cleanup) {
    if (!target)
      return none().release();
    return make_caster<cudaq::CompileTarget>::from_cpp(*target, policy,
                                                       cleanup);
  }
};

} // namespace nanobind::detail
