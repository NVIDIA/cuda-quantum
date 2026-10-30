/*****************************************************************- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/LoweringHandler.h"

namespace {

/// The built-in generic lowering handler for the quake-reference-semantics
/// profile. It returns the passes the profile's options request, placed where
/// the provider targets' pipelines have run them. `expand-measurements` runs
/// before deploy when the profile requests single-qubit measurements or
/// `reset-after-measure`. `basis-conversion` runs between deploy and finalize
/// when the profile lists basis gates. `combine-quantum-alloc` and
/// `qubit-reset-before-reuse` run after finalize when the profile requests
/// `reset-after-measure`.
///
/// Only a target's first stage may add passes before deploy, and only its
/// last stage may add passes after finalize. So a stage with this profile must
/// be first when it requests single-qubit measurements or
/// `reset-after-measure`, and last when it requests `reset-after-measure`.
/// The handler is generic over the profile options and never inspects
/// provider names.
///
/// `QuakeUnmappedHandler` supersedes this handler, which goes away with its
/// profile.
class QuakeReferenceSemanticsHandler
    : public cudaq_internal::compiler::LoweringHandler {
public:
  bool supports(const cudaq::config::Profile &input,
                const cudaq::config::Profile &output,
                const cudaq::CompileOptions &options) const override {
    const auto *referenceProfile =
        std::get_if<cudaq::config::QuakeReferenceSemanticsProfile>(&output);
    if (!referenceProfile ||
        (!std::holds_alternative<cudaq::config::UncheckedProfile>(input) &&
         !std::holds_alternative<cudaq::config::QuakeReferenceSemanticsProfile>(
             input)))
      return false;
    return true;
  }

  std::optional<cudaq_internal::compiler::StagePasses>
  getStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options) const override {
    if (!supports(input, output, options))
      return std::nullopt;
    const auto &referenceProfile =
        std::get<cudaq::config::QuakeReferenceSemanticsProfile>(output);

    cudaq_internal::compiler::StagePasses passes;
    // Single-qubit measurements and `reset-after-measure` both need
    // `expand-measurements`.
    if (referenceProfile.measurementType ==
            cudaq::config::MeasurementForm::single_qubit ||
        referenceProfile.resetAfterMeasure)
      passes.preDeployPasses = "expand-measurements";

    if (!referenceProfile.basisGates.empty()) {
      passes.postDeployPasses = "basis-conversion{basis=";
      bool first = true;
      for (const auto &gate : referenceProfile.basisGates) {
        if (!first)
          passes.postDeployPasses += ",";
        first = false;
        passes.postDeployPasses += gate;
      }
      passes.postDeployPasses += "}";
    }

    if (referenceProfile.resetAfterMeasure)
      passes.postFinalizePasses =
          "func.func(combine-quantum-alloc,qubit-reset-before-reuse)";

    return passes;
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeReferenceSemanticsHandler() {
  return std::make_unique<QuakeReferenceSemanticsHandler>();
}
