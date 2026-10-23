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
/// profile.
///
/// Builds its pipeline from the common JIT prep/deploy/finalize stages plus
/// the profile's requested basis-conversion, expand-measurements, and
/// qubit-reset-before-reuse passes. It is generic over the profile options and
/// never inspects provider names.
class QuakeReferenceSemanticsHandler
    : public cudaq_internal::compiler::LoweringHandler {
  const bool emulate;
  const std::string emissionFormat;

public:
  QuakeReferenceSemanticsHandler(bool emulate_, std::string emissionFormat_)
      : emulate(emulate_), emissionFormat(std::move(emissionFormat_)) {}

  bool supports(const cudaq::config::Profile &input,
                const cudaq::config::Profile &output) const override;

  std::optional<std::string>
  getPipeline(const cudaq::config::Profile &input,
              const cudaq::config::Profile &output) const override;
};

} // namespace

bool QuakeReferenceSemanticsHandler::supports(
    const cudaq::config::Profile &input,
    const cudaq::config::Profile &output) const {
  const auto *referenceProfile =
      std::get_if<cudaq::config::QuakeReferenceSemanticsProfile>(&output);
  if (!referenceProfile ||
      (!std::holds_alternative<cudaq::config::UncheckedProfile>(input) &&
       !std::holds_alternative<cudaq::config::QuakeReferenceSemanticsProfile>(
           input)))
    return false;
  return true;
}

std::optional<std::string> QuakeReferenceSemanticsHandler::getPipeline(
    const cudaq::config::Profile &input,
    const cudaq::config::Profile &output) const {
  if (!supports(input, output))
    return std::nullopt;
  const auto &referenceProfile =
      std::get<cudaq::config::QuakeReferenceSemanticsProfile>(output);

  // Common preparation, deployment, and finalization, mirroring the
  // custom-pipeline lowering path.
  const bool nopCodegen = emissionFormat == "nop";
  const std::string allowEarlyExit =
      emissionFormat.starts_with("qir-adaptive") ? "true" : "false";
  const std::string noLoopUnroll = nopCodegen ? " no-loop-unroll=true" : "";
  bool expandMeasurements = referenceProfile.measurementType ==
                            cudaq::config::MeasurementForm::single_qubit;
  // Force expand measurements if `reset-after-measure` is requested.
  if (referenceProfile.resetAfterMeasure)
    expandMeasurements = true;

  std::string pipeline = "canonicalize,";
  pipeline +=
      emulate ? "emul-jit-prep-pipeline{erase-noise=true allow-early-exit=" +
                    allowEarlyExit + noLoopUnroll + "}"
              : "hw-jit-prep-pipeline{allow-early-exit=" + allowEarlyExit +
                    noLoopUnroll + "}";

  if (expandMeasurements)
    pipeline += ",expand-measurements";

  pipeline += nopCodegen ? ",jit-deploy-pipeline{no-loop-unroll=true}"
                         : ",jit-deploy-pipeline";

  if (!referenceProfile.basisGates.empty()) {
    pipeline += ",basis-conversion{basis=";
    bool first = true;
    for (const auto &gate : referenceProfile.basisGates) {
      if (!first)
        pipeline += ",";
      first = false;
      pipeline += gate;
    }
    pipeline += "}";
  }

  pipeline += ",jit-finalize-pipeline{lower-device-calls=";
  pipeline += nopCodegen && !emulate ? "false" : "true";
  pipeline += "}";

  if (referenceProfile.resetAfterMeasure)
    pipeline += ",func.func(combine-quantum-alloc,qubit-reset-before-reuse)";

  return pipeline;
}

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeReferenceSemanticsHandler(
    bool emulate, std::string emissionFormat) {
  return std::make_unique<QuakeReferenceSemanticsHandler>(
      emulate, std::move(emissionFormat));
}
