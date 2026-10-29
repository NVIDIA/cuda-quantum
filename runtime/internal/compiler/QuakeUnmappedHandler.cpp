/*****************************************************************- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CompileOptions.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Optimizer/Transforms/CliffordTPipelines.h"
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>

/// Loops up to this length are unrolled so that their qubits become wires, as
/// in `classical-optimization-pipeline`. Longer loops stay on references.
static constexpr unsigned maxUnrollIterations = 1024;

/// Join \p items with commas, as pass pipelines and list options expect.
static std::string join(const std::vector<std::string> &items) {
  std::string joined;
  for (const auto &item : items) {
    if (!joined.empty())
      joined += ',';
    joined += item;
  }
  return joined;
}

/// Print \p tolerance with every digit the passes need to read it back,
/// independent of the locale. This avoids floating-point `std::to_chars`, which
/// macOS provides only from 13.3.
static std::string formatTolerance(double tolerance) {
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << std::setprecision(std::numeric_limits<double>::max_digits10)
       << tolerance;
  return text.str();
}

/// Whether a stage lowering to \p profile schedules
/// `clifford-t-synthesis-pipeline`. The pipeline itself then decides, through
/// `needsCliffordTSynthesis`, whether the basis needs any rotation
/// approximated.
static bool schedulesCliffordTSynthesis(
    const cudaq::config::QuakeUnmappedProfile &profile) {
  return profile.costModel == cudaq::config::CostModel::t_count &&
         !profile.basisGates.empty();
}

/// The passes that optimize for, and synthesize to, a `t-count` cost model. A
/// non-empty basis first gains the rotations the optimizer keeps, so that it
/// can merge and snap them before synthesis or basis conversion.
static std::vector<std::string>
cliffordTPasses(const cudaq::config::QuakeUnmappedProfile &profile,
                const cudaq::CompileOptions &options) {
  const std::string tolerance = formatTolerance(profile.approximationTolerance);
  const auto level = static_cast<unsigned>(options.optimizationLevel);
  std::vector<std::string> passes;
  if (!profile.basisGates.empty())
    passes.push_back(
        "decomposition{basis=" +
        join(cudaq::opt::cliffordTDecompositionBasis(profile.basisGates)) +
        "}");
  if (level >= 1) {
    // Without `add-wireset`, this converts to wires but adds no wire set.
    passes.push_back("prepare-for-wireset{maximum-iterations=" +
                     std::to_string(maxUnrollIterations) + "}");
    passes.push_back("clifford-t-optimization-pipeline{level=" +
                     std::to_string(level) + " epsilon=" + tolerance + "}");
  }
  // The synthesis pipeline adds no passes when the basis expresses any of
  // these rotations exactly.
  if (schedulesCliffordTSynthesis(profile)) {
    const std::string seed =
        options.seed ? " seed=" + std::to_string(*options.seed) : "";
    passes.push_back(
        "clifford-t-synthesis-pipeline{basis=" + join(profile.basisGates) +
        " level=" + std::to_string(level) + " epsilon=" + tolerance + seed +
        "}");
  }
  return passes;
}

namespace {

/// The built-in handler for the quake-unmapped profile. It picks passes from
/// the profile's options and the level and seed in `CompileOptions`, and runs
/// them all between deploy and finalize.
class QuakeUnmappedHandler : public cudaq_internal::compiler::LoweringHandler {
public:
  bool supports(const cudaq::config::Profile &input,
                const cudaq::config::Profile &output) const override {
    return std::holds_alternative<cudaq::config::UncheckedProfile>(input) &&
           std::holds_alternative<cudaq::config::QuakeUnmappedProfile>(output);
  }

  std::optional<cudaq_internal::compiler::StagePasses>
  getStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options) const override {
    if (!supports(input, output))
      return std::nullopt;
    const auto &profile = std::get<cudaq::config::QuakeUnmappedProfile>(output);
    const auto &basis = profile.basisGates;
    std::vector<std::string> passes;
    if (profile.costModel == cudaq::config::CostModel::t_count)
      passes = cliffordTPasses(profile, options);
    if (!basis.empty())
      passes.push_back("basis-conversion{basis=" + join(basis) + "}");
    cudaq_internal::compiler::StagePasses stagePasses;
    stagePasses.postDeployPasses = join(passes);
    return stagePasses;
  }

  /// Synthesis needs the value of every rotation angle, and angles often
  /// depend on an argument, as in a QFT whose width is an argument. This asks
  /// the synthesis pipeline's own question, for the stages where
  /// `cliffordTPasses` schedules it.
  bool
  requiresArgumentValues(const cudaq::config::Profile &input,
                         const cudaq::config::Profile &output) const override {
    const auto *profile =
        std::get_if<cudaq::config::QuakeUnmappedProfile>(&output);
    return profile && schedulesCliffordTSynthesis(*profile) &&
           cudaq::opt::needsCliffordTSynthesis(profile->basisGates);
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeUnmappedHandler() {
  return std::make_unique<QuakeUnmappedHandler>();
}
