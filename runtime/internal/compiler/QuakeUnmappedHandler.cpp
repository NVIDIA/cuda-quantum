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
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

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

/// The passes that optimize for the profile's cost model. A non-empty basis
/// first gains the gates the optimizer works with: for `t-count`, the
/// rotations it merges and snaps before synthesis; for `two-qubit-gate-count`,
/// the controlled `x` across which `phase-folding` folds rotations. The
/// handler's closing `basis-conversion` removes them again.
static std::vector<std::string>
costModelPasses(const cudaq::config::QuakeUnmappedProfile &profile,
                const cudaq::CompileOptions &options) {
  const bool tCount = profile.costModel == cudaq::config::CostModel::t_count;
  const std::string tolerance = formatTolerance(profile.approximationTolerance);
  const auto level = static_cast<unsigned>(options.optimizationLevel);
  std::vector<std::string> passes;
  if (!profile.basisGates.empty()) {
    auto basis =
        tCount ? cudaq::opt::cliffordTDecompositionBasis(profile.basisGates)
               : profile.basisGates;
    if (!tCount && !llvm::is_contained(basis, "x(1)"))
      basis.emplace_back("x(1)");
    passes.push_back("decomposition{basis=" + llvm::join(basis, ",") + "}");
  }
  passes.push_back(cudaq_internal::compiler::convertToWires(profile));
  // In a `t-count` stage, decomposition leaves a `quake.phase` beside some of
  // the gates it rewrites, such as a controlled rotation, and only the level-2
  // optimizer would lower it. Lower the phase here, at every level and before
  // the optimizer and synthesis. Left in place, the phase keeps the level-1
  // cleanup from merging the rotations on either side of it, and a controlled
  // phase lowers to an `r1` that synthesis has to see.
  if (tCount)
    passes.push_back("func.func(normalize-phase-placement,lower-phase)");
  if (level >= 1)
    passes.push_back(
        std::string(tCount ? "clifford-t" : "two-qubit-gate-count") +
        "-optimization-pipeline{level=" + std::to_string(level) +
        (tCount ? " approximation-tolerance=" + tolerance : "") + "}");
  // The synthesis pipeline adds no passes when the basis can express one of
  // the rotations it synthesizes exactly.
  if (schedulesCliffordTSynthesis(profile)) {
    const std::string seed =
        options.seed ? " seed=" + std::to_string(*options.seed) : "";
    passes.push_back("clifford-t-synthesis-pipeline{basis=" +
                     llvm::join(profile.basisGates, ",") +
                     " level=" + std::to_string(level) +
                     " approximation-tolerance=" + tolerance + seed + "}");
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
    // The two-qubit-gate-count optimization works on controlled `x` gates, so
    // fail here rather than in the closing `basis-conversion` when the basis
    // can't express one.
    if (profile.costModel == cudaq::config::CostModel::two_qubit_gate_count &&
        !cudaq::opt::canOptimizeTwoQubitGateCount(basis))
      return std::nullopt;
    std::vector<std::string> passes;
    if (profile.costModel)
      passes = costModelPasses(profile, options);
    else if (profile.semantics == cudaq::config::Semantics::value)
      passes.push_back(cudaq_internal::compiler::convertToWires(profile));
    if (!basis.empty()) {
      passes.push_back("basis-conversion{basis=" + llvm::join(basis, ",") +
                       "}");
      // When the basis lacks a rotation, the conversion rewrites it with an
      // uncontrolled `quake.phase`, as when it rewrites `r1` as `rz`, and
      // `lower-phase` erases that phase. After a `t-count` stage's opening
      // decomposition, every gate left to rewrite is uncontrolled, so no
      // controlled phase, which would lower to new gates, reaches this point.
      if (profile.costModel == cudaq::config::CostModel::t_count)
        passes.push_back("func.func(lower-phase)");
    }
    // Lowering handlers run no verifier, except here: no later pass rejects a
    // register that the conversion left on references.
    if (profile.semantics == cudaq::config::Semantics::value)
      passes.push_back("verify-quake-unmapped-profile{wires-only=true}");
    cudaq_internal::compiler::StagePasses stagePasses;
    stagePasses.postDeployPasses = llvm::join(passes, ",");
    return stagePasses;
  }

  /// Synthesis needs the value of every rotation angle, and angles often
  /// depend on an argument, as in a QFT whose width is an argument. A stage
  /// synthesizes when `costModelPasses` schedules synthesis and
  /// `needsCliffordTSynthesis` holds for its basis, the same test the
  /// synthesis pipeline makes. Value semantics needs every register size and
  /// every loop bound that picks a qubit, which also often come from
  /// arguments.
  bool
  requiresArgumentValues(const cudaq::config::Profile &input,
                         const cudaq::config::Profile &output) const override {
    const auto *profile =
        std::get_if<cudaq::config::QuakeUnmappedProfile>(&output);
    if (!profile)
      return false;
    return profile->semantics == cudaq::config::Semantics::value ||
           (schedulesCliffordTSynthesis(*profile) &&
            cudaq::opt::needsCliffordTSynthesis(profile->basisGates));
  }

  /// Prep and deploy unroll every loop of up to 1024 iterations, the default
  /// threshold of `createClassicalOptimizationPipeline`, which the profile's
  /// default limit matches. When the stage keeps some loops rolled, because it
  /// unrolls only `index_use` loops or has a lower limit, prep and deploy must
  /// not unroll those loops first.
  bool
  requiresRolledLoops(const cudaq::config::Profile &input,
                      const cudaq::config::Profile &output) const override {
    const auto *profile =
        std::get_if<cudaq::config::QuakeUnmappedProfile>(&output);
    return profile &&
           (profile->unrollLoops != cudaq::config::LoopUnrolling::all ||
            profile->maxUnrollIterations <
                cudaq::config::QuakeUnmappedProfile{}.maxUnrollIterations);
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeUnmappedHandler() {
  return std::make_unique<QuakeUnmappedHandler>();
}
