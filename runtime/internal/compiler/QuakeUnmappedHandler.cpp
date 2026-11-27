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
#include "cudaq/runtime/logger/logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
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
/// `clifford-t-synthesis-pipeline`.
static bool schedulesCliffordTSynthesis(
    const cudaq::config::QuakeUnmappedProfile &profile) {
  if (profile.basisGates.empty() ||
      profile.costModel != cudaq::config::CostModel::t_count)
    return false;
  return cudaq::opt::needsCliffordTSynthesis(profile.basisGates);
}

/// The passes that fuse each one-qubit gate sequence into a single `u3`. A
/// sequence is replaced only when that is cheaper, or when it contains a gate
/// outside of the profile basis.
///
/// If `u3` cannot be decomposed into the provided basis, `std::nullopt` is
/// returned and a warning stating that single-qubits cannot be fused for the
/// provided basis is emitted.
static std::optional<std::string>
fusePasses(const cudaq::config::QuakeUnmappedProfile &profile) {
  const bool tCount = profile.costModel == cudaq::config::CostModel::t_count;

  // Define the set of allowed gates. The presence of gates outside of the
  // basis forces fusion no matter the cost model. If rotation synthesis will
  // run, allow rotation gates. An empty basis allows all gates.
  const std::vector<std::string> basis =
      tCount && schedulesCliffordTSynthesis(profile)
          ? cudaq::opt::cliffordTDecompositionBasis(profile.basisGates)
          : profile.basisGates;
  if (!cudaq::opt::canFuseToU3(basis)) {
    CUDAQ_WARN("Single-qubit gate fusion skipped: no known decomposition "
               "lowers u3 to the basis gates [{}]. Is the basis complete? "
               "Add u3, or a rotation gate, or register the required gate "
               "decomposition to solve this.",
               llvm::join(profile.basisGates, ", "));
    return std::nullopt;
  }

  // The cost function to minimize: either the number of gates or only the
  // rotations that need approximate synthesis.
  const std::string fuseCost =
      tCount ? "clifford-t-free epsilon=" +
                   formatTolerance(profile.approximationTolerance)
             : std::string("gate-count");
  return "func.func(fuse-1q-to-u3{cost=" + fuseCost +
         " basis=" + llvm::join(basis, ",") + "},canonicalize)";
}

/// The passes that optimize for the profile's cost model.
///
/// When the profile has a basis, the kernel is first decomposed to that basis
/// plus the gates the optimizer needs to work with. For `t-count` those are
/// the rotations it merges and snaps before synthesis. For
/// `two-qubit-gate-count` it is the controlled `x`, because `phase-folding`
/// folds rotations across it. The handler's closing `basis-conversion` removes
/// the extra gates again.
///
/// `gate-count` adds no gates. From level 1 it cancels and merges gates.
static std::vector<std::string>
costModelPasses(const cudaq::config::QuakeUnmappedProfile &profile,
                const cudaq::CompileOptions &options) {
  const bool tCount = profile.costModel == cudaq::config::CostModel::t_count;
  const bool gateCount =
      cudaq_internal::compiler::compilesForSimulator(profile);
  const std::string tolerance = formatTolerance(profile.approximationTolerance);
  const auto level = static_cast<unsigned>(options.optimizationLevel);
  std::vector<std::string> passes;
  if (!profile.basisGates.empty()) {
    auto basis =
        tCount && schedulesCliffordTSynthesis(profile)
            ? cudaq::opt::cliffordTDecompositionBasis(profile.basisGates)
            : profile.basisGates;
    if (!tCount && !gateCount && !llvm::is_contained(basis, "x(1)"))
      basis.emplace_back("x(1)");
    passes.push_back("decomposition{basis=" + llvm::join(basis, ",") + "}");
  }
  // A profile without a basis accepts any gate, so it keeps `exp_pauli`.
  const bool acceptsExpPauli =
      profile.basisGates.empty() ||
      llvm::is_contained(profile.basisGates, "exp_pauli");
  passes.push_back(cudaq_internal::compiler::convertToWires(
      profile, /*addWireSet=*/false, /*decomposeExpPauli=*/!acceptsExpPauli));
  // Handle any new phases the t-count decomposition may have introduced.
  if (tCount)
    passes.push_back("func.func(normalize-phase-placement,lower-phase)");
  std::optional<std::string> fusion;
  if (level >= 1 && gateCount) {
    // Like `two-qubit-gate-count`, `gate-count` only removes a rotation that is
    // exactly the identity. `quake-simplify` adds two constant angles exactly,
    // so a merged pair that cancels already gives 0, and a nonzero threshold
    // would only erase small rotations the kernel asked for. A simulator
    // applies noise by gate name, so `gate-count` does not fuse gates into
    // `u3`.
    passes.push_back("func.func(quake-simplify{threshold=0},dqe)");
  } else if (level >= 1) {
    passes.push_back(
        std::string(tCount ? "clifford-t" : "two-qubit-gate-count") +
        "-optimization-pipeline{level=" + std::to_string(level) +
        (tCount ? " approximation-tolerance=" + tolerance : "") + "}");
    fusion = fusePasses(profile);
    if (fusion)
      passes.push_back(*fusion);
  }
  // The synthesis pipeline adds no passes when the basis can express one of
  // the rotations it synthesizes exactly. Otherwise it needs the rotations
  // that each fused `u3` stands for.
  if (schedulesCliffordTSynthesis(profile)) {
    if (fusion)
      passes.push_back("decomposition{enable-patterns=U3ToRotations}");
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
    cudaq_internal::compiler::StagePasses stagePasses;
    std::vector<std::string> passes;
    // Every cost model converts to wires and applies the level.
    if (profile.costModel) {
      passes = costModelPasses(profile, options);
      stagePasses.normalizesKernel = true;
    }
    if (!basis.empty())
      passes.push_back("basis-conversion{basis=" + llvm::join(basis, ",") +
                       "}");
    stagePasses.needsGateLowering = !basis.empty();
    stagePasses.postDeployPasses = llvm::join(passes, ",");
    return stagePasses;
  }

  /// Synthesis needs the value of every rotation angle, and angles often
  /// depend on an argument, as in a QFT whose width is an argument. This asks
  /// the synthesis pipeline's own question, for the stages where
  /// `costModelPasses` schedules it.
  bool
  requiresArgumentValues(const cudaq::config::Profile &input,
                         const cudaq::config::Profile &output) const override {
    const auto *profile =
        std::get_if<cudaq::config::QuakeUnmappedProfile>(&output);
    return profile && schedulesCliffordTSynthesis(*profile);
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeUnmappedHandler() {
  return std::make_unique<QuakeUnmappedHandler>();
}
