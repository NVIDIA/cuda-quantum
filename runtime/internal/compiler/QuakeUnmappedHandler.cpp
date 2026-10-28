/*****************************************************************- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CompileOptions.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include <algorithm>

/// Loops of up to this many iterations are unrolled so that their qubits can be
/// promoted to wires. This is the default limit of
/// `classical-optimization-pipeline`. A longer loop stays rolled on references,
/// where the optimization pipeline skips it and resource counting still counts
/// it.
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

/// Return the level-2 cleanup that runs after Clifford+T synthesis.
/// `normalize-1q-clifford-t` emits an omega global phase, so `lower-phase` runs
/// after it.
static std::string cleanupAfterSynthesis(const std::string &tolerance) {
  return "func.func(quake-simplify{threshold=" + tolerance +
         "},normalize-1q-clifford-t,phase-folding,normalize-phase-placement,"
         "lower-phase,canonicalize,dqe)";
}

namespace {

/// The built-in handler for the quake-unmapped profile. It chooses passes from
/// the profile's options and the optimization level and seed in
/// `CompileOptions`, and runs them all between deploy and finalize. With the
/// `t-count` cost model, the stage keeps rotations through optimization,
/// converts to wires at level 1 and above, and synthesizes Clifford+T when the
/// basis has no rotation gate. Without a cost model, the stage only converts
/// to the basis and verifies the profile.
class QuakeUnmappedHandler : public cudaq_internal::compiler::LoweringHandler {
public:
  bool supports(const cudaq::config::Profile &input,
                const cudaq::config::Profile &output,
                const cudaq::CompileOptions &) const override {
    return std::holds_alternative<cudaq::config::UncheckedProfile>(input) &&
           std::holds_alternative<cudaq::config::QuakeUnmappedProfile>(output);
  }

  std::optional<cudaq_internal::compiler::StagePasses>
  getStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options) const override {
    if (!supports(input, output, options))
      return std::nullopt;
    const auto &profile = std::get<cudaq::config::QuakeUnmappedProfile>(output);
    const auto &basis = profile.basisGates;

    std::vector<std::string> passes;
    if (profile.costModel == cudaq::config::CostModel::t_count) {
      // A tolerance of 0 only accepts exact rewrites. `CompileTarget` rejects
      // a profile that needs synthesis without a tolerance.
      const std::string tolerance = profile.approximationTolerance.empty()
                                        ? "0"
                                        : profile.approximationTolerance;
      const auto level = static_cast<unsigned>(options.optimizationLevel);

      // Keep rotations through optimization so that later passes can merge
      // and snap them before synthesis or the final basis conversion.
      if (!basis.empty()) {
        auto basisWithRotations = basis;
        for (const char *rotation : {"r1", "rx", "ry", "rz"})
          if (std::find(basis.begin(), basis.end(), rotation) == basis.end())
            basisWithRotations.push_back(rotation);
        passes.push_back("decomposition{basis=" + join(basisWithRotations) +
                         "}");
      }
      if (level >= 1) {
        // Without `add-wireset`, this unrolls loops and converts to wires but
        // adds no wire set.
        passes.push_back("prepare-for-wireset{maximum-iterations=" +
                         std::to_string(maxUnrollIterations) + "}");
        passes.push_back("clifford-t-optimization-pipeline{level=" +
                         std::to_string(level) + " epsilon=" + tolerance + "}");
      }
      if (profile.requiresSynthesis()) {
        passes.push_back("clifford-t-synthesis-pipeline{epsilon=" + tolerance +
                         " seed=" + std::to_string(options.seed) + "}");
        if (level >= 2)
          passes.push_back(cleanupAfterSynthesis(tolerance));
      }
    }

    if (basis.empty()) {
      passes.push_back("verify-profile-quake-unmapped");
    } else {
      const std::string basisList = join(basis);
      passes.push_back("basis-conversion{basis=" + basisList + "}");
      passes.push_back("verify-profile-quake-unmapped{basis=" + basisList +
                       "}");
    }
    return cudaq_internal::compiler::StagePasses{.postDeployPasses =
                                                     join(passes)};
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeUnmappedHandler() {
  return std::make_unique<QuakeUnmappedHandler>();
}
