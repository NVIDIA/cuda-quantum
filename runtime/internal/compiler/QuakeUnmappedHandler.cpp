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
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>

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

/// Whether reaching \p profile's basis requires Clifford+T synthesis, which is
/// when the cost model is `t-count` and a non-empty basis cannot express an
/// uncontrolled rotation.
static bool
requiresSynthesis(const cudaq::config::QuakeUnmappedProfile &profile) {
  // A basis with any of these continuous rotations, uncontrolled or with any
  // number of controls, can express every rotation without approximating it.
  static constexpr std::string_view rotations[] = {"r1", "rx",        "ry",
                                                   "rz", "phased_rx", "u3"};
  const auto isRotation = [](std::string_view gate) {
    const auto controls = gate.find('(');
    if (controls != std::string_view::npos && gate.substr(controls) != "(n)")
      return false;
    gate = gate.substr(0, controls);
    return std::find(std::begin(rotations), std::end(rotations), gate) !=
           std::end(rotations);
  };
  const auto &basis = profile.basisGates;
  return profile.costModel == cudaq::config::CostModel::t_count &&
         !basis.empty() && std::none_of(basis.begin(), basis.end(), isRotation);
}

namespace {

/// The built-in handler for the quake-unmapped profile. It picks passes from
/// the profile's options and the level and seed in `CompileOptions`, and runs
/// them all between deploy and finalize.
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
      // Print every digit the passes need to read back the same tolerance.
      std::ostringstream toleranceText;
      toleranceText.imbue(std::locale::classic());
      toleranceText << std::setprecision(
                           std::numeric_limits<double>::max_digits10)
                    << profile.approximationTolerance;
      const std::string tolerance = toleranceText.str();
      const auto level = static_cast<unsigned>(options.optimizationLevel);

      // Keep rotations so that the optimizer can merge and snap them.
      if (!basis.empty()) {
        auto basisWithRotations = basis;
        for (const char *rotation : {"r1", "rx", "ry", "rz"})
          if (std::find(basis.begin(), basis.end(), rotation) == basis.end())
            basisWithRotations.push_back(rotation);
        passes.push_back("decomposition{basis=" + join(basisWithRotations) +
                         "}");
      }
      if (level >= 1) {
        // Without `add-wireset`, this converts to wires but adds no wire set.
        passes.push_back("prepare-for-wireset{maximum-iterations=" +
                         std::to_string(maxUnrollIterations) + "}");
        passes.push_back("clifford-t-optimization-pipeline{level=" +
                         std::to_string(level) + " epsilon=" + tolerance + "}");
      }
      if (requiresSynthesis(profile)) {
        passes.push_back("clifford-t-synthesis-pipeline{epsilon=" + tolerance +
                         " seed=" + std::to_string(options.seed) + "}");
        // Clifford+T normalization emits a global phase, so lower it again.
        if (level >= 2)
          passes.push_back("func.func(quake-simplify{threshold=" + tolerance +
                           "},normalize-1q-clifford-t,phase-folding,"
                           "normalize-phase-placement,lower-phase,canonicalize,"
                           "dqe)");
      }
    }

    if (!basis.empty())
      passes.push_back("basis-conversion{basis=" + join(basis) + "}");
    cudaq_internal::compiler::StagePasses stagePasses;
    stagePasses.postDeployPasses = join(passes);
    return stagePasses;
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeUnmappedHandler() {
  return std::make_unique<QuakeUnmappedHandler>();
}
