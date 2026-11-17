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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

/// Whether a stage lowering to \p profile schedules
/// `clifford-t-synthesis-pipeline`.
static bool schedulesCliffordTSynthesis(
    const cudaq::config::QuakeUnmappedProfile &profile) {
  if (profile.basisGates.empty() ||
      profile.costModel != cudaq::config::CostModel::t_count)
    return false;
  return cudaq::opt::needsCliffordTSynthesis(profile.basisGates);
}

/// Add the passes that fuse each one-qubit gate sequence into a single `u3`.
/// A sequence is replaced only when that is cheaper, or, when it contains a
/// gate outside of the profile basis.
static void addFusePasses(mlir::OpPassManager &pm,
                          const cudaq::config::QuakeUnmappedProfile &profile) {
  const bool tCount = profile.costModel == cudaq::config::CostModel::t_count;

  // The cost function to minimize: either the total number of rotations or
  // only non-Clifford rotations. The basis is the set of allowed gates: gates
  // outside of it force fusion no matter the cost model. If rotation synthesis
  // will run, allow rotation gates. An empty basis allows all gates.
  cudaq::opt::Fuse1QToU3Options fuseOptions;
  std::vector<std::string> basis = profile.basisGates;
  if (tCount) {
    fuseOptions.cost = "clifford-t-free";
    fuseOptions.epsilon = profile.approximationTolerance;
    if (schedulesCliffordTSynthesis(profile))
      basis = cudaq::opt::cliffordTDecompositionBasis(profile.basisGates);
  } else {
    fuseOptions.cost = "rotation-count";
  }
  fuseOptions.basis.assign(basis.begin(), basis.end());

  auto &funcPM = pm.nest<mlir::func::FuncOp>();
  funcPM.addPass(cudaq::opt::createFuse1QToU3(fuseOptions));
  funcPM.addPass(mlir::createCanonicalizerPass());
}

/// Add the passes that optimize for the profile's cost model. A non-empty
/// basis first gains the gates the optimizer works with: for `t-count`, the
/// rotations it merges and snaps before synthesis; for `two-qubit-gate-count`,
/// the controlled `x` across which `phase-folding` folds rotations. The
/// handler's closing `basis-conversion` removes them again.
static void
addCostModelPasses(mlir::OpPassManager &pm,
                   const cudaq::config::QuakeUnmappedProfile &profile,
                   unsigned level, std::uint64_t seed) {
  const bool tCount = profile.costModel == cudaq::config::CostModel::t_count;
  const double tolerance = profile.approximationTolerance;
  if (!profile.basisGates.empty()) {
    auto basis =
        tCount && schedulesCliffordTSynthesis(profile)
            ? cudaq::opt::cliffordTDecompositionBasis(profile.basisGates)
            : profile.basisGates;
    if (!tCount && !llvm::is_contained(basis, "x(1)"))
      basis.emplace_back("x(1)");
    cudaq::opt::DecompositionOptions decompositionOptions;
    decompositionOptions.basis.assign(basis.begin(), basis.end());
    pm.addPass(cudaq::opt::createDecomposition(decompositionOptions));
  }
  cudaq_internal::compiler::addConvertToWires(pm, profile);
  // Handle any new phases the t-count decomposition may have introduced.
  if (tCount) {
    auto &funcPM = pm.nest<mlir::func::FuncOp>();
    funcPM.addPass(cudaq::opt::createNormalizePhasePlacement());
    funcPM.addPass(cudaq::opt::createLowerPhase());
  }
  if (level >= 1) {
    if (tCount)
      cudaq::opt::addCliffordTOptimizationPipeline(pm, level, tolerance);
    else
      cudaq::opt::addTwoQubitGateCountOptimizationPipeline(pm, level);
    addFusePasses(pm, profile);
  }
  // The synthesis pipeline adds no passes when the basis can express one of
  // the rotations it synthesizes exactly. Otherwise it needs the rotations
  // that each fused `u3` stands for.
  if (schedulesCliffordTSynthesis(profile)) {
    if (level >= 1)
      cudaq::opt::addDecomposition(pm, {"U3ToRotations"});
    cudaq::opt::addCliffordTSynthesisPipeline(pm, profile.basisGates, level,
                                              tolerance, seed);
  }
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

  mlir::LogicalResult
  addStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options,
                 cudaq_internal::compiler::StagePasses &passes) const override {
    if (!supports(input, output))
      return mlir::failure();
    const auto &profile = std::get<cudaq::config::QuakeUnmappedProfile>(output);
    const auto &basis = profile.basisGates;
    // The two-qubit-gate-count optimization works on controlled `x` gates, so
    // fail here rather than in the closing `basis-conversion` when the basis
    // can't express one.
    if (profile.costModel == cudaq::config::CostModel::two_qubit_gate_count &&
        !cudaq::opt::canOptimizeTwoQubitGateCount(basis))
      return mlir::failure();
    passes.postDeployPasses =
        [profile, level = static_cast<unsigned>(options.optimizationLevel),
         seed = options.seed.value_or(0)](mlir::OpPassManager &pm) {
          if (profile.costModel)
            addCostModelPasses(pm, profile, level, seed);
          if (!profile.basisGates.empty()) {
            cudaq::opt::BasisConversionOptions basisOptions;
            basisOptions.basis.assign(profile.basisGates.begin(),
                                      profile.basisGates.end());
            pm.addPass(cudaq::opt::createBasisConversion(basisOptions));
          }
        };
    return mlir::success();
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
