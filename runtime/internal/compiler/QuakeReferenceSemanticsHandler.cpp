/*****************************************************************- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

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
                const cudaq::config::Profile &output) const override {
    const auto *referenceProfile =
        std::get_if<cudaq::config::QuakeReferenceSemanticsProfile>(&output);
    if (!referenceProfile ||
        (!std::holds_alternative<cudaq::config::UncheckedProfile>(input) &&
         !std::holds_alternative<cudaq::config::QuakeReferenceSemanticsProfile>(
             input)))
      return false;
    return true;
  }

  mlir::LogicalResult
  addStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options,
                 cudaq_internal::compiler::StagePasses &passes) const override {
    if (!supports(input, output))
      return mlir::failure();
    const auto &referenceProfile =
        std::get<cudaq::config::QuakeReferenceSemanticsProfile>(output);

    // Single-qubit measurements and `reset-after-measure` both need
    // `expand-measurements`.
    if (referenceProfile.measurementType ==
            cudaq::config::MeasurementForm::single_qubit ||
        referenceProfile.resetAfterMeasure)
      passes.preDeployPasses = [](mlir::OpPassManager &pm) {
        pm.addPass(cudaq::opt::createExpandMeasurementsPass());
      };

    if (!referenceProfile.basisGates.empty())
      passes.postDeployPasses =
          [basis = referenceProfile.basisGates](mlir::OpPassManager &pm) {
            cudaq::opt::BasisConversionOptions basisOptions;
            basisOptions.basis.assign(basis.begin(), basis.end());
            pm.addPass(cudaq::opt::createBasisConversion(basisOptions));
          };

    if (referenceProfile.resetAfterMeasure)
      passes.postFinalizePasses = [](mlir::OpPassManager &pm) {
        auto &funcPM = pm.nest<mlir::func::FuncOp>();
        funcPM.addPass(cudaq::opt::createCombineQuantumAllocations());
        funcPM.addPass(cudaq::opt::createQubitResetBeforeReuse());
      };

    return mlir::success();
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeReferenceSemanticsHandler() {
  return std::make_unique<QuakeReferenceSemanticsHandler>();
}
