/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CompileOptions.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace {

/// The built-in handler from `quake-unmapped` to `quake-mapped`. It borrows
/// every qubit from a wire set, routes the kernel onto the profile's device,
/// and converts the swaps that routing adds to the basis.
class QuakeMappedHandler : public cudaq_internal::compiler::LoweringHandler {
public:
  bool supports(const cudaq::config::Profile &input,
                const cudaq::config::Profile &output) const override {
    return std::holds_alternative<cudaq::config::QuakeUnmappedProfile>(input) &&
           std::holds_alternative<cudaq::config::QuakeMappedProfile>(output);
  }

  mlir::LogicalResult
  addStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options,
                 cudaq_internal::compiler::StagePasses &passes) const override {
    if (!supports(input, output))
      return mlir::failure();
    passes.postDeployPasses =
        [profile = std::get<cudaq::config::QuakeMappedProfile>(output),
         level = static_cast<unsigned>(options.optimizationLevel)](
            mlir::OpPassManager &pm) { addPasses(pm, profile, level); };
    return mlir::success();
  }

  /// Borrowing every qubit needs each register's size, which often depends on
  /// an argument, whether or not the stage routes.
  bool
  requiresArgumentValues(const cudaq::config::Profile &input,
                         const cudaq::config::Profile &output) const override {
    return supports(input, output);
  }

private:
  static void addPasses(mlir::OpPassManager &pm,
                        const cudaq::config::QuakeMappedProfile &profile,
                        unsigned level) {
    // Accepts the references or wires that `quake-unmapped` allows.
    cudaq_internal::compiler::addConvertToWires(
        pm, cudaq::config::QuakeUnmappedProfile{}, /*addWireSet=*/true);
    // `raise-fatal-errors` makes a kernel that cannot be routed a compile
    // error instead of leaving it unrouted.
    // TODO: Search for a better layout with `placement=auto search=sabre` once
    // downstream code no longer assumes that mapped device qubits are
    // contiguous from 0 (NVIDIA/cuda-quantum#4753). A layout that is not the
    // identity also needs full-QIR codegen to allocate qubits by borrow
    // identity rather than in borrow order.
    if (profile.device) {
      cudaq::opt::MappingFuncOptions mappingOptions;
      mappingOptions.device = *profile.device;
      mappingOptions.nonComposable = true;
      cudaq::opt::addMappingPipeline(pm, mappingOptions);
    }
    // The level-1 cleanup that every cost model shares.
    if (level >= 1)
      cudaq::opt::addTwoQubitGateCountOptimizationPipeline(pm, /*level=*/1);
    if (!profile.basisGates.empty()) {
      cudaq::opt::BasisConversionOptions basisOptions;
      basisOptions.basis.assign(profile.basisGates.begin(),
                                profile.basisGates.end());
      pm.addPass(cudaq::opt::createBasisConversion(basisOptions));
    }
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeMappedHandler() {
  return std::make_unique<QuakeMappedHandler>();
}
