/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_GETCONCRETEMATRIX
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "get-concrete-matrix"

using namespace mlir;

namespace {
class GetConcreteMatrixPass
    : public cudaq::opt::impl::GetConcreteMatrixBase<GetConcreteMatrixPass> {
public:
  using GetConcreteMatrixBase::GetConcreteMatrixBase;

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<Operation *> candidates;
    module.walk<WalkOrder::PreOrder>(
        [&](cudaq::quake::CustomUnitaryCallOp customOp) {
          candidates.push_back(customOp.getOperation());
        });
    if (candidates.empty())
      return;

    IRRewriter rewriter(&getContext());
    llvm::SmallPtrSet<Operation *, 4> convertedGenerators;
    for (Operation *candidate : candidates) {
      auto customOp = cast<cudaq::quake::CustomUnitaryCallOp>(candidate);
      auto parentModule = customOp->getParentOfType<ModuleOp>();
      auto generator =
          parentModule.lookupSymbol<func::FuncOp>(customOp.getGenerator());
      if (!generator)
        continue;

      rewriter.modifyOpInPlace(generator, [&] { generator.setPrivate(); });
      StringRef concreteMatrix;
      generator.walk([&](cudaq::cc::AddressOfOp address) {
        concreteMatrix = address.getGlobalName();
      });
      if (concreteMatrix.empty()) {
        customOp.emitError(
            "Constant matrix corresponding to custom operation's generator "
            "function not found in the module.");
        continue;
      }
      if (!parentModule.lookupSymbol<cudaq::cc::GlobalOp>(concreteMatrix))
        continue;

      rewriter.setInsertionPoint(customOp);
      rewriter.replaceOpWithNewOp<cudaq::quake::CustomUnitaryConstantOp>(
          customOp,
          FlatSymbolRefAttr::get(parentModule.getContext(), concreteMatrix),
          customOp.getIsAdj(), customOp.getParameters(), customOp.getControls(),
          customOp.getTargets(), customOp.getNegatedQubitControlsAttr());
      convertedGenerators.insert(generator.getOperation());
    }

    SmallVector<Operation *> deadAddresses;
    for (Operation *generator : convertedGenerators)
      generator->walk([&](cudaq::cc::AddressOfOp address) {
        if (address->use_empty())
          deadAddresses.push_back(address.getOperation());
      });
    for (Operation *address : deadAddresses)
      rewriter.eraseOp(address);
  }
};

} // namespace
