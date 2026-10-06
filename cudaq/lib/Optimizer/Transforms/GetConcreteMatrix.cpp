/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
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
    // Collect all calls before walking any generator body. Replacing a call
    // cannot create another candidate, so this snapshot is complete.
    llvm::SmallMapVector<func::FuncOp,
                         SmallVector<cudaq::quake::CustomUnitaryCallOp>, 4>
        generatorWorklist;
    module.walk<WalkOrder::PreOrder>(
        [&](cudaq::quake::CustomUnitaryCallOp customOp) {
          // Resolve the generator from the call's nearest module. This matters
          // when the pass root contains nested modules with their own symbol
          // tables.
          auto parentModule = customOp->getParentOfType<ModuleOp>();
          auto generator =
              parentModule.lookupSymbol<func::FuncOp>(customOp.getGenerator());
          if (!generator)
            return;

          generatorWorklist[generator].push_back(customOp);
        });
    if (generatorWorklist.empty())
      return;

    IRRewriter rewriter(&getContext());
    for (auto &[generator, calls] : generatorWorklist) {
      rewriter.modifyOpInPlace(generator, [&] { generator.setPrivate(); });
      StringRef concreteMatrix;
      SmallVector<cudaq::cc::AddressOfOp> addresses;
      generator.walk([&](cudaq::cc::AddressOfOp address) {
        concreteMatrix = address.getGlobalName();
        addresses.push_back(address);
      });

      bool hasConvertedCall = false;
      for (cudaq::quake::CustomUnitaryCallOp customOp : calls) {
        if (concreteMatrix.empty()) {
          customOp.emitError(
              "Constant matrix corresponding to custom operation's generator "
              "function not found in the module.");
          continue;
        }

        auto parentModule = customOp->getParentOfType<ModuleOp>();
        if (!parentModule.lookupSymbol<cudaq::cc::GlobalOp>(concreteMatrix))
          continue;

        rewriter.setInsertionPoint(customOp);
        rewriter.replaceOpWithNewOp<cudaq::quake::CustomUnitaryConstantOp>(
            customOp,
            FlatSymbolRefAttr::get(parentModule.getContext(), concreteMatrix),
            customOp.getIsAdj(), customOp.getParameters(),
            customOp.getControls(), customOp.getTargets(),
            customOp.getNegatedQubitControlsAttr());
        hasConvertedCall = true;
      }

      // A generator may be shared by several calls. Wait until every call has
      // been handled before removing matrix addresses that no longer have
      // users.
      if (hasConvertedCall)
        for (cudaq::cc::AddressOfOp address : addresses)
          if (address->use_empty())
            rewriter.eraseOp(address);
    }
  }
};

} // namespace
