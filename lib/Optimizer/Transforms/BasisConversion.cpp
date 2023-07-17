/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_BASISCONVERSIONPASS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {

struct BasisTarget : public ConversionTarget {
  struct OperatorInfo {
    StringRef name;
    size_t numControls;
  };

  BasisTarget(MLIRContext &context, ArrayRef<std::string> targetBasis)
      : ConversionTarget(context) {
    constexpr size_t unbounded = std::numeric_limits<size_t>::max();

    // Parse the list of target operations and build a set of legal operations
    for (const std::string &targetInfo : targetBasis) {
      StringRef option = targetInfo;
      auto nameEnd = option.find_first_of('(');
      auto name = option.take_front(nameEnd);
      if (nameEnd < option.size())
        option = option.drop_front(nameEnd);

      auto &info = legalOperatorSet.emplace_back(OperatorInfo{name, 0});
      if (option.consume_front("(")) {
        option = option.ltrim();
        if (option.consume_front("n"))
          info.numControls = unbounded;
        else
          option.consumeInteger(10, info.numControls);
        assert(option.trim().consume_front(")"));
      }
    }

    addLegalDialect<func::FuncDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, cudaq::cc::CCDialect>();
    addDynamicallyLegalDialect<quake::QuakeDialect>([&](Operation *op) {
      if (auto optor = dyn_cast<quake::OperatorInterface>(op)) {
        auto name = optor->getName().stripDialect();
        for (auto info : legalOperatorSet) {
          if (info.name != name)
            continue;
          if (info.numControls == unbounded ||
              optor.getControls().size() == info.numControls)
            return info.numControls == optor.getControls().size();
        }
        return false;
      }
      return true;
    });
  }

  SmallVector<OperatorInfo, 8> legalOperatorSet;
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct BasisConversion
    : public cudaq::opt::impl::BasisConversionPassBase<BasisConversion> {
  using BasisConversionPassBase::BasisConversionPassBase;

  void runOnOperation() override {
    auto module = getOperation();
    if (basis.empty()) {
      module.emitError("Basis conversion requires a target basis");
      signalPassFailure();
      return;
    }

    // First, we walk the whole module in search for controlled `quake.apply`
    // operations: If present, we conservatively don't do any decompostions. We
    // also collect quantum kernels.
    //
    // TODO: Evaluate if preventing decompostion when there is at least one
    // controlled `quake.apply` in the whole module is too convervative.
    SmallVector<Operation *, 16> kernels;
    auto walkResult = module.walk([&kernels](Operation *op) {
      // Check if it is a quantum kernel
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp->hasAttr(cudaq::entryPointAttrName)) {
          kernels.push_back(funcOp);
          return WalkResult::advance();
        }
        for (auto arg : funcOp.getArguments())
          if (isa<quake::RefType, quake::VeqType>(arg.getType())) {
            kernels.push_back(funcOp);
            return WalkResult::advance();
          }
        // Skip functions which are not quantum kernels
        return WalkResult::skip();
      }
      // Check if it is controlled quake.apply
      if (auto applyOp = dyn_cast<quake::ApplyOp>(op))
        if (!applyOp.getControls().empty())
          return WalkResult::interrupt();

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      module.emitError("Basis conversion doesn't work with `quake.apply`");
      signalPassFailure();
      return;
    }

    if (kernels.empty())
      return;

    // Setup target and patterns
    BasisTarget target(getContext(), basis);
    RewritePatternSet owningPatterns(&getContext());
    cudaq::populateWithAllDecompositionPatterns(owningPatterns);
    auto patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                            disabledPatterns, enabledPatterns);

    // Process kernels in parallel
    LogicalResult rewriteResult = failableParallelForEach(
        module.getContext(), kernels, [&target, &patterns](Operation *op) {
          return applyFullConversion(op, target, patterns);
        });

    if (failed(rewriteResult))
      signalPassFailure();
  }
};

} // namespace
