/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_EXPANDCONTROLNEGATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

/// Replace any operations with negative controls with the same operation with
/// negative controls and the addition of X operations on each control qubit
/// before and after the operation.
template <typename Op>
class ReplaceNegativeControl : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto negations = op.getNegatedQubitControls();
    if (!negations.has_value())
      return failure();

    // Process the negated controls, flipping each one to a negated state.
    auto *ctx = rewriter.getContext();
    auto ctrlTy = cudaq::quake::ControlType::get(ctx);
    SmallVector<Value> originalControls{op.getControls().begin(),
                                        op.getControls().end()};
    SmallVector<Value> newControls{originalControls.size()};
    for (auto negationIter : llvm::enumerate(negations.value())) {
      auto i = negationIter.index();
      Type ty = originalControls[i].getType();
      // Is the i-th control negated?
      if (ty == ctrlTy) {
        // We cannot process !quake.control types here. Run the linear-ctrl-form
        // pass first.
        return failure();
      }
      if (cudaq::quake::isLinearType(ty)) {
        // Quantum value types are *explicitly* threaded.
        if (negationIter.value()) {
          newControls[i] =
              cudaq::quake::XOp::create(rewriter, loc, TypeRange{ty},
                                        UnitAttr{}, ValueRange{}, ValueRange{},
                                        ValueRange{originalControls[i]}, {})
                  .getResult(0);
        } else {
          newControls[i] = originalControls[i];
        }
      } else {
        if (negationIter.value())
          cudaq::quake::XOp::create(rewriter, loc, ValueRange{},
                                    ValueRange{originalControls[i]});
        newControls[i] = originalControls[i];
      }
    }

    // Create a new op to erase the negated controls attribute.
    Op newOp;
    if constexpr (std::is_same_v<Op, cudaq::quake::ExpPauliOp>) {
      newOp = cudaq::quake::ExpPauliOp::create(
          rewriter, loc, op->getResultTypes(), op.getIsAdjAttr(),
          op.getParameters(), newControls, op.getTargets(),
          /*negatedControls=*/{}, op.getPauli(), op.getPauliLiteralAttr());
    } else if constexpr (std::is_same_v<Op,
                                        cudaq::quake::CustomUnitaryCallOp>) {
      newOp = cudaq::quake::CustomUnitaryCallOp::create(
          rewriter, loc, op->getResultTypes(), op.getGeneratorAttr(),
          op.getIsAdj(), op.getParameters(), newControls, op.getTargets(),
          /*negatedConstrols=*/{});
    } else if constexpr (std::is_same_v<
                             Op, cudaq::quake::CustomUnitaryConstantOp>) {
      newOp = cudaq::quake::CustomUnitaryConstantOp::create(
          rewriter, loc, op->getResultTypes(), op.getMatrixAttr(),
          op.getIsAdj(), op.getParameters(), newControls, op.getTargets(),
          /*negatedControls=*/{});
    } else {
      newOp = Op::create(rewriter, loc, op->getResultTypes(), op.getIsAdj(),
                         op.getParameters(), newControls, op.getTargets(),
                         /*negatedControls=*/{});
    }

    // Process the negated controls, flipping each back to the original state.
    SmallVector<Value> newResults;
    SmallVector<Value> newOpResults{newOp.getResults().begin(),
                                    newOp.getResults().end()};
    unsigned j = 0;
    for (auto iter : llvm::enumerate(negations.value())) {
      auto i = iter.index();
      Type ty = newControls[i].getType();
      // Is the i-th control negated?
      if (cudaq::quake::isLinearType(ty)) {
        // Quantum value types are *explicitly* threaded.
        if (iter.value()) {
          newResults.push_back(
              cudaq::quake::XOp::create(
                  rewriter, loc, TypeRange{ty}, /*is_adj=*/UnitAttr{},
                  /*params=*/ValueRange{}, /*controls=*/ValueRange{},
                  ValueRange{newOpResults[j]}, /*negatedControls=*/{})
                  .getResult(0));
        } else {
          newResults.push_back(newOpResults[j]);
        }
        ++j;
      } else {
        if (iter.value())
          cudaq::quake::XOp::create(rewriter, loc, ValueRange{},
                                    ValueRange{originalControls[i]});
      }
    }

    // Collect up any target values as apropos.
    newResults.append(newOp->getResults().begin() + newResults.size(),
                      newOp->getResults().end());

    // Replace the old op with the new wires as apropos.
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

namespace {

struct ExpandControlNegationsPass
    : public cudaq::opt::impl::ExpandControlNegationsBase<
          ExpandControlNegationsPass> {
  using ExpandControlNegationsBase::ExpandControlNegationsBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns
        .insert<ReplaceNegativeControl<cudaq::quake::XOp>,
                ReplaceNegativeControl<cudaq::quake::YOp>,
                ReplaceNegativeControl<cudaq::quake::ZOp>,
                ReplaceNegativeControl<cudaq::quake::HOp>,
                ReplaceNegativeControl<cudaq::quake::SOp>,
                ReplaceNegativeControl<cudaq::quake::TOp>,
                ReplaceNegativeControl<cudaq::quake::RxOp>,
                ReplaceNegativeControl<cudaq::quake::RyOp>,
                ReplaceNegativeControl<cudaq::quake::RzOp>,
                ReplaceNegativeControl<cudaq::quake::R1Op>,
                ReplaceNegativeControl<cudaq::quake::U3Op>,
                ReplaceNegativeControl<cudaq::quake::SwapOp>,
                ReplaceNegativeControl<cudaq::quake::ExpPauliOp>,
                ReplaceNegativeControl<cudaq::quake::CustomUnitaryCallOp>,
                ReplaceNegativeControl<cudaq::quake::CustomUnitaryConstantOp>>(
            ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<cudaq::cc::CCDialect, arith::ArithDialect,
                           LLVM::LLVMDialect>();
    target.addDynamicallyLegalDialect<cudaq::quake::QuakeDialect>(
        [](Operation *op) {
          auto quantumOp = dyn_cast<cudaq::quake::OperatorInterface>(op);
          if (!quantumOp)
            return true;

          auto negations = quantumOp.getNegatedControls();
          if (!negations.has_value())
            return true;

          for (auto negation : negations.value())
            if (negation)
              return false;

          return true;
        });
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp->emitOpError("could not replace negations");
      signalPassFailure();
    }
  }
};
} // namespace
