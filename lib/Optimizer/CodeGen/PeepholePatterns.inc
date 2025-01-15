/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

namespace {

//===----------------------------------------------------------------------===//

// %1 = address_of @__quantum__qis__x__ctl
// %2 = call @invokewithControlBits %1, %ctrl, %targ
// ─────────────────────────────────────────────────
// %2 = call __quantum__qis__cnot %ctrl, %targ
struct XCtrlOneTargetToCNot : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = call.getCallee();
    if (!callee)
      return failure();
    auto args = call.getOperands();
    if (!callToInvokeWithXCtrlOneTarget(*callee, args))
      return failure();
    auto *ctx = rewriter.getContext();
    auto funcSymbol = FlatSymbolRefAttr::get(ctx, cudaq::opt::QIRCnot);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        call, TypeRange{}, funcSymbol, args.drop_front(2),
        call.getFastmathFlagsAttr(), call.getBranchWeightsAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//

// %4 = address_of @__quantum__cis__*
// ────────────────────────────────────────
// %4 = address_of @__quantum__cis__*__body
struct AddrOfCisToBase : public OpRewritePattern<LLVM::AddressOfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AddressOfOp addr,
                                PatternRewriter &rewriter) const override {
    auto global = addr.getGlobalName();
    if (!needsToBeRenamed(global))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(addr, addr.getType(),
                                                   global.str() + "__body");
    return success();
  }
};

//===----------------------------------------------------------------------===//

// This rule does not apply to measurements.
//
// %4 = call @__quantum__cis__*
// ──────────────────────────────────
// %4 = call @__quantum__cis__*__body
struct CalleeConv : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = call.getCallee();
    if (!callee)
      return failure();
    if (!needsToBeRenamed(*callee) ||
        callee->startswith(cudaq::opt::QIRMeasure))
      return failure();
    auto *ctx = rewriter.getContext();
    auto symbol = FlatSymbolRefAttr::get(ctx, callee->str() + "__body");
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        call, TypeRange{}, symbol, call.getOperands(),
        call.getFastmathFlagsAttr(), call.getBranchWeightsAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//

// Manually erase dead calls to QIRArrayGetElementPtr1d.
struct EraseDeadArrayGEP : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = call.getCallee();
    if (!callee)
      return failure();
    if (*callee != cudaq::opt::QIRArrayGetElementPtr1d)
      return failure();
    if (!call->use_empty())
      return failure();
    rewriter.eraseOp(call);
    return success();
  }
};

//===----------------------------------------------------------------------===//

// Replace the call with a dead op to DCE.
//
// %0 = call @allocate ... : ... -> T*
// ───────────────────────────────────
// %0 = undef : T*
struct EraseArrayAlloc : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = call.getCallee();
    if (!callee)
      return failure();
    if (*callee != cudaq::opt::QIRArrayQubitAllocateArray)
      return failure();
    auto *ctx = rewriter.getContext();
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(call,
                                               cudaq::opt::getArrayType(ctx));
    return success();
  }
};

//===----------------------------------------------------------------------===//

// Remove the release calls. This removes both array allocations as well as
// qubit singletons.
//
// call @release %5 : (!Qubit) -> ()
// ─────────────────────────────────
//
struct EraseArrayRelease : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = call.getCallee();
    if (!callee)
      return failure();
    if (*callee != cudaq::opt::QIRArrayQubitReleaseArray &&
        *callee != cudaq::opt::QIRArrayQubitReleaseQubit)
      return failure();
    rewriter.eraseOp(call);
    return success();
  }
};

//===----------------------------------------------------------------------===//

// %result = call @__quantum__qis__mz(%qbit) : (!Qubit) -> i1
// ──────────────────────────────────────────────────────────────
// call @__quantum__qis__mz_body(%qbit, %result) : (Q*, R*) -> ()
struct MeasureCallConv : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = call.getCallee();
    if (!callee)
      return failure();
    auto args = call.getOperands();
    if (*callee != cudaq::opt::QIRMeasure)
      return failure();
    auto inttoptr = args[0].getDefiningOp<LLVM::IntToPtrOp>();
    if (!inttoptr)
      return failure();
    rewriter.replaceOp(call,
                       createMeasureCall(rewriter, call.getLoc(), call, args));
    return success();
  }
};

//===----------------------------------------------------------------------===//

// %result = call @__quantum__qis__mz__to__register(%qbit, i8) : (!Qubit) -> i1
// ────────────────────────────────────────────────────────────────────────────
// call @__quantum__qis__mz_body(%qbit, %result) : (Q*, R*) -> ()
struct MeasureToRegisterCallConv : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = call.getCallee();
    if (!callee)
      return failure();
    auto args = call.getOperands();
    if (*callee != cudaq::opt::QIRMeasureToRegister)
      return failure();
    auto inttoptr = args[0].getDefiningOp<LLVM::IntToPtrOp>();
    if (!inttoptr)
      return failure();
    rewriter.replaceOp(call,
                       createMeasureCall(rewriter, call.getLoc(), call, args));
    return success();
  }
};

//===----------------------------------------------------------------------===//

// %1 = llvm.constant 1
// %2 = llvm.inttoptr %1 : i64 -> Result*
// %3 = llvm.bitcast %2 : Result* -> i1*
// %4 = llvm.load %3
// ─────────────────────────────────────
// %4 = call @read_result %2
struct LoadMeasureResult : public OpRewritePattern<LLVM::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp load,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto bitcast = load.getAddr().getDefiningOp<LLVM::BitcastOp>();
    if (!bitcast)
      return failure();
    auto inttoptr = bitcast.getArg().getDefiningOp<LLVM::IntToPtrOp>();
    if (!inttoptr)
      return failure();
    auto conint = inttoptr.getArg().getDefiningOp<LLVM::ConstantOp>();
    if (!conint)
      return failure();
    if (bitcast.getType() !=
        cudaq::opt::factory::getPointerType(IntegerType::get(ctx, 1)))
      return failure();
    if (inttoptr.getType() != cudaq::opt::getResultType(ctx))
      return failure();
    if (!isa<IntegerAttr>(conint.getValue()))
      return failure();

    rewriter.replaceOp(load, createReadResultCall(rewriter, load.getLoc(),
                                                  inttoptr.getResult()));
    return success();
  }
};

} // namespace
