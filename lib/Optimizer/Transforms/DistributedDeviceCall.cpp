/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_DISTRIBUTEDDEVICECALL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "distributed-device-call"

using namespace mlir;

namespace {

class QIRVendorDeviceCallPat
    : public OpRewritePattern<cudaq::cc::DeviceCallOp> {
  bool insertTrapImplementation;

public:
  using OpRewritePattern::OpRewritePattern;
  static constexpr const char TrapFuncAttr[] = "cudaq-trap-device-call-impl";

  QIRVendorDeviceCallPat(MLIRContext *context, bool insertTrapImpl)
      : OpRewritePattern(context), insertTrapImplementation(insertTrapImpl) {}

  LogicalResult matchAndRewrite(cudaq::cc::DeviceCallOp devcall,
                                PatternRewriter &rewriter) const override {
    constexpr const char PassthroughAttr[] = "passthrough";
    constexpr const char QIRVendorAttr[] = "cudaq-fnid";
    auto module = devcall->getParentOfType<ModuleOp>();
    auto devFuncName = devcall.getCallee();
    auto devFunc = module.lookupSymbol<func::FuncOp>(devFuncName);
    if (!devFunc) {
      LLVM_DEBUG(llvm::dbgs() << "cannot find the function " << devFuncName
                              << " in module\n");
      return failure();
    }

    llvm::MD5 hash;
    hash.update(devFuncName);
    llvm::MD5::MD5Result result;
    hash.final(result);
    std::uint32_t callbackCode = result.low();

    if (insertTrapImplementation && devFunc.isDeclaration()) {
      // If `insertTrapImplementation` is enabled (e.g., AOT compilation for
      // remote hardware providers), we want to insert a trap implementation for
      // any unresolved device function (declaration only), so that we can
      // perform AOT compilation without needing the actual device function
      // definitions. This trap function will never be executed as the remote
      // JIT pipeline would not be using the `device_call` functions anyway.
      // Rather, these functions will only be resolved at runtime by the remote
      // provider's runtime library.

      // Add an attribute to the declaration to indicate that this function is
      // an unresolved device function and the trap implementation is inserted
      // for it. We will use this attribute to identify and remove these
      // declarations later.
      devFunc->setAttr(TrapFuncAttr, rewriter.getUnitAttr());
      // (1) Create a trap function that has the same signature as the device
      // function.
      auto insPt = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(module.getBody());
      auto trapFunc = rewriter.create<func::FuncOp>(
          devcall.getLoc(), devFuncName, devFunc.getFunctionType());
      trapFunc.setPrivate();
      // Set weak_odr linkage to allow multiple definitions across translation
      // units without linker errors. e.g., compiling for a remote hardware
      // provider with the actual device call library linkage (even though
      // unused) should not cause any problems.
      auto weakOdrLinkage = mlir::LLVM::linkage::Linkage::WeakODR;
      auto linkage =
          mlir::LLVM::LinkageAttr::get(rewriter.getContext(), weakOdrLinkage);
      trapFunc->setAttr("llvm.linkage", linkage);
      auto &entryBlock = *trapFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(&entryBlock);
      // Create a call to the trap intrinsic.
      // Error code 2 is used to indicate illegal execution of unreachable code.
      Value errorCodeTwo =
          rewriter.create<arith::ConstantIntOp>(devcall.getLoc(), 2, 64);
      rewriter.create<func::CallOp>(devcall.getLoc(), TypeRange{},
                                    cudaq::opt::QISTrap,
                                    ValueRange{errorCodeTwo});
      // For return (after the trap), load from nullptr to create return value
      // of the same type as the device function, i.e., `return *(T*)nullptr;`
      // for return type `T`.
      // Note: this will never be executed because of the trap above.
      SmallVector<Value> trapResults;
      for (Type resTy : devFunc.getFunctionType().getResults()) {
        auto nullPtr = rewriter.create<arith::ConstantOp>(
            devcall.getLoc(),
            rewriter.getZeroAttr(rewriter.getIntegerType(64)));
        auto ptrTy = cudaq::cc::PointerType::get(resTy);
        auto castedNullPtr = rewriter.create<cudaq::cc::CastOp>(
            devcall.getLoc(), ptrTy, nullPtr);
        auto loadedVal =
            rewriter.create<cudaq::cc::LoadOp>(devcall.getLoc(), castedNullPtr);
        trapResults.push_back(loadedVal);
      }

      rewriter.create<func::ReturnOp>(devcall.getLoc(), trapResults);
      rewriter.restoreInsertionPoint(insPt);

      // (2) Replace the device call with a call to the trap function.
      // Prevent inlining of the trap function.
      rewriter.replaceOpWithNewOp<cudaq::cc::NoInlineCallOp>(
          devcall, trapFunc.getFunctionType().getResults(),
          trapFunc.getSymNameAttr(), devcall.getArgs());
      return success();
    }

    bool needToAddIt = true;
    SmallVector<Attribute> funcIdAttr;
    if (auto passthruAttr = devFunc->getAttr(PassthroughAttr)) {
      auto arrayAttr = cast<ArrayAttr>(passthruAttr);
      funcIdAttr.append(arrayAttr.begin(), arrayAttr.end());
      for (auto a : arrayAttr) {
        if (auto strArrAttr = dyn_cast<ArrayAttr>(a)) {
          auto strAttr = dyn_cast<StringAttr>(strArrAttr[0]);
          if (!strAttr)
            continue;
          if (strAttr.getValue() == QIRVendorAttr) {
            needToAddIt = false;
            break;
          }
        }
      }
    }
    if (needToAddIt) {
      auto callbackCodeAsStr = std::to_string(callbackCode);
      funcIdAttr.push_back(rewriter.getStrArrayAttr(
          {QIRVendorAttr, rewriter.getStringAttr(callbackCodeAsStr)}));
      devFunc->setAttr(PassthroughAttr, rewriter.getArrayAttr(funcIdAttr));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        devcall, devFunc.getFunctionType().getResults(), devFuncName,
        devcall.getArgs());
    return success();
  }
};

class ResolveDevicePtrOpPat
    : public OpRewritePattern<cudaq::cc::ResolveDevicePtrOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::ResolveDevicePtrOp resolve,
                                PatternRewriter &rewriter) const override {
    auto loc = resolve.getLoc();
    auto call = rewriter.create<func::CallOp>(
        loc, TypeRange{cudaq::cc::PointerType::get(rewriter.getI8Type())},
        cudaq::runtime::extractDevPtr, ValueRange{resolve.getDevicePtr()});
    rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(
        resolve, resolve.getResult().getType(), call.getResult(0));
    return success();
  }
};

// Remove device call declarations that we have already inserted trap
// implementations for, to avoid duplicate declarations of the same device
// function.
class RemoveDuplicateDeviceCallDeclaration
    : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    if (func.isDeclaration() &&
        func->hasAttr(QIRVendorDeviceCallPat::TrapFuncAttr)) {
      rewriter.eraseOp(func);
      return success();
    }

    return failure();
  }
};

class DistributedDeviceCallPass
    : public cudaq::opt::impl::DistributedDeviceCallBase<
          DistributedDeviceCallPass> {
public:
  using DistributedDeviceCallBase::DistributedDeviceCallBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ModuleOp module = getOperation();
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::runtime::extractDevPtr))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::CudaqRegisterCallbackName);
      return;
    }

    if (failed(irBuilder.loadIntrinsic(module, cudaq::opt::QISTrap))) {
      module.emitError("could not load QIR trap function.");
      signalPassFailure();
      return;
    }

    patterns.add<ResolveDevicePtrOpPat>(ctx);
    patterns.insert<QIRVendorDeviceCallPat>(ctx, insertTrapImplementation);
    if (insertTrapImplementation)
      patterns.add<RemoveDuplicateDeviceCallDeclaration>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
    return;
  }
};
} // namespace
