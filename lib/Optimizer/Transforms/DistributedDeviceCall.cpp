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
#include "llvm/Support/MD5.h"
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

      // (1) Add a trap implementation for this device function declaration.
      {
        OpBuilder::InsertionGuard guard(rewriter);
        // Add an entry block
        auto &entryBlock = *devFunc.addEntryBlock();
        rewriter.setInsertionPointToStart(&entryBlock);
        // Create a call to the trap intrinsic.
        // Error code 2 is used to indicate illegal execution of unreachable
        // code.
        Value errorCodeTwo =
            arith::ConstantIntOp::create(rewriter, devcall.getLoc(), 2, 64);
        func::CallOp::create(rewriter, devcall.getLoc(), TypeRange{},
                             cudaq::opt::QISTrap, ValueRange{errorCodeTwo});
        // For return (after the trap), load from nullptr to create return value
        // of the same type as the device function, i.e., `return *(T*)nullptr;`
        // for return type `T`.
        // Note: this will never be executed because of the trap above. It's
        // only to create a valid IR with the correct return type for the
        // function.
        SmallVector<Value> trapResults;
        for (Type resTy : devFunc.getFunctionType().getResults()) {
          auto nullPtr = arith::ConstantOp::create(
              rewriter, devcall.getLoc(),
              rewriter.getZeroAttr(rewriter.getIntegerType(64)));
          auto ptrTy = cudaq::cc::PointerType::get(resTy);
          auto castedNullPtr = cudaq::cc::CastOp::create(
              rewriter, devcall.getLoc(), ptrTy, nullPtr);
          auto loadedVal = cudaq::cc::LoadOp::create(rewriter, devcall.getLoc(),
                                                     castedNullPtr);
          trapResults.push_back(loadedVal);
        }

        func::ReturnOp::create(rewriter, devcall.getLoc(), trapResults);
      }
      // (2) Set this trap function as private and weak_odr linkage, to allow
      // multiple definitions across translation units without linker errors.
      // For example, compiling for a remote hardware provider with the actual
      // device call library linkage (even though unused) should not cause any
      // problems.
      devFunc.setPrivate();
      auto weakOdrLinkage = mlir::LLVM::linkage::Linkage::WeakODR;
      auto linkage =
          mlir::LLVM::LinkageAttr::get(rewriter.getContext(), weakOdrLinkage);
      devFunc->setAttr("llvm.linkage", linkage);

      // (3) Replace the device call with a no-inline call to prevent inlining
      // of the trap function.
      // We use a no-inline call here to ensure that the call to the device
      // function is preserved as a call in the IR (even in the presence of the
      // trap implementation). If the actual implementation is provided at link
      // time, it will be used instead of the trap implementation due to the
      // weak_odr linkage.
      rewriter.replaceOpWithNewOp<cudaq::cc::NoInlineCallOp>(
          devcall, devFunc.getFunctionType().getResults(), devFuncName,
          devcall.getArgs(), ArrayAttr{}, ArrayAttr{});

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
    auto call = func::CallOp::create(
        rewriter, loc,
        TypeRange{cudaq::cc::PointerType::get(rewriter.getI8Type())},
        cudaq::runtime::extractDevPtr, ValueRange{resolve.getDevicePtr()});
    rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(
        resolve, resolve.getResult().getType(), call.getResult(0));
    return success();
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
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
    return;
  }
};
} // namespace
