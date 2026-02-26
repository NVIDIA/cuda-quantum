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
public:
  using OpRewritePattern::OpRewritePattern;

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

    if (devFunc->getAttr(cudaq::autoDeviceCallAttrName) &&
        devFunc.isDeclaration()) {
      // This is an auto-generated device call, e.g., arbitrary functions
      // provided by remote hardware providers
      // In the JIT pipeline, the device call name will be sent on to the
      // provider, which will resolve it to the actual function to call on the
      // service side.

      // This pass, which may be executed as part of the AOT pipeline, will
      // convert this to a runtime trap (unreachable) since we don't expect to
      // run this locally. This allows us to fully lower the code to LLVM IR as
      // we don't have the actual device function definition available.
      // Note: the generated code should never be executed unless there is a bug
      // in the kernel launch.

      // (1) Create a trap function that has the same signature as the device
      // function.
      auto insPt = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(module.getBody());
      auto trapFunc = rewriter.create<func::FuncOp>(
          devcall.getLoc(), devFuncName, devFunc.getFunctionType());
      trapFunc.setPrivate();
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
      rewriter.replaceOpWithNewOp<func::CallOp>(
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

// Remove auto-generated device call declarations that are not resolved.
// We have already replaced all resolved auto-generated device calls with traps.
class RemoveAutoGenDeviceCallDecl : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    if (func->getAttr(cudaq::autoDeviceCallAttrName) && func.isDeclaration()) {
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
    patterns.insert<QIRVendorDeviceCallPat>(ctx);
    patterns.add<RemoveAutoGenDeviceCallDecl>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
    return;
  }
};
} // namespace
