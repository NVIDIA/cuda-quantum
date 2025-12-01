/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Support/Plugin.h"

using namespace mlir;
using namespace cudaq;

namespace {

struct ReplaceDeviceCall : public OpRewritePattern<cc::DeviceCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cc::DeviceCallOp deviceCallOp,
                                PatternRewriter &rewriter) const override {
    Location loc = deviceCallOp.getLoc();
    auto *ctx = rewriter.getContext();

    auto parentModule = deviceCallOp->getParentOfType<ModuleOp>();

    // Declare the intrinsic function if it doesn't exist
    func::FuncOp intrinsicFunc = parentModule.lookupSymbol<func::FuncOp>(
        "__nvqlink_device_call_dispatch");
    if (!intrinsicFunc) {
      // Create function type for the intrinsic
      auto i64Type = rewriter.getI64Type();
      auto i8Type = rewriter.getI8Type();
      auto voidPtrType = cc::PointerType::get(i8Type);
      auto charPtrType = cc::PointerType::get(i8Type);
      auto voidPtrPtrType = cc::PointerType::get(voidPtrType);
      auto i64PtrType = cc::PointerType::get(i64Type);

      auto intrinsicFuncType = rewriter.getFunctionType(
          {i64Type, charPtrType, i64Type, voidPtrPtrType, i64PtrType,
           voidPtrType, i64Type},
          {});

      // Insert the function declaration at the module level
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());

      intrinsicFunc = rewriter.create<func::FuncOp>(
          loc, "__nvqlink_device_call_dispatch", intrinsicFuncType);
      intrinsicFunc.setPrivate();
    }

    // Get the original arguments and result type
    ValueRange args = deviceCallOp.getArgs();
    Type resultType = deviceCallOp.getResultTypes()[0];

    // Get function symbol name
    StringRef callbackName = deviceCallOp.getCallee();
    std::size_t stringLength = callbackName.size() + 1;

    // Get device ID (default to 0 if not specified)
    Value deviceId;
    if (deviceCallOp.getDevice()) {
      deviceId = deviceCallOp.getDevice();
    } else {
      deviceId = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    }

    // Create types needed for the intrinsic function
    auto i64Type = rewriter.getI64Type();
    auto i8Type = rewriter.getI8Type();
    auto voidPtrType = cc::PointerType::get(i8Type);
    auto charPtrType = cc::PointerType::get(i8Type);
    auto stringArrayType = cc::ArrayType::get(ctx, i8Type, stringLength);
    auto stringPtrType = cc::PointerType::get(stringArrayType);

    // Create string literal for callback name
    Value callbackNameLiteral = rewriter.create<cc::CreateStringLiteralOp>(
        loc, stringPtrType, rewriter.getStringAttr(callbackName));
    Value callbackNameStr =
        rewriter.create<cc::CastOp>(loc, charPtrType, callbackNameLiteral);

    // Create number of arguments constant
    Value numArgs = rewriter.create<arith::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(args.size()));

    // Allocate arrays for argument pointers and sizes
    auto voidPtrArrayType = cc::ArrayType::get(ctx, voidPtrType, args.size());
    auto i64ArrayType = cc::ArrayType::get(ctx, i64Type, args.size());

    Value argsPtrArray = rewriter.create<cc::AllocaOp>(loc, voidPtrArrayType);
    Value argsSizeArray = rewriter.create<cc::AllocaOp>(loc, i64ArrayType);

    // Allocate storage for each argument and populate arrays
    for (size_t i = 0; i < args.size(); ++i) {
      Value arg = args[i];
      Type argType = arg.getType();

      // Allocate storage for this argument
      Value argAlloca = rewriter.create<cc::AllocaOp>(loc, argType);
      rewriter.create<cc::StoreOp>(loc, arg, argAlloca);

      // Cast to void pointer
      Value argPtr = rewriter.create<cc::CastOp>(loc, voidPtrType, argAlloca);

      // Compute pointer to array element for storing pointer
      Value ptrIndex = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(i));
      Value argsPtrArrayIdx = rewriter.create<cc::ComputePtrOp>(
          loc, cc::PointerType::get(voidPtrType), argsPtrArray,
          ValueRange{ptrIndex});
      rewriter.create<cc::StoreOp>(loc, argPtr, argsPtrArrayIdx);

      // Get size of argument type
      Value argSize = rewriter.create<cc::SizeOfOp>(loc, i64Type, argType);

      // Compute pointer to array element for storing size
      Value argsSizeArrayIdx = rewriter.create<cc::ComputePtrOp>(
          loc, cc::PointerType::get(i64Type), argsSizeArray,
          ValueRange{ptrIndex});
      rewriter.create<cc::StoreOp>(loc, argSize, argsSizeArrayIdx);
    }

    // Allocate result storage
    Value resultAlloca = rewriter.create<cc::AllocaOp>(loc, resultType);
    Value resultPtr =
        rewriter.create<cc::CastOp>(loc, voidPtrType, resultAlloca);
    Value resultSize = rewriter.create<cc::SizeOfOp>(loc, i64Type, resultType);

    // Get pointers to the arrays (cast to void**)
    Value argsPtrArrayPtr = rewriter.create<cc::CastOp>(
        loc, cc::PointerType::get(voidPtrType), argsPtrArray);
    Value argsSizeArrayPtr = rewriter.create<cc::CastOp>(
        loc, cc::PointerType::get(i64Type), argsSizeArray);

    // Create the intrinsic function call
    // void __nvqlink_device_call_dispatch(std::size_t device_id,
    //                                    const char *callbackName,
    //                                    std::size_t num_args, void **args_vec,
    //                                    std::size_t *args_sizes, void *result,
    //                                    std::size_t result_size)
    rewriter.create<func::CallOp>(
        loc, "__nvqlink_device_call_dispatch", TypeRange{},
        ValueRange{deviceId, callbackNameStr, numArgs, argsPtrArrayPtr,
                   argsSizeArrayPtr, resultPtr, resultSize});

    // Replace the original operation
    rewriter.replaceOpWithNewOp<cc::LoadOp>(deviceCallOp, resultAlloca);
    parentModule.dump();
    return success();
  }
};

/// Custom pass implementation
class DeviceCallShmem
    : public PassWrapper<DeviceCallShmem, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeviceCallShmem)

  StringRef getArgument() const override { return "device-call-shmem"; }

  StringRef getDescription() const override { return ""; }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceDeviceCall>(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<cc::CCDialect, func::FuncDialect,
                           arith::ArithDialect>();
    target.addIllegalOp<cc::DeviceCallOp>();

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitOpError("device call simple failed");
      signalPassFailure();
    }
  }
};

} // anonymous namespace

// Register the plugin with CUDA-Q
CUDAQ_REGISTER_MLIR_PASS(DeviceCallShmem)
