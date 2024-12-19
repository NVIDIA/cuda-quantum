/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "convert-to-qir-api"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKETOQIRAPI
#define GEN_PASS_DEF_QUAKETOQIRAPIPREP
#define GEN_PASS_DEF_QUAKETOQIRAPIFINAL
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

//===----------------------------------------------------------------------===//

static std::string getGateFunctionPrefix(Operation *op) {
  return cudaq::opt::QIRQISPrefix + op->getName().stripDialect().str();
}

/// Use modifier class classes to specialize the QIR API to a particular flavor
/// of QIR. For example, the names of the actual functions in "full QIR" are
/// different than the names used by the other API flavors.
namespace {

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

/// Type converter for converting quake dialect to one of the QIR APIs. This
/// class is used for conversions as well as instantiating QIR types in
/// conversion patterns.
struct QIRAPITypeConverter : public TypeConverter {
  QIRAPITypeConverter(MLIRContext *ctx, bool useOpaque)
      : ctx(ctx), useOpaque(useOpaque) {}

  Type getQubitType() { return cudaq::opt::getQubitType(ctx, useOpaque); }
  Type getArrayType() { return cudaq::opt::getArrayType(ctx, useOpaque); }
  Type getResultType() { return cudaq::opt::getResultType(ctx, useOpaque); }
  Type getCharPointerType() {
    return cudaq::opt::getCharPointerType(ctx, useOpaque);
  }

  MLIRContext *ctx;
  bool useOpaque;
};

inline Type getQubitType(TypeConverter *converter) {
  return static_cast<QIRAPITypeConverter *>(converter)->getQubitType();
}

inline Type getArrayType(TypeConverter *converter) {
  return static_cast<QIRAPITypeConverter *>(converter)->getArrayType();
}

inline Type getResultType(TypeConverter *converter) {
  return static_cast<QIRAPITypeConverter *>(converter)->getResultType();
}

inline Type getCharPointerType(TypeConverter *converter) {
  return static_cast<QIRAPITypeConverter *>(converter)->getCharPointerType();
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

template <typename M>
struct AllocaOpRewrite : public OpConversionPattern<quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::AllocaOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If this alloc is just returning a qubit
    if (auto resultType =
            dyn_cast_if_present<quake::RefType>(alloc.getType())) {

      // StringRef qirQubitAllocate = cudaq::opt::QIRQubitAllocate;
      StringRef qirQubitAllocate = M::getQIRQubitAllocate();
      Type qubitTy = getQubitType(getTypeConverter());

      rewriter.replaceOpWithNewOp<func::CallOp>(alloc, TypeRange{qubitTy},
                                                qirQubitAllocate, ValueRange{});
      return success();
    }

    // Create a QIR call to allocate the qubits.
    StringRef qirQubitArrayAllocate = M::getQIRArrayQubitAllocateArray();
    Type arrayQubitTy = getArrayType(getTypeConverter());

    // AllocaOp could have a size operand, or the size could
    // be compile time known and encoded in the veq return type.
    Value sizeOperand;
    auto loc = alloc.getLoc();
    if (adaptor.getOperands().empty()) {
      auto type = alloc.getType().cast<quake::VeqType>();
      if (!type.hasSpecifiedSize())
        return failure();
      auto constantSize = type.getSize();
      sizeOperand =
          rewriter.create<arith::ConstantIntOp>(loc, constantSize, 64);
    } else {
      sizeOperand = adaptor.getOperands().front();
      if (sizeOperand.getType().cast<IntegerType>().getWidth() < 64)
        sizeOperand = rewriter.create<cudaq::cc::CastOp>(
            loc, rewriter.getI64Type(), sizeOperand,
            cudaq::cc::CastOpMode::Unsigned);
    }

    // Replace the AllocaOp with the QIR call.
    rewriter.replaceOpWithNewOp<func::CallOp>(alloc, TypeRange{arrayQubitTy},
                                              qirQubitArrayAllocate,
                                              ValueRange{sizeOperand});
    return success();
  }
};

struct EraseAllocaOp : public OpConversionPattern<quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::AllocaOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Modifier classes
//===----------------------------------------------------------------------===//

/// The modifier class for the "full QIR" API.
struct FullQIR {
  using Self = FullQIR;

  template <typename QuakeOp>
  static std::string quakeToFuncName(QuakeOp op) {
    auto prefix = getGateFunctionPrefix(op.getOperation());
    if (op.getIsAdj())
      return prefix + "__adj";
    return prefix;
  }

  static void populateRewritePatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
    patterns.insert<AllocaOpRewrite<Self>>(typeConverter,
                                           patterns.getContext());
  }

  static StringRef getQIRQubitAllocate() {
    return cudaq::opt::QIRQubitAllocate;
  }

  static StringRef getQIRArrayQubitAllocateArray() {
    return cudaq::opt::QIRArrayQubitAllocateArray;
  }

  // No quake ops are allowed. We convert them all in the conversion phase.
  static void allowedQuakeOps(ConversionTarget &target) {}
};

/// The base modifier class for the "profile QIR" APIs.
struct AnyProfileQIR {
  using Self = AnyProfileQIR;

  template <typename QuakeOp>
  static std::string quakeToFuncName(QuakeOp op) {
    auto prefix = getGateFunctionPrefix(op.getOperation());
    if (op.getIsAdj())
      return getGateFunctionPrefix(op) + "__adj";
    return prefix + "__body";
  }

  static void populateRewritePatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
    patterns.insert<EraseAllocaOp>(typeConverter, patterns.getContext());
  }

  // Some quake ops are allowed to pass through the conversion step. They will
  // be erased in finalization.
  static void allowedQuakeOps(ConversionTarget &target) {
    target.addLegalOp<quake::AllocaOp>();
  }
};

/// The QIR base profile modifier class.
struct BaseProfileQIR : public AnyProfileQIR {
  using Self = BaseProfileQIR;

  static void populateRewritePatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
    AnyProfileQIR::populateRewritePatterns(patterns, typeConverter);
  }
};

/// The QIR adaptive profile modifier class.
struct AdaptiveProfileQIR : public AnyProfileQIR {
  using Self = AdaptiveProfileQIR;

  static void populateRewritePatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
    AnyProfileQIR::populateRewritePatterns(patterns, typeConverter);
  }
};

//===----------------------------------------------------------------------===//
// Quake conversion to the QIR API driver pass.
//
// This is done in 3 phased: preparation, conversion, and finalization.
//===----------------------------------------------------------------------===//

struct QuakeToQIRAPIPass
    : public cudaq::opt::impl::QuakeToQIRAPIBase<QuakeToQIRAPIPass> {
  using QuakeToQIRAPIBase::QuakeToQIRAPIBase;

  template <typename A>
  void processOperation() {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before QIR API conversion:\n" << *op << '\n');
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    QIRAPITypeConverter typeConverter(ctx, opaquePtr);
    A::populateRewritePatterns(patterns, typeConverter);
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                           cf::ControlFlowDialect, func::FuncDialect,
                           LLVM::LLVMDialect>();
    target.addIllegalDialect<quake::QuakeDialect>();
    A::allowedQuakeOps(target);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After QIR API conversion:\n" << *op << '\n');
  }

  void runOnOperation() override {
    if (api == "full")
      processOperation<FullQIR>();
    else if (api == "base-profile")
      processOperation<BaseProfileQIR>();
    else if (api == "adaptive-profile")
      processOperation<AdaptiveProfileQIR>();
    else
      signalPassFailure();
  }
};

struct QuakeToQIRAPIPrepPass
    : public cudaq::opt::impl::QuakeToQIRAPIPrepBase<QuakeToQIRAPIPrepPass> {
  using QuakeToQIRAPIPrepBase::QuakeToQIRAPIPrepBase;

  void runOnOperation() override {}
};

struct QuakeToQIRAPIFinalPass
    : public cudaq::opt::impl::QuakeToQIRAPIFinalBase<QuakeToQIRAPIFinalPass> {
  using QuakeToQIRAPIFinalBase::QuakeToQIRAPIFinalBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    ModuleOp module = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<EraseAllocaOp>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void cudaq::opt::addConvertToQIRAPIPipeline(OpPassManager &pm, StringRef api) {
  QuakeToQIRAPIPrepOptions prepApiOpt{.api = api.str()};
  pm.addPass(cudaq::opt::createQuakeToQIRAPIPrep(prepApiOpt));
  QuakeToQIRAPIOptions apiOpt{.api = api.str()};
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeToQIRAPI(apiOpt));
  pm.addPass(cudaq::opt::createQuakeToQIRAPIFinal());
}

namespace {
struct QIRAPIPipelineOptions
    : public PassPipelineOptions<QIRAPIPipelineOptions> {
  PassOptions::Option<std::string> api{
      *this, "api",
      llvm::cl::desc("select the profile to convert to [full, base-profile, "
                     "adaptive-profile]"),
      llvm::cl::init("full")};
};
} // namespace

void cudaq::opt::registerToQIRAPIPipeline() {
  PassPipelineRegistration<QIRAPIPipelineOptions>(
      "convert-to-qir-api", "Convert quake to one of the QIR APIs.",
      [](OpPassManager &pm, const QIRAPIPipelineOptions &opt) {
        addConvertToQIRAPIPipeline(pm, opt.api);
      });
}
