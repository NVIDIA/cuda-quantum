/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CodeGenOps.h"
#include "PassDetails.h"
#include "QuakeToCodegen.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/CCToLLVM.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Peephole.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/CodeGen/QuakeToLLVM.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ComplexToLibm/ComplexToLibm.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "convert-to-qir"

/**
   \file

   This file translates Quake to full QIR. This pass \e only supports QIR
   version 0.1.
 */

namespace cudaq::opt {
#define GEN_PASS_DEF_CONVERTTOQIR
#define GEN_PASS_DEF_LOWERTOCG
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

#include "PeepholePatterns.inc"

/// Greedy pass to match subgraphs in the IR and replace them with codegen ops.
/// This step makes converting a DAG of nodes in the conversion step simpler.
static LogicalResult fuseSubgraphPatterns(MLIRContext *ctx, ModuleOp module) {
  RewritePatternSet patterns(ctx);
  cudaq::codegen::populateQuakeToCodegenPatterns(patterns);
  LLVM_DEBUG(llvm::dbgs() << "Before codegen dialect:\n"; module.dump());
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "After codegen dialect:\n"; module.dump());
  return success();
}

namespace {
/// Convert Quake dialect to LLVM-IR and QIR.
class ConvertToQIR : public cudaq::opt::impl::ConvertToQIRBase<ConvertToQIR> {
public:
  using ConvertToQIRBase::ConvertToQIRBase;

  /// Measurement counter for unnamed measurements. Resets every module.
  unsigned measureCounter = 0;

  // This is an ad hoc transformation to convert constant array values into a
  // buffer of constants.
  LogicalResult eraseConstantArrayOps() {
    bool ok = true;
    SmallVector<Operation *> cleanUps;
    getOperation().walk([&](cudaq::cc::ConstantArrayOp carr) {
      // If there is a constant array, then we expect that it is involved in
      // a stdvec initializer expression. So look for the pattern and expand
      // the store into a series of scalar stores.
      //
      //   %100 = cc.const_array [c1, c2, ... cN] : ...
      //   %110 = cc.alloca ...
      //   cc.store %100, %110 : ...
      //   __________________________
      //
      //   cc.store c1, %110[0]
      //   cc.store c2, %110[1]
      //   ...
      //   cc.store cN, %110[N-1]

      // Are all uses the value to a store?
      if (!std::all_of(carr->getUsers().begin(), carr->getUsers().end(),
                       [&](auto *op) {
                         auto st = dyn_cast<cudaq::cc::StoreOp>(op);
                         return st && st.getValue() == carr.getResult();
                       })) {
        ok = false;
        return;
      }

      auto eleTy = cast<cudaq::cc::ArrayType>(carr.getType()).getElementType();
      auto ptrTy = cudaq::cc::PointerType::get(eleTy);
      auto loc = carr.getLoc();

      for (auto *user : carr->getUsers()) {
        auto origStore = cast<cudaq::cc::StoreOp>(user);
        OpBuilder builder(origStore);
        auto buffer = origStore.getPtrvalue();
        const std::int32_t numConstants = carr.getConstantValues().size();
        auto constantValues = carr.getConstantValues();
        for (std::int32_t idx = 0; idx < numConstants; idx++) {
          auto v = [&]() -> Value {
            auto val = constantValues[idx];
            if (auto fTy = dyn_cast<FloatType>(eleTy))
              return builder.create<arith::ConstantFloatOp>(
                  loc, cast<FloatAttr>(val).getValue(), fTy);
            if (auto iTy = dyn_cast<IntegerType>(eleTy))
              return builder.create<arith::ConstantIntOp>(
                  loc, cast<IntegerAttr>(val).getInt(), iTy);
            auto cTy = cast<ComplexType>(eleTy);
            return builder.create<complex::ConstantOp>(loc, cTy,
                                                       cast<ArrayAttr>(val));
          }();
          Value arrWithOffset = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrTy, buffer, ArrayRef<cudaq::cc::ComputePtrArg>{idx});
          builder.create<cudaq::cc::StoreOp>(loc, v, arrWithOffset);
        }
        cleanUps.push_back(user);
      }
      cleanUps.push_back(carr.getOperation());
    });

    for (auto *op : cleanUps) {
      op->dropAllUses();
      op->erase();
    }
    return ok ? success() : failure();
  }

  /// Greedy pass to match subgraphs in the IR and replace them with codegen
  /// ops. This step makes converting a DAG of nodes in the conversion step
  /// simpler.
  void runOnOperation() override final {
    auto *context = &getContext();
    if (failed(fuseSubgraphPatterns(context, getOperation()))) {
      signalPassFailure();
      return;
    }
    // Ad hoc deal with ConstantArrayOp transformation.
    // TODO: Merge this into the codegen dialect once that gets to main.
    if (failed(eraseConstantArrayOps())) {
      getOperation().emitOpError("unexpected constant arrays");
      signalPassFailure();
      return;
    }

    LLVMConversionTarget target{*context};
    LLVMTypeConverter typeConverter(&getContext());
    cudaq::opt::initializeTypeConversions(typeConverter);
    RewritePatternSet patterns(context);

    populateComplexToLibmConversionPatterns(patterns, 1);
    populateComplexToLLVMConversionPatterns(typeConverter, patterns);

    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);

    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    cudaq::opt::populateCCToLLVMPatterns(typeConverter, patterns);
    cudaq::opt::populateQuakeToLLVMPatterns(typeConverter, patterns,
                                            measureCounter);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();

    auto op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before conversion to QIR:\n"; op.dump());
    if (failed(applyFullConversion(op, target, std::move(patterns)))) {
      LLVM_DEBUG(getOperation().dump());
      signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "After conversion to QIR:\n"; op.dump());
  }
};

} // namespace

void cudaq::opt::initializeTypeConversions(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](quake::VeqType type) { return getArrayType(type.getContext()); });
  typeConverter.addConversion(
      [](quake::RefType type) { return getQubitType(type.getContext()); });
  typeConverter.addConversion([&](quake::StruqType type) {
    SmallVector<Type> mems;
    for (auto m : type.getMembers())
      mems.push_back(typeConverter.convertType(m));
    return LLVM::LLVMStructType::getLiteral(type.getContext(), mems,
                                            /*packed=*/false);
  });
  typeConverter.addConversion([](quake::MeasureType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  cudaq::opt::populateCCTypeConversions(&typeConverter);
}

namespace {
class LowerToCG : public cudaq::opt::impl::LowerToCGBase<LowerToCG> {
public:
  using LowerToCGBase::LowerToCGBase;

  void runOnOperation() override {
    if (failed(fuseSubgraphPatterns(&getContext(), getOperation())))
      signalPassFailure();
  }
};
} // namespace
