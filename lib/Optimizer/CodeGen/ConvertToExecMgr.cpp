/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/CudaqFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QuakeToExecMgr.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "convert-to-cc"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKETOCCPREP
#define GEN_PASS_DEF_QUAKETOCC
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
struct QuakeTypeConverter : public TypeConverter {
  QuakeTypeConverter() {
    addConversion([](Type ty) { return ty; });
    addConversion([](quake::VeqType ty) {
      return cudaq::cc::PointerType::get(
          cudaq::opt::getCudaqQubitSpanType(ty.getContext()));
    });
    addConversion([](quake::RefType ty) {
      return cudaq::cc::PointerType::get(
          cudaq::opt::getCudaqQubitSpanType(ty.getContext()));
    });
    addConversion([&](quake::StruqType ty) {
      SmallVector<Type> mems;
      for (auto m : ty.getMembers())
        mems.push_back(convertType(m));
      return cudaq::cc::StructType::get(ty.getContext(), mems);
    });
    addConversion([](quake::MeasureType ty) {
      return IntegerType::get(ty.getContext(), 64);
    });
  }
};

struct QuakeToCCPass : public cudaq::opt::impl::QuakeToCCBase<QuakeToCCPass> {
  using QuakeToCCBase::QuakeToCCBase;

  void runOnOperation() override {
    auto op = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    QuakeTypeConverter quakeTypeConverter;
    cudaq::opt::populateQuakeToCCPatterns(quakeTypeConverter, patterns);
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                           cf::ControlFlowDialect, func::FuncDialect,
                           LLVM::LLVMDialect>();
    target.addIllegalDialect<quake::QuakeDialect>();

    LLVM_DEBUG(llvm::dbgs() << "Module before:\n"; op.dump());
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "Module after:\n"; op->dump());
  }
};

struct QuakeToCCPrepPass
    : public cudaq::opt::impl::QuakeToCCPrepBase<QuakeToCCPrepPass> {
  using QuakeToCCPrepBase::QuakeToCCPrepBase;

  void runOnOperation() override {
    auto op = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    cudaq::opt::populateQuakeToCCPrepPatterns(patterns);

    LLVM_DEBUG(llvm::dbgs() << "Module before prep:\n"; op.dump());
    // Preload all our intrinsics.
    cudaq::IRBuilder irBuilder(context);
    if (failed(irBuilder.loadIntrinsic(op, cudaq::opt::CudaqEMAllocate)) ||
        failed(irBuilder.loadIntrinsic(op, cudaq::opt::CudaqEMAllocateVeq)) ||
        failed(irBuilder.loadIntrinsic(op, cudaq::opt::CudaqEMApply)) ||
        failed(irBuilder.loadIntrinsic(op, cudaq::opt::CudaqEMConcatSpan)) ||
        failed(irBuilder.loadIntrinsic(op, cudaq::opt::CudaqEMMeasure)) ||
        failed(irBuilder.loadIntrinsic(op, cudaq::opt::CudaqEMReset)) ||
        failed(irBuilder.loadIntrinsic(op, cudaq::opt::CudaqEMReturn)) ||
        failed(irBuilder.loadIntrinsic(op, cudaq::opt::CudaqEMWriteToSpan))) {
      signalPassFailure();
      return;
    }

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "Module after prep:\n"; op->dump());
  }
};
} // namespace

void cudaq::opt::addLowerToCCPipeline(mlir::OpPassManager &pm) {
  pm.addPass(cudaq::opt::createQuakeToCCPrep());
  pm.addPass(cudaq::opt::createQuakeToCC());
}

void cudaq::opt::registerToExecutionManagerCCPipeline() {
  PassPipelineRegistration<>(
      "lower-quake", "Convert quake directly to calls to execution manager.",
      addLowerToCCPipeline);
}
