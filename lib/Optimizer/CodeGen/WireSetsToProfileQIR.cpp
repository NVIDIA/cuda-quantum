/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/CudaqFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/CodeGen/QuakeToCC.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "wireset-to-profile-qir"

namespace cudaq::opt {
#define GEN_PASS_DEF_WIRESETTOPROFILEQIR
#define GEN_PASS_DEF_WIRESETTOPROFILEQIRPREP
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
    addConversion([](quake::MeasureType ty) {
      return IntegerType::get(ty.getContext(), 64);
    });
  }
};

struct WireSetToProfileQIRPass
    : public cudaq::opt::impl::WireSetToProfileQIRBase<
          WireSetToProfileQIRPass> {
  using WireSetToProfileQIRBase::WireSetToProfileQIRBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    QuakeTypeConverter quakeTypeConverter;
    cudaq::opt::populateQuakeToCCPatterns(quakeTypeConverter, patterns);
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                           func::FuncDialect, LLVM::LLVMDialect>();
    target.addIllegalDialect<quake::QuakeDialect>();

    LLVM_DEBUG(llvm::dbgs() << "Module before:\n"; op->dump());
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "Module after:\n"; op->dump());
  }
};

struct WireSetToProfileQIRPrepPass
    : public cudaq::opt::impl::WireSetToProfileQIRPrepBase<
          WireSetToProfileQIRPrepPass> {
  using WireSetToProfileQIRPrepBase::WireSetToProfileQIRPrepBase;

  void runOnOperation() override {
    ModuleOp op = getOperation();
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

void cudaq::opt::addWiresetToProfileQIRPipeline(OpPassManager &pm,
                                                StringRef profile) {
  pm.addPass(cudaq::opt::createWireSetToProfileQIRPrep());
  WireSetToProfileQIROptions wopt;
  if (!profile.empty())
    wopt.convertTo = profile.str();
  pm.addPass(cudaq::opt::createWireSetToProfileQIR(wopt));
}

// Pipeline option: let the user specify the profile name.
struct WiresetToProfileQIRPipelineOptions
    : public PassPipelineOptions<WiresetToProfileQIRPipelineOptions> {
  PassOptions::Option<std::string> profile{
      *this, "convert-to", llvm::cl::desc(""), llvm::cl::init("qir-base")};
};

void cudaq::opt::registerWireSetToProfileQIRPipeline() {
  PassPipelineRegistration<WiresetToProfileQIRPipelineOptions>(
      "lower-wireset-to-profile-qir",
      "Convert quake directly to one of the profiles of QIR.",
      [](OpPassManager &pm, const WiresetToProfileQIRPipelineOptions &opt) {
        addWiresetToProfileQIRPipeline(pm, opt.profile);
      });
}
