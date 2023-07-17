/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/CUDAQBuilder.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Peephole.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Todo.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

/// This file maps full QIR to the Base Profile QIR. It is generally assumed
/// that the input QIR here will be generated after the quake-synth pass,
/// thereby greatly simplifying the transformations required here.

using namespace mlir;

/// For a call to `__quantum__rt__qubit_allocate_array`, get the number of
/// qubits allocated.
static std::size_t getNumQubits(LLVM::CallOp callOp) {
  auto sizeOperand = callOp.getOperand(0);
  auto defOp = sizeOperand.getDefiningOp();
  // walk back up to the defining op, has to be a constant
  while (defOp && !dyn_cast<LLVM::ConstantOp>(defOp))
    defOp = defOp->getOperand(0).getDefiningOp();
  if (auto constOp = dyn_cast_or_null<LLVM::ConstantOp>(defOp))
    return constOp.getValue().cast<IntegerAttr>().getValue().getLimitedValue();
  TODO_loc(callOp.getLoc(), "cannot compute number of qubits allocated");
}

static bool isQIRSliceCall(Operation *op) {
  if (auto call = dyn_cast_or_null<LLVM::CallOp>(op)) {
    StringRef funcName = call.getCalleeAttr().getValue();
    return funcName.equals(cudaq::opt::QIRArraySlice);
  }
  return false;
}

static std::optional<std::int64_t> sliceLowerBound(Operation *op) {
  Value low = op->getOperand(2);
  if (auto con = low.getDefiningOp<LLVM::ConstantOp>())
    return con.getValue().cast<IntegerAttr>().getInt();
  return {};
}

namespace {
struct FunctionAnalysisData {
  std::size_t nQubits = 0;
  std::size_t nResults = 0;
  // Use std::map to keep these sorted in ascending order.
  std::map<std::size_t, std::pair<std::size_t, StringAttr>> resultPtrValues;
  DenseMap<Operation *, std::size_t> allocationOffsets;
};

using FunctionAnalysisInfo = DenseMap<Operation *, FunctionAnalysisData>;

/// The analysis on an entry function.
struct FunctionProfileAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FunctionProfileAnalysis)

  FunctionProfileAnalysis(Operation *op) { performAnalysis(op); }

  const FunctionAnalysisInfo &getAnalysisInfo() const { return infoMap; }

private:
  // Scan the body of a function for ops that will be used for profiling.
  void performAnalysis(Operation *operation) {
    auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(operation);
    if (!funcOp)
      return;
    FunctionAnalysisData data;
    funcOp->walk([&](Operation *op) {
      if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
        StringRef funcName = callOp.getCalleeAttr().getValue();
        auto addAllocation = [&](auto incrementBy) {
          // Maybe add an allocation, setting its offset, and the total number
          // of qubits.
          if (data.allocationOffsets.find(callOp) ==
              data.allocationOffsets.end()) {
            data.allocationOffsets[callOp] = data.nQubits;
            data.nQubits += incrementBy();
          }
        };

        if (funcName.equals(cudaq::opt::QIRMeasure) ||
            // FIXME Store the register names for the record_output functions
            funcName.equals(cudaq::opt::QIRMeasureToRegister)) {
          std::optional<std::size_t> optQb;
          if (auto allocCall =
                  callOp.getOperand(0).getDefiningOp<LLVM::CallOp>()) {
            auto iter = data.allocationOffsets.find(allocCall.getOperation());
            if (iter != data.allocationOffsets.end())
              optQb = iter->second;
          } else if (auto load =
                         callOp.getOperand(0).getDefiningOp<LLVM::LoadOp>()) {
            if (auto bitcast =
                    load.getOperand().getDefiningOp<LLVM::BitcastOp>()) {
              std::optional<Value> constVal;
              std::size_t allocOffset = 0u;
              if (auto call =
                      bitcast.getOperand().getDefiningOp<LLVM::CallOp>()) {
                auto *callOp0 = call.getOperand(0).getDefiningOp();
                while (isQIRSliceCall(callOp0)) {
                  if (auto optOff = sliceLowerBound(callOp0)) {
                    allocOffset += *optOff;
                    callOp0 = callOp0->getOperand(0).getDefiningOp();
                  } else {
                    callOp0->emitError("cannot compute offset");
                    callOp0 = nullptr;
                  }
                }
                auto iter = callOp0 ? data.allocationOffsets.find(callOp0)
                                    : data.allocationOffsets.end();
                if (iter != data.allocationOffsets.end()) {
                  allocOffset += iter->second;
                  auto o1 = call.getOperand(1);
                  if (auto c = o1.getDefiningOp<LLVM::ConstantOp>()) {
                    constVal = c;
                  } else {
                    // Skip over any potential intermediate cast.
                    auto *defOp = o1.getDefiningOp();
                    if (isa_and_nonnull<LLVM::ZExtOp, LLVM::SExtOp,
                                        LLVM::PtrToIntOp, LLVM::BitcastOp,
                                        LLVM::TruncOp>(defOp))
                      constVal = defOp->getOperand(0);
                  }
                }
              }
              if (constVal)
                if (auto incr = constVal->getDefiningOp<LLVM::ConstantOp>())
                  optQb = allocOffset +
                          incr.getValue().cast<IntegerAttr>().getInt();
            }
          }
          if (optQb) {
            auto qb = *optQb;
            auto iter = data.resultPtrValues.find(qb);
            auto *ctx = callOp.getContext();
            auto intTy = IntegerType::get(ctx, 64);
            if (iter == data.resultPtrValues.end()) {
              auto resIdx = IntegerAttr::get(intTy, data.nResults);
              callOp->setAttr(resultIndexName, resIdx);
              auto regName = [&]() -> StringAttr {
                if (auto nameAttr = callOp->getAttr("registerName")
                                        .dyn_cast_or_null<StringAttr>())
                  return nameAttr;
                return {};
              }();
              data.resultPtrValues.insert(
                  std::make_pair(qb, std::make_pair(data.nResults++, regName)));
            } else {
              auto resIdx = IntegerAttr::get(intTy, iter->second.first);
              callOp->setAttr(resultIndexName, resIdx);
            }
          } else {
            callOp.emitError("could not trace offset value");
          }
        } else if (funcName.equals(cudaq::opt::QIRArrayQubitAllocateArray)) {
          addAllocation([&]() { return getNumQubits(callOp); });
        } else if (funcName.equals(cudaq::opt::QIRQubitAllocate)) {
          addAllocation([]() { return 1; });
        }
      }
    });
    infoMap.insert({operation, data});
  }

  FunctionAnalysisInfo infoMap;
};

struct AddFuncAttribute : public OpRewritePattern<LLVM::LLVMFuncOp> {
  explicit AddFuncAttribute(MLIRContext *ctx, const FunctionAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp op,
                                PatternRewriter &rewriter) const override {
    // Rewrite the exit block.
    // Add attributes to the function.
    auto iter = infoMap.find(op);
    assert(iter != infoMap.end());
    rewriter.startRootUpdate(op);
    const auto &info = iter->second;
    // QIR functions need certain attributes, add them here.
    auto arrAttr = rewriter.getArrayAttr(ArrayRef<Attribute>{
        rewriter.getStringAttr("EntryPoint"),
        rewriter.getStrArrayAttr(
            {"requiredQubits", std::to_string(info.nQubits)}),
        rewriter.getStrArrayAttr(
            {"requiredResults", std::to_string(info.nResults)})});
    op.setPassthroughAttr(arrAttr);

    // Stick the record calls in the exit block.
    auto builder = cudaq::IRBuilder::atBlockTerminator(&op.getBody().back());
    auto loc = op.getBody().back().getTerminator()->getLoc();

    builder.create<LLVM::CallOp>(loc, TypeRange{},
                                 cudaq::opt::QIRBaseProfileStartRecordOutput,
                                 ArrayRef<Value>{});
    auto resultTy = cudaq::opt::getResultType(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto module = op->getParentOfType<ModuleOp>();
    for (auto &iv : info.resultPtrValues) {
      auto &rec = iv.second;
      Value idx = builder.create<LLVM::ConstantOp>(loc, i64Ty, rec.first);
      Value ptr = builder.create<LLVM::IntToPtrOp>(loc, resultTy, idx);
      auto regName = [&]() -> Value {
        auto charPtrTy = cudaq::opt::getCharPointerType(builder.getContext());
        if (rec.second) {
          // Note: it should be the case that this string literal has already
          // been added to the IR, so this step does not actually update the
          // module.
          auto globl =
              builder.genCStringLiteralAppendNul(loc, module, rec.second);
          auto addrOf = builder.create<LLVM::AddressOfOp>(
              loc, cudaq::opt::factory::getPointerType(globl.getType()),
              globl.getName());
          return builder.create<LLVM::BitcastOp>(loc, charPtrTy, addrOf);
        }
        Value zero = builder.create<LLVM::ConstantOp>(loc, i64Ty, 0);
        return builder.create<LLVM::IntToPtrOp>(loc, charPtrTy, zero);
      }();
      builder.create<LLVM::CallOp>(loc, TypeRange{},
                                   cudaq::opt::QIRBaseProfileRecordOutput,
                                   ValueRange{ptr, regName});
    }
    builder.create<LLVM::CallOp>(loc, TypeRange{},
                                 cudaq::opt::QIRBaseProfileEndRecordOutput,
                                 ArrayRef<Value>{});
    rewriter.finalizeRootUpdate(op);
    return success();
  }

  const FunctionAnalysisInfo &infoMap;
};

/// QIR to Base Profile QIR on the function level.
///
/// With FuncOps, we want to add attributes to the function op and also add
/// calls to the "record" API in the exit block of the function. The record
/// calls are bijective with all distinct measurement calls in the original
/// function, however the indices used may be renumbered and start at 0.
struct QIRToBaseQIRFuncPass
    : public cudaq::opt::QIRToBaseQIRFuncBase<QIRToBaseQIRFuncPass> {
  using QIRToBaseQIRFuncBase::QIRToBaseQIRFuncBase;

  void runOnOperation() override {
    auto op = getOperation();
    auto *ctx = op.getContext();
    RewritePatternSet patterns(ctx);
    const auto &analysis = getAnalysis<FunctionProfileAnalysis>();
    const auto &funcAnalysisInfo = analysis.getAnalysisInfo();
    patterns.insert<AddFuncAttribute>(ctx, funcAnalysisInfo);
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp op) {
      // If the function is a definition that doesn't have the attributes
      // applied, then it is illegal.
      return op.empty() || op.getPassthroughAttr();
    });
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op.getLoc(), "failed to convert to QIR base profile");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createConvertToQIRFuncPass() {
  return std::make_unique<QIRToBaseQIRFuncPass>();
}

//===----------------------------------------------------------------------===//

namespace {
/// QIR to the Base Profile QIR
///
/// This pass converts patterns in LLVM-IR dialect using QIR calls, etc. into a
/// subset of QIR, the base profile. This pass uses a greedy rewrite to match
/// DAGs in the IR and replace them to meet the requirements of the base
/// profile. The patterns are defined in Peephole.td.
struct QIRToBaseProfileQIRPass
    : public cudaq::opt::QIRToBaseQIRBase<QIRToBaseProfileQIRPass> {
  QIRToBaseProfileQIRPass() = default;
  QIRToBaseProfileQIRPass(const GreedyRewriteConfig &config,
                          ArrayRef<std::string> disabledPatterns,
                          ArrayRef<std::string> enabledPatterns) {
    this->topDownProcessingEnabled = config.useTopDownTraversal;
    this->enableRegionSimplification = config.enableRegionSimplification;
    this->maxIterations = config.maxIterations;
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    owningPatterns.insert<AddrOfCisToBase, ArrayGetElementPtrConv, CalleeConv,
                          EraseArrayAlloc, EraseArrayRelease, EraseDeadArrayGEP,
                          LoadMeasureResult, MeasureCallConv,
                          MeasureToRegisterCallConv, XCtrlOneTargetToCNot>(
        context);
    patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                       disabledPatterns, enabledPatterns);
    return success();
  }

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patterns, config)))
      signalPassFailure();
  }

private:
  FrozenRewritePatternSet patterns;
  ArrayRef<std::string> disabledPatterns;
  ArrayRef<std::string> enabledPatterns;
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createQIRToBaseProfilePass() {
  return std::make_unique<QIRToBaseProfileQIRPass>();
}

//===----------------------------------------------------------------------===//

namespace {
/// Base Profile Preparation:
///
/// Before we can do the conversion to the QIR base profile with different
/// threads running on different functions, the module is updated with the
/// signatures of functions from the QIR ABI that may be called by the
/// translation. This trivial pass only does this preparation work. It performs
/// no analysis and does not rewrite function body's, etc.
struct BaseProfilePreparationPass
    : public cudaq::opt::QIRToBaseQIRPrepBase<BaseProfilePreparationPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();

    // Add cnot declaration as it may be referenced after peepholes run.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRCnot, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getQubitType(ctx), cudaq::opt::getQubitType(ctx)}, module);

    // Add measure_body as it has a different signature than measure.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRMeasureBody, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getQubitType(ctx), cudaq::opt::getResultType(ctx)},
        module);
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRReadResultBody, IntegerType::get(ctx, 1),
        {cudaq::opt::getResultType(ctx)}, module);

    // Add record functions for any measurements.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRBaseProfileStartRecordOutput,
        LLVM::LLVMVoidType::get(ctx), {}, module);
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRBaseProfileEndRecordOutput, LLVM::LLVMVoidType::get(ctx),
        {}, module);
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRBaseProfileRecordOutput, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getResultType(ctx), cudaq::opt::getCharPointerType(ctx)},
        module);

    // Add functions `__quantum__qis__*__body` for all functions matching
    // `__quantum__qis__*` that are found.
    for (auto &global : module)
      if (auto func = dyn_cast<LLVM::LLVMFuncOp>(global))
        if (needsToBeRenamed(func.getName()))
          cudaq::opt::factory::createLLVMFunctionSymbol(
              func.getName().str() + "__body",
              func.getFunctionType().getReturnType(),
              func.getFunctionType().getParams(), module);
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createBaseProfilePreparationPass() {
  return std::make_unique<BaseProfilePreparationPass>();
}

//===----------------------------------------------------------------------===//

namespace {
/// Verify that the base profile QIR code is sane. For now, this simply checks
/// that the QIR base profile doesn't have any "bonus" calls to arbitrary code
/// that is not possibly defined in the QIR standard.
struct VerifyBaseProfilePass
    : public cudaq::opt::VerifyBaseProfileBase<VerifyBaseProfilePass> {

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    bool passFailed = false;
    if (!func->hasAttr(cudaq::entryPointAttrName))
      return;
    func.walk([&](Operation *op) {
      if (auto call = dyn_cast<LLVM::CallOp>(op)) {
        auto funcName = call.getCalleeAttr().getValue();
        if (!funcName.startswith("__quantum_")) {
          call.emitOpError("unexpected call in QIR base profile");
          passFailed = true;
        }
      } else if (isa<LLVM::BrOp, LLVM::CondBrOp, LLVM::ResumeOp,
                     LLVM::UnreachableOp, LLVM::SwitchOp>(op)) {
        op->emitOpError("QIR base profile does not support control-flow");
        passFailed = true;
      }
    });
    if (passFailed) {
      emitError(func.getLoc(),
                "function " + func.getName() +
                    " not compatible with the QIR base profile.");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::verifyBaseProfilePass() {
  return std::make_unique<VerifyBaseProfilePass>();
}

// The various passes defined here should be added as a pass pipeline.
void cudaq::opt::addBaseProfilePipeline(OpPassManager &pm) {
  pm.addPass(createBaseProfilePreparationPass());
  pm.addNestedPass<LLVM::LLVMFuncOp>(createConvertToQIRFuncPass());
  pm.addPass(createQIRToBaseProfilePass());
  pm.addNestedPass<LLVM::LLVMFuncOp>(verifyBaseProfilePass());
}

void cudaq::opt::registerBaseProfilePipeline() {
  PassPipelineRegistration<>(
      "base-profile-pipeline",
      "Pass pipeline to generate code for the QIR base profile.",
      addBaseProfilePipeline);
}
