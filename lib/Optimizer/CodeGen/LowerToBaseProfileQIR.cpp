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
#include "nlohmann/json.hpp"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "qir-base-profile"

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

static constexpr char StartingOffsetAttrName[] = "StartingOffset";

namespace {
struct FunctionAnalysisData {
  std::size_t nQubits = 0;
  std::size_t nResults = 0;
  // Use std::map to keep these sorted in ascending order.
  // map[qb] --> [result,regName]
  std::map<std::size_t, std::pair<std::size_t, StringAttr>> resultPtrValues;
  // Additionally store by result to prevent collisions on a single qubit having
  // multiple measurements (Adaptive Profile)
  // map[result] --> [qb,regName]
  std::map<std::size_t, std::pair<std::size_t, std::string>> resultQubitVals;
  DenseMap<Operation *, std::size_t> allocationOffsets;
};

using FunctionAnalysisInfo = DenseMap<Operation *, FunctionAnalysisData>;

/// The analysis on an entry function.
struct FunctionProfileAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FunctionProfileAnalysis)

  explicit FunctionProfileAnalysis(Operation *op) { performAnalysis(op); }

  const FunctionAnalysisInfo &getAnalysisInfo() const { return infoMap; }

private:
  // Scan the body of a function for ops that will be used for profiling.
  void performAnalysis(Operation *operation) {
    auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(operation);
    if (!funcOp)
      return;
    FunctionAnalysisData data;
    funcOp->walk([&](LLVM::CallOp callOp) {
      StringRef funcName = callOp.getCalleeAttr().getValue();

      // For every allocation call, create a range of integers to uniquely
      // identify the qubits in the allocation.
      auto addAllocation = [&](auto incrementBy) {
        // Maybe add an allocation, setting its offset, and the total number
        // of qubits.
        if (data.allocationOffsets.find(callOp) ==
            data.allocationOffsets.end()) {
          data.allocationOffsets[callOp] = data.nQubits;
          data.nQubits += incrementBy();
        }
      };

      if (funcName.equals(cudaq::opt::QIRArrayQubitAllocateArray)) {
        addAllocation([&]() { return getNumQubits(callOp); });
      } else if (funcName.equals(cudaq::opt::QIRQubitAllocate)) {
        addAllocation([]() { return 1; });
      } else if (funcName.equals(cudaq::opt::QIRMeasure) ||
                 // FIXME Store the register names for the record_output
                 // functions
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
                optQb =
                    allocOffset + incr.getValue().cast<IntegerAttr>().getInt();
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
            data.resultQubitVals.insert(std::make_pair(
                data.nResults, std::make_pair(qb, regName.data())));
            data.resultPtrValues.insert(
                std::make_pair(qb, std::make_pair(data.nResults++, regName)));
          } else {
            auto resIdx = IntegerAttr::get(intTy, iter->second.first);
            callOp->setAttr(resultIndexName, resIdx);
          }
        } else {
          callOp.emitError("could not trace offset value");
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
    nlohmann::json resultQubitJSON{info.resultQubitVals};
    // QIR functions need certain attributes, add them here.
    // TODO: Update schema_id with valid value (issues #385 and #556)
    auto arrAttr = rewriter.getArrayAttr(ArrayRef<Attribute>{
        rewriter.getStringAttr("entry_point"),
        rewriter.getStrArrayAttr({"qir_profiles", "base_profile"}),
        rewriter.getStrArrayAttr({"output_labeling_schema", "schema_id"}),
        rewriter.getStrArrayAttr({"output_names", resultQubitJSON.dump()}),
        rewriter.getStrArrayAttr(
            // TODO: change to required_num_qubits once providers support it
            // (issues #385 and #556)
            {"requiredQubits", std::to_string(info.nQubits)}),
        rewriter.getStrArrayAttr(
            // TODO: change to required_num_results once providers support it
            // (issues #385 and #556)
            {"requiredResults", std::to_string(info.nResults)})});
    op.setPassthroughAttr(arrAttr);

    // Stick the record calls in the exit block.
    auto builder = cudaq::IRBuilder::atBlockTerminator(&op.getBody().back());
    auto loc = op.getBody().back().getTerminator()->getLoc();

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
    rewriter.finalizeRootUpdate(op);
    return success();
  }

  const FunctionAnalysisInfo &infoMap;
};

struct AddCallAttribute : public OpRewritePattern<LLVM::CallOp> {
  explicit AddCallAttribute(MLIRContext *ctx, const FunctionAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rewriter) const override {
    // Rewrite the exit block.
    // Add attributes to the function.
    auto iter = infoMap.find(op->getParentOfType<LLVM::LLVMFuncOp>());
    assert(iter != infoMap.end());
    auto &info = iter->second;
    auto startIter = info.allocationOffsets.find(op.getOperation());
    assert(startIter != info.allocationOffsets.end());
    auto startVal = startIter->second;
    rewriter.startRootUpdate(op);
    op->setAttr(StartingOffsetAttrName,
                rewriter.getIntegerAttr(rewriter.getI64Type(), startVal));
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
    patterns.insert<AddFuncAttribute, AddCallAttribute>(ctx, funcAnalysisInfo);
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp op) {
      // If the function is a definition that doesn't have the attributes
      // applied, then it is illegal.
      return op.empty() || op.getPassthroughAttr();
    });
    target.addDynamicallyLegalOp<LLVM::CallOp>([](LLVM::CallOp op) {
      StringRef funcName = op.getCalleeAttr().getValue();
      return (!funcName.equals(cudaq::opt::QIRArrayQubitAllocateArray) &&
              !funcName.equals(cudaq::opt::QIRQubitAllocate)) ||
             op->hasAttr(StartingOffsetAttrName);
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
// Here we are replacing loads from calls to QIRArrayGetElementPtr1d or
// QIRQubitAllocate with a unique qubit index. Each allocated qubit was already
// assigned an index in the prep pass.
struct ArrayGetElementPtrConv : public OpRewritePattern<LLVM::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto bc = op.getAddr().getDefiningOp<LLVM::BitcastOp>();
    if (!bc)
      return failure();
    auto call = bc.getArg().getDefiningOp<LLVM::CallOp>();
    if (!call)
      return failure();
    auto loc = op.getLoc();
    if (call.getCallee()->equals(cudaq::opt::QIRArrayGetElementPtr1d)) {
      auto *alloc = call.getOperand(0).getDefiningOp();
      if (!alloc->hasAttr(StartingOffsetAttrName))
        return failure();
      Value disp = call.getOperand(1);
      Value off = rewriter.create<LLVM::ConstantOp>(
          loc, disp.getType(), alloc->getAttr(StartingOffsetAttrName));
      Value qubit = rewriter.create<LLVM::AddOp>(loc, off, disp);
      rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, op.getType(), qubit);
      return success();
    }
    return failure();
  }
};

struct CallAlloc : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (!call.getCallee()->equals(cudaq::opt::QIRQubitAllocate))
      return failure();
    if (!call->hasAttr(StartingOffsetAttrName))
      return failure();
    auto loc = call.getLoc();
    Value qubit = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), call->getAttr(StartingOffsetAttrName));
    auto resTy = call.getResult().getType();
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(call, resTy, qubit);
    return success();
  }
};

/// QIR to the Base Profile QIR
///
/// This pass converts patterns in LLVM-IR dialect using QIR calls, etc. into a
/// subset of QIR, the base profile. This pass uses a greedy rewrite to match
/// DAGs in the IR and replace them to meet the requirements of the base
/// profile. The patterns are defined in Peephole.td.
struct QIRToBaseProfileQIRPass
    : public cudaq::opt::QIRToBaseQIRBase<QIRToBaseProfileQIRPass> {
  explicit QIRToBaseProfileQIRPass() = default;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before base profile:\n" << *op << '\n');
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    // Note: LoadMeasureResult is not compliant with the Base Profile
    patterns.insert<AddrOfCisToBase, ArrayGetElementPtrConv, CallAlloc,
                    CalleeConv, EraseArrayAlloc, EraseArrayRelease,
                    EraseDeadArrayGEP, MeasureCallConv,
                    MeasureToRegisterCallConv, XCtrlOneTargetToCNot>(context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After base profile:\n" << *op << '\n');
  }

private:
  FrozenRewritePatternSet patterns;
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

static const std::vector<std::string> measurementFunctionNames{
    cudaq::opt::QIRMeasureBody, cudaq::opt::QIRMeasure,
    cudaq::opt::QIRMeasureToRegister};

struct BaseProfilePreparationPass
    : public cudaq::opt::QIRToBaseQIRPrepBase<BaseProfilePreparationPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();

    // Add cnot declaration as it may be
    // referenced after peepholes run.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRCnot, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getQubitType(ctx), cudaq::opt::getQubitType(ctx)}, module);

    // Add measure_body as it has a different
    // signature than measure.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRMeasureBody, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getQubitType(ctx), cudaq::opt::getResultType(ctx)},
        module);

    // Add record functions for any
    // measurements.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRBaseProfileRecordOutput, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getResultType(ctx), cudaq::opt::getCharPointerType(ctx)},
        module);

    // Add functions
    // `__quantum__qis__*__body` for all
    // functions matching
    // `__quantum__qis__*` that are found.
    for (auto &global : module)
      if (auto func = dyn_cast<LLVM::LLVMFuncOp>(global))
        if (needsToBeRenamed(func.getName()))
          cudaq::opt::factory::createLLVMFunctionSymbol(
              func.getName().str() + "__body",
              func.getFunctionType().getReturnType(),
              func.getFunctionType().getParams(), module);

    // Apply irreversible attribute to measurement functions
    for (auto &funcName : measurementFunctionNames) {
      Operation *op = SymbolTable::lookupSymbolIn(module, funcName);
      auto funcOp = llvm::dyn_cast_or_null<LLVM::LLVMFuncOp>(op);
      if (funcOp) {
        auto builder = OpBuilder(op);
        auto arrAttr = builder.getArrayAttr(
            ArrayRef<Attribute>{builder.getStringAttr("irreversible")});
        funcOp.setPassthroughAttr(arrAttr);
      }
    }
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
    auto *ctx = &getContext();
    func.walk([&](Operation *op) {
      if (auto call = dyn_cast<LLVM::CallOp>(op)) {
        auto funcName = call.getCalleeAttr().getValue();
        if (!funcName.startswith("__quantum_")) {
          call.emitOpError("unexpected call in QIR base profile");
          passFailed = true;
          return WalkResult::advance();
        }

        // Check that qubits are unique values.
        const std::size_t numOpnds = call.getNumOperands();
        auto qubitTy = cudaq::opt::getQubitType(ctx);
        if (numOpnds > 0)
          for (std::size_t i = 0; i < numOpnds - 1; ++i)
            if (call.getOperand(i).getType() == qubitTy)
              for (std::size_t j = i + 1; j < numOpnds; ++j)
                if (call.getOperand(j).getType() == qubitTy) {
                  auto i1 =
                      call.getOperand(i).getDefiningOp<LLVM::IntToPtrOp>();
                  auto j1 =
                      call.getOperand(j).getDefiningOp<LLVM::IntToPtrOp>();
                  if (i1 && j1 && i1.getOperand() == j1.getOperand()) {
                    call.emitOpError("uses same qubit as multiple operands");
                    passFailed = true;
                    return WalkResult::interrupt();
                  }
                }
        return WalkResult::advance();
      }
      if (isa<LLVM::BrOp, LLVM::CondBrOp, LLVM::ResumeOp, LLVM::UnreachableOp,
              LLVM::SwitchOp>(op)) {
        op->emitOpError("QIR base profile does not support control-flow");
        passFailed = true;
      }
      return WalkResult::advance();
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

namespace cudaq {

struct EraseMeasurements : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  void initialize() { setDebugName("EraseMeasurements"); }

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (auto callee = call.getCallee()) {
      if (callee->equals(cudaq::opt::QIRMeasureBody) ||
          callee->equals(cudaq::opt::QIRBaseProfileRecordOutput)) {
        rewriter.eraseOp(call);
        return success();
      }
    }
    return failure();
  }
};

/// Remove Measurements
///
/// This pass removes measurements and the corresponding output recording calls.
/// This is needed for backends that don't support selective measurement calls.
/// For example: https://github.com/NVIDIA/cuda-quantum/issues/512
struct RemoveMeasurementsPass
    : public cudaq::opt::RemoveMeasurementsBase<RemoveMeasurementsPass> {
  explicit RemoveMeasurementsPass() = default;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<EraseMeasurements>(context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  FrozenRewritePatternSet patterns;
};
} // namespace cudaq

std::unique_ptr<Pass> cudaq::opt::createRemoveMeasurementsPass() {
  return std::make_unique<RemoveMeasurementsPass>();
}
