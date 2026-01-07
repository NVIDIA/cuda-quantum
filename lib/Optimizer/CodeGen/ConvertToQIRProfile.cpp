/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Peephole.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
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

#define DEBUG_TYPE "qir-profile"

/**
   \file

   This file maps full QIR to either the Adaptive Profile QIR or Base Profile
   QIR. It is generally assumed that the input QIR here will be generated after
   the quake-synth pass, thereby greatly simplifying the transformations
   required here.

   This pass \e only supports QIR version 0.1.
 */

using namespace mlir;

#include "PeepholePatterns.inc"

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
    return funcName == cudaq::opt::QIRArraySlice;
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
  // Store by result to prevent collisions on a single qubit having
  // multiple measurements (Adaptive Profile)
  // map[result] --> [qb,regName]
  // Use std::map to keep these sorted in ascending order. While this isn't
  // required, it makes viewing the QIR easier.
  std::map<std::size_t, std::pair<std::size_t, std::string>> resultQubitVals;

  // resultOperation[QIR Result Number] = corresponding measurement op
  DenseMap<std::size_t, Operation *> resultOperation;
  DenseMap<Operation *, std::size_t> allocationOffsets;
};

using FunctionAnalysisInfo = DenseMap<Operation *, FunctionAnalysisData>;

/// The analysis on an entry function.
struct FunctionProfileAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FunctionProfileAnalysis)

  explicit FunctionProfileAnalysis(Operation *op) { performAnalysis(op); }

  const FunctionAnalysisInfo &getAnalysisInfo() const { return infoMap; }

private:
  static bool isAllocateArray(StringRef f) {
    return f == cudaq::opt::QIRArrayQubitAllocateArray ||
           f == cudaq::opt::QIRArrayQubitAllocateArrayWithStateFP64 ||
           f == cudaq::opt::QIRArrayQubitAllocateArrayWithStateFP32 ||
           f == cudaq::opt::QIRArrayQubitAllocateArrayWithStateComplex64 ||
           f == cudaq::opt::QIRArrayQubitAllocateArrayWithStateComplex32;
  }

  // Scan the body of a function for ops that will be used for profiling.
  void performAnalysis(Operation *operation) {
    auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(operation);
    if (!funcOp)
      return;
    FunctionAnalysisData data;
    funcOp->walk([&](LLVM::CallOp callOp) {
      auto funcNameAttr = callOp.getCalleeAttr();
      if (!funcNameAttr)
        return;
      auto funcName = funcNameAttr.getValue();

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

      if (isAllocateArray(funcName)) {
        addAllocation([&]() { return getNumQubits(callOp); });
      } else if (funcName == cudaq::opt::QIRQubitAllocate) {
        addAllocation([]() { return 1; });
      } else if (funcName == cudaq::opt::QIRMeasure ||
                 // FIXME Store the register names for the record_output
                 // functions
                 funcName == cudaq::opt::QIRMeasureToRegister) {
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
          auto *ctx = callOp.getContext();
          auto intTy = IntegerType::get(ctx, 64);
          auto resIdx = IntegerAttr::get(intTy, data.nResults);
          callOp->setAttr(resultIndexName, resIdx);
          auto regName = [&]() -> StringAttr {
            if (auto nameAttr = callOp->getAttr(cudaq::opt::QIRRegisterNameAttr)
                                    .dyn_cast_or_null<StringAttr>())
              return nameAttr;
            return {};
          }();
          data.resultOperation.insert({data.nResults, callOp.getOperation()});
          data.resultQubitVals.insert(std::make_pair(
              data.nResults++, std::make_pair(qb, regName.data())));
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
  explicit AddFuncAttribute(MLIRContext *ctx, const FunctionAnalysisInfo &info,
                            llvm::StringRef convertTo_)
      : OpRewritePattern(ctx), infoMap(info), convertTo(convertTo_) {}

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp op,
                                PatternRewriter &rewriter) const override {
    // Rewrite the exit block.
    // Add attributes to the function.
    auto iter = infoMap.find(op);
    assert(iter != infoMap.end());
    rewriter.startRootUpdate(op);
    const auto &info = iter->second;
    nlohmann::json resultQubitJSON{info.resultQubitVals};
    bool isAdaptive = convertTo == "qir-adaptive";
    const char *profileName = isAdaptive ? "adaptive_profile" : "base_profile";

    auto requiredQubitsStr = std::to_string(info.nQubits);
    StringRef requiredQubitsStrRef = requiredQubitsStr;
    if (auto stringAttr =
            op->getAttr(cudaq::opt::qir0_1::RequiredQubitsAttrName)
                .dyn_cast_or_null<mlir::StringAttr>())
      requiredQubitsStrRef = stringAttr;
    auto requiredResultsStr = std::to_string(info.nResults);
    StringRef requiredResultsStrRef = requiredResultsStr;
    if (auto stringAttr =
            op->getAttr(cudaq::opt::qir0_1::RequiredResultsAttrName)
                .dyn_cast_or_null<mlir::StringAttr>())
      requiredResultsStrRef = stringAttr;
    StringRef outputNamesStrRef;
    std::string resultQubitJSONStr;
    if (auto strAttr = op->getAttr(cudaq::opt::QIROutputNamesAttrName)
                           .dyn_cast_or_null<mlir::StringAttr>()) {
      outputNamesStrRef = strAttr;
    } else {
      resultQubitJSONStr = resultQubitJSON.dump();
      outputNamesStrRef = resultQubitJSONStr;
    }

    // QIR functions need certain attributes, add them here.
    // This pass is deprecated and will always use QIR 0.1. Future extensions
    // are not required either.
    SmallVector<Attribute> attrArray{
        rewriter.getStringAttr(cudaq::opt::QIREntryPointAttrName),
        rewriter.getStrArrayAttr(
            {cudaq::opt::QIRProfilesAttrName, profileName}),
        rewriter.getStrArrayAttr(
            {cudaq::opt::QIROutputLabelingSchemaAttrName, "schema_id"}),
        rewriter.getStrArrayAttr(
            {cudaq::opt::QIROutputNamesAttrName, outputNamesStrRef}),
        rewriter.getStrArrayAttr(
            {cudaq::opt::qir0_1::RequiredQubitsAttrName, requiredQubitsStrRef}),
        rewriter.getStrArrayAttr(
            // This pass is deprecated and will always use QIR 0.1.
            {cudaq::opt::qir0_1::RequiredResultsAttrName,
             requiredResultsStrRef})};

    op.setPassthroughAttr(rewriter.getArrayAttr(attrArray));

    // Stick the record calls in the exit block.
    auto builder = cudaq::IRBuilder::atBlockTerminator(&op.getBody().back());
    auto loc = op.getBody().back().getTerminator()->getLoc();

    auto resultTy = cudaq::opt::getResultType(rewriter.getContext());
    auto i64Ty = rewriter.getI64Type();
    auto module = op->getParentOfType<ModuleOp>();
    for (auto &iv : info.resultQubitVals) {
      auto &rec = iv.second;
      // All measurements must come at the end of Base Profile programs, but
      // Adaptive Profile programs are permitted (and in some cases required) to
      // interleave the output recording calls in the QIR.
      if (isAdaptive)
        builder.setInsertionPointAfter(
            info.resultOperation.find(iv.first)->getSecond());
      Value idx = builder.create<LLVM::ConstantOp>(loc, i64Ty, iv.first);
      Value ptr = builder.create<LLVM::IntToPtrOp>(loc, resultTy, idx);
      auto regName = [&]() -> Value {
        auto charPtrTy = cudaq::opt::getCharPointerType(builder.getContext());
        if (!rec.second.empty()) {
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
                                   cudaq::opt::QIRRecordOutput,
                                   ValueRange{ptr, regName});
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }

  const FunctionAnalysisInfo &infoMap;
  std::string convertTo;
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
    op->setAttr(cudaq::opt::StartingOffsetAttrName,
                rewriter.getIntegerAttr(rewriter.getI64Type(), startVal));
    rewriter.finalizeRootUpdate(op);
    return success();
  }

  const FunctionAnalysisInfo &infoMap;
};

/// QIR to Specific Profile QIR on the function level.
///
/// With FuncOps, we want to add attributes to the function op and also add
/// calls to the "record" API in the exit block of the function. The record
/// calls are bijective with all distinct measurement calls in the original
/// function, however the indices used may be renumbered and start at 0.
struct QIRToQIRProfileFuncPass
    : public cudaq::opt::QIRToQIRProfileFuncBase<QIRToQIRProfileFuncPass> {
  using QIRToQIRProfileFuncBase::QIRToQIRProfileFuncBase;

  explicit QIRToQIRProfileFuncPass(llvm::StringRef convertTo_)
      : QIRToQIRProfileFuncBase() {
    convertTo.setValue(convertTo_.str());
  }

  void runOnOperation() override {
    auto op = getOperation();
    auto *ctx = op.getContext();
    RewritePatternSet patterns(ctx);
    const auto &analysis = getAnalysis<FunctionProfileAnalysis>();
    const auto &funcAnalysisInfo = analysis.getAnalysisInfo();
    patterns.insert<AddFuncAttribute>(ctx, funcAnalysisInfo,
                                      convertTo.getValue());
    patterns.insert<AddCallAttribute>(ctx, funcAnalysisInfo);
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp op) {
      // If the function is a definition that doesn't have the attributes
      // applied, then it is illegal.
      return op.empty() || op.getPassthroughAttr();
    });
    target.addDynamicallyLegalOp<LLVM::CallOp>([](LLVM::CallOp op) {
      auto funcNameAttr = op.getCalleeAttr();
      if (!funcNameAttr)
        return true;
      auto funcName = funcNameAttr.getValue();
      return (funcName != cudaq::opt::QIRArrayQubitAllocateArray &&
              funcName != cudaq::opt::QIRQubitAllocate) ||
             op->hasAttr(cudaq::opt::StartingOffsetAttrName);
    });

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op.getLoc(), "failed to convert to " + convertTo.getValue());
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass>
cudaq::opt::createConvertToQIRFuncPass(llvm::StringRef convertTo) {
  return std::make_unique<QIRToQIRProfileFuncPass>(convertTo);
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
      if (!alloc->hasAttr(cudaq::opt::StartingOffsetAttrName))
        return failure();
      Value disp = call.getOperand(1);
      Value off = rewriter.create<LLVM::ConstantOp>(
          loc, disp.getType(),
          alloc->getAttr(cudaq::opt::StartingOffsetAttrName));
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
    if (!call->hasAttr(cudaq::opt::StartingOffsetAttrName))
      return failure();
    auto loc = call.getLoc();
    Value qubit = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(),
        call->getAttr(cudaq::opt::StartingOffsetAttrName));
    auto resTy = call.getResult().getType();
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(call, resTy, qubit);
    return success();
  }
};

// %1 = address_of @__quantum__qis__z__ctl
// %2 = call @invokewithControlBits %1, %ctrl, %targ
// ─────────────────────────────────────────────────
// %2 = call __quantum__qis__cz %ctrl, %targ
struct ZCtrlOneTargetToCZ : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    ValueRange args(call.getArgOperands());
    if (args.size() == 4 && call.getCallee() &&
        call.getCallee()->equals(cudaq::opt::NVQIRInvokeWithControlBits)) {
      if (auto addrOf = dyn_cast_or_null<mlir::LLVM::AddressOfOp>(
              args[1].getDefiningOp())) {
        if (addrOf.getGlobalName().startswith(
                std::string(cudaq::opt::QIRQISPrefix) + "z__ctl")) {
          rewriter.replaceOpWithNewOp<LLVM::CallOp>(
              call, TypeRange{}, cudaq::opt::QIRCZ, args.drop_front(2));
          return success();
        }
      }
    }
    return failure();
  }
};

/// QIR to the Specific QIR Profile
///
/// This pass converts patterns in LLVM-IR dialect using QIR calls, etc. into a
/// subset of QIR, the base profile. This pass uses a greedy rewrite to match
/// DAGs in the IR and replace them to meet the requirements of the base
/// profile. The patterns are defined in Peephole.td.
struct QIRToQIRProfileQIRPass
    : public cudaq::opt::QIRToQIRProfileBase<QIRToQIRProfileQIRPass> {
  explicit QIRToQIRProfileQIRPass() = default;

  /// @brief Construct pass
  /// @param convertTo_ expected "qir-base" or "qir-adaptive"
  QIRToQIRProfileQIRPass(llvm::StringRef convertTo_) : QIRToQIRProfileBase() {
    convertTo.setValue(convertTo_.str());
  }

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before QIR profile:\n" << *op << '\n');
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    // Note: LoadMeasureResult is not compliant with the Base Profile, so don't
    // add it here unless we're specifically doing the Adaptive Profile.
    patterns
        .insert<AddrOfCisToBase, ArrayGetElementPtrConv, CallAlloc, CalleeConv,
                EraseArrayAlloc, EraseArrayRelease, EraseDeadArrayGEP,
                MeasureCallConv, MeasureToRegisterCallConv,
                XCtrlOneTargetToCNot, ZCtrlOneTargetToCZ>(context);
    if (convertTo.getValue() == "qir-adaptive")
      patterns.insert<LoadMeasureResult>(context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After QIR profile:\n" << *op << '\n');
  }

private:
  FrozenRewritePatternSet patterns;
};
} // namespace

std::unique_ptr<Pass>
cudaq::opt::createQIRToQIRProfilePass(llvm::StringRef convertTo) {
  return std::make_unique<QIRToQIRProfileQIRPass>(convertTo);
}

//===----------------------------------------------------------------------===//

namespace {
/// QIR Profile Preparation:
///
/// Before we can do the conversion to the QIR profile with different threads
/// running on different functions, the module is updated with the signatures of
/// functions from the QIR ABI that may be called by the translation. This
/// trivial pass only does this preparation work. It performs no analysis and
/// does not rewrite function body's, etc.

static constexpr std::array<const char *, 3> measurementFunctionNames{
    cudaq::opt::QIRMeasureBody, cudaq::opt::QIRMeasure,
    cudaq::opt::QIRMeasureToRegister};

struct QIRProfilePreparationPass
    : public cudaq::opt::QIRToQIRProfilePrepBase<QIRProfilePreparationPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();

    // Add cnot declaration as it may be referenced after peepholes run.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRCnot, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getQubitType(ctx), cudaq::opt::getQubitType(ctx)}, module);

    // Add cz declaration as it may be referenced after peepholes run.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRCZ, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getQubitType(ctx), cudaq::opt::getQubitType(ctx)}, module);

    // Add measure_body as it has a different signature than measure.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRMeasureBody, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getQubitType(ctx), cudaq::opt::getResultType(ctx)},
        module);

    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::qir0_1::ReadResultBody, IntegerType::get(ctx, 1),
        {cudaq::opt::getResultType(ctx)}, module);

    // Add record functions for any measurements.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRRecordOutput, LLVM::LLVMVoidType::get(ctx),
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

    // Apply irreversible attribute to measurement functions
    for (auto *funcName : measurementFunctionNames) {
      Operation *op = SymbolTable::lookupSymbolIn(module, funcName);
      auto funcOp = llvm::dyn_cast_if_present<LLVM::LLVMFuncOp>(op);
      if (funcOp) {
        auto builder = OpBuilder(op);
        auto arrAttr = builder.getArrayAttr(ArrayRef<Attribute>{
            builder.getStringAttr(cudaq::opt::QIRIrreversibleFlagName)});
        funcOp.setPassthroughAttr(arrAttr);
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createQIRProfilePreparationPass() {
  return std::make_unique<QIRProfilePreparationPass>();
}

//===----------------------------------------------------------------------===//
// The various passes defined here should be added as a pass pipeline.

void cudaq::opt::addQIRProfileVerify(OpPassManager &pm,
                                     llvm::StringRef convertTo) {
  VerifyQIRProfileOptions vqpo = {convertTo.str()};
  pm.addNestedPass<LLVM::LLVMFuncOp>(createVerifyQIRProfile(vqpo));
}

void cudaq::opt::addQIRProfilePipeline(OpPassManager &pm,
                                       llvm::StringRef convertTo) {
  assert(convertTo == "qir-adaptive" || convertTo == "qir-base");
  pm.addPass(createQIRProfilePreparationPass());
  pm.addNestedPass<LLVM::LLVMFuncOp>(createConvertToQIRFuncPass(convertTo));
  pm.addPass(createQIRToQIRProfilePass(convertTo));
  addQIRProfileVerify(pm, convertTo);
}
