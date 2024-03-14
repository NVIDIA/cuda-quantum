/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Peephole.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
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

/// This file maps full QIR to either the Adaptive Profile QIR or Base Profile
/// QIR. It is generally assumed that the input QIR here will be generated after
/// the quake-synth pass, thereby greatly simplifying the transformations
/// required here.

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

static Value createIntToQIRIntrinsic(Location loc, OpBuilder &builder,
                                     Type resultTy, StringRef name,
                                     Value intArg) {
  auto i64Ty = builder.getI64Type();
  auto fnTy = LLVM::LLVMFunctionType::get(resultTy, ArrayRef<Type>{i64Ty});
  return builder.create<LLVM::CallOp>(loc, fnTy, name, ArrayRef<Value>{intArg})
      .getResult();
}

static Value createIntToQIRIntrinsic(Location loc, OpBuilder &builder,
                                     Type resultTy, StringRef name,
                                     std::int64_t intArg) {
  auto i64Ty = builder.getI64Type();
  Value idx = builder.create<LLVM::ConstantOp>(loc, i64Ty, intArg);
  return createIntToQIRIntrinsic(loc, builder, resultTy, name, idx);
}

static Value createIntToQubitIntrinsic(Location loc, OpBuilder &builder,
                                       std::int64_t intArg) {
  return createIntToQIRIntrinsic(loc, builder,
                                 cudaq::opt::getQubitType(builder.getContext()),
                                 cudaq::opt::LLVMIntToQubit, intArg);
}

static Value createIntToQubitIntrinsic(Location loc, OpBuilder &builder,
                                       Value idx) {
  return createIntToQIRIntrinsic(loc, builder,
                                 cudaq::opt::getQubitType(builder.getContext()),
                                 cudaq::opt::LLVMIntToQubit, idx);
}

static Value createIntToResultIntrinsic(Location loc, OpBuilder &builder,
                                        std::int64_t intArg) {
  return createIntToQIRIntrinsic(
      loc, builder, cudaq::opt::getResultType(builder.getContext()),
      cudaq::opt::LLVMIntToResult, intArg);
}

static bool isConstructResultOp(Value v) {
  if (auto call = v.getDefiningOp<LLVM::CallOp>())
    if (auto callee = call.getCallee())
      return callee->equals(cudaq::opt::LLVMIntToResult);
  return false;
}

static bool isConstructQubitOp(Value v) {
  if (auto call = v.getDefiningOp<LLVM::CallOp>())
    if (auto callee = call.getCallee())
      return callee->equals(cudaq::opt::LLVMIntToQubit);
  return false;
}

static bool isResultAddrOp(Value v) {
  if (auto call = v.getDefiningOp<LLVM::CallOp>())
    if (auto callee = call.getCallee())
      return callee->equals(cudaq::opt::LLVMResultAddr);
  return false;
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
        if (auto call = callOp.getOperand(0).getDefiningOp<LLVM::CallOp>()) {
          if (auto callee = call.getCallee()) {
            if (callee->equals(cudaq::opt::QIRArrayGetElementPtr1d))
              if (auto conOp =
                      call.getOperand(1).getDefiningOp<LLVM::ConstantOp>())
                if (auto intAttr = dyn_cast<IntegerAttr>(conOp.getValue()))
                  optQb = intAttr.getInt();
          } else {
            auto iter = data.allocationOffsets.find(call.getOperation());
            if (iter != data.allocationOffsets.end())
              optQb = iter->second;
          }
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
            if (auto nameAttr = callOp->getAttr("registerName")
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
    rewriter.startOpModification(op);
    const auto &info = iter->second;
    nlohmann::json resultQubitJSON{info.resultQubitVals};
    bool isAdaptive = convertTo == "qir-adaptive";
    const char *profileName = isAdaptive ? "adaptive_profile" : "base_profile";

    // QIR functions need certain attributes, add them here.
    // TODO: Update schema_id with valid value (issues #385 and #556)
    auto arrAttr = rewriter.getArrayAttr(ArrayRef<Attribute>{
        rewriter.getStringAttr("entry_point"),
        rewriter.getStrArrayAttr({"qir_profiles", profileName}),
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
      auto ptr = createIntToResultIntrinsic(loc, builder, iv.first);
      auto regName = [&]() -> Value {
        if (!rec.second.empty()) {
          // Note: it should be the case that this string literal has already
          // been added to the IR, so this step does not actually update the
          // module.
          auto globl =
              builder.genCStringLiteralAppendNul(loc, module, rec.second);
          return builder.create<LLVM::AddressOfOp>(
              loc, LLVM::LLVMPointerType::get(builder.getContext()),
              globl.getName());
        }
        Value zero = builder.create<LLVM::ConstantOp>(loc, i64Ty, 0);
        return builder.create<LLVM::IntToPtrOp>(
            loc, LLVM::LLVMPointerType::get(builder.getContext()), zero);
      }();
      builder.create<LLVM::CallOp>(loc, TypeRange{},
                                   cudaq::opt::QIRRecordOutput,
                                   ValueRange{ptr, regName});
    }
    rewriter.finalizeOpModification(op);
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
    rewriter.startOpModification(op);
    op->setAttr(StartingOffsetAttrName,
                rewriter.getIntegerAttr(rewriter.getI64Type(), startVal));
    rewriter.finalizeOpModification(op);
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
      StringRef funcName = op.getCalleeAttr().getValue();
      return (!funcName.equals(cudaq::opt::QIRArrayQubitAllocateArray) &&
              !funcName.equals(cudaq::opt::QIRQubitAllocate)) ||
             op->hasAttr(StartingOffsetAttrName);
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
struct ArrayGetElementPtrConv : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<StringRef> callee = op.getCallee();
    if (!callee)
      return failure();
    auto loc = op.getLoc();
    if (!callee->equals(cudaq::opt::QIRArrayGetElementPtr1d))
      return failure();
    auto *alloc = op.getOperand(0).getDefiningOp();
    if (!alloc->hasAttr(StartingOffsetAttrName))
      return failure();
    Value disp = op.getOperand(1);
    Value off = rewriter.create<LLVM::ConstantOp>(
        loc, disp.getType(), alloc->getAttr(StartingOffsetAttrName));
    Value qubit = rewriter.create<LLVM::AddOp>(loc, off, disp);
    Value v = createIntToQubitIntrinsic(loc, rewriter, qubit);
    rewriter.replaceOp(op, v);
    return success();
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
    Value qubit = createIntToQubitIntrinsic(
        loc, rewriter,
        cast<IntegerAttr>(call->getAttr(StartingOffsetAttrName)).getInt());
    rewriter.replaceOp(call, qubit);
    return success();
  }
};

// %1 = address_of @__quantum__qis__x__ctl
// %2 = call @invokewithControlBits %1, %ctrl, %targ
// ─────────────────────────────────────────────────
// %2 = call __quantum__qis__cnot %ctrl, %targ
struct XCtrlOneTargetToCNot : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto args = call.getOperands();
    if (auto callee = call.getCallee())
      if (!callToInvokeWithXCtrlOneTarget(*callee, args))
        return failure();
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        call, mlir::TypeRange{}, cudaq::opt::QIRCnot, args.drop_front(2));
    return success();
  }
};

// %4 = call @__quantum__cis__*
// ──────────────────────────────────
// %4 = call @__quantum__cis__*__body
struct CalleeConv : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    std::optional<StringRef> callee = call.getCallee();
    if (!callee)
      return failure();
    if (!needsToBeRenamed(*callee))
      return failure();
    if (callee->starts_with(cudaq::opt::QIRMeasure))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        call, TypeRange{}, callee->str() + "__body", call.getOperands());
    return success();
  }
};

struct EraseDeadArrayGEP : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    std::optional<StringRef> callee = call.getCallee();
    if (!callee)
      return failure();
    if (!callee->equals(cudaq::opt::QIRArrayGetElementPtr1d))
      return failure();
    if (!call->use_empty())
      return failure();
    auto *context = rewriter.getContext();
    auto qubitTy = cudaq::opt::getQubitType(context);
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(call, qubitTy);
    return success();
  }
};

// Replace the call with a dead op to DCE.
//
// %0 = call @allocate ... : ... -> T*
// ───────────────────────────────────
// %0 = undef : T*
struct EraseArrayAlloc : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    std::optional<StringRef> callee = call.getCallee();
    if (!callee)
      return failure();
    if (!callee->equals(cudaq::opt::QIRArrayQubitAllocateArray))
      return failure();
    auto *context = rewriter.getContext();
    auto arrayTy = cudaq::opt::getArrayType(context);
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(call, arrayTy);
    return success();
  }
};

// Remove the release calls. This removes both array allocations as well as
// qubit singletons.
//
// call @release %5 : (!Qubit) -> ()
// ─────────────────────────────────
//
struct EraseArrayRelease : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    std::optional<StringRef> callee = call.getCallee();
    if (!callee)
      return failure();
    if (!(callee->equals(cudaq::opt::QIRArrayQubitReleaseArray) ||
          callee->equals(cudaq::opt::QIRArrayQubitReleaseQubit)))
      return failure();
    rewriter.eraseOp(call);
    return success();
  }
};

// Do the following two rewrites:
//
//   %result = call @__quantum__qis__mz(%qbit) : (!Qubit) -> i1
//   ──────────────────────────────────────────────────────────────
//   call @__quantum__qis__mz_body(%qbit, %result) : (Q*, R*) -> ()
//
// and
//
//   %r = call @__quantum__qis__mz__to__register(%q, i8) : (!Qubit) -> i1
//   ────────────────────────────────────────────────────────────────────
//   call @__quantum__qis__mz_body(%q, %r) : (Q*, R*) -> ()
struct MeasureCallConv : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    std::optional<StringRef> callee = call.getCallee();
    if (!callee)
      return failure();
    if (!(callee->equals(cudaq::opt::QIRMeasure) ||
          callee->equals(cudaq::opt::QIRMeasureToRegister)))
      return failure();
    auto args = call.getOperands();
    if (!isConstructQubitOp(args[0]))
      return failure();
    auto intAttr =
        dyn_cast_or_null<IntegerAttr>(call->getAttr(resultIndexName));
    if (!intAttr)
      return failure();
    auto loc = call.getLoc();
    auto resultOp = createIntToResultIntrinsic(loc, rewriter, intAttr.getInt());
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, cudaq::opt::QIRMeasureBody,
                                  ArrayRef<Value>{args[0], resultOp});
    rewriter.replaceOp(call, resultOp);
    return success();
  }
};

// %4 = address_of @__quantum__cis__*
// ────────────────────────────────────────
// %4 = address_of @__quantum__cis__*__body
struct AddrOfCisToBase : public OpRewritePattern<LLVM::AddressOfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AddressOfOp addr,
                                PatternRewriter &rewriter) const override {
    auto global = addr.getGlobalName();
    if (!needsToBeRenamed(global))
      return failure();
    auto addrTy = addr.getType();
    std::string newName(global.str() + "__body");
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(addr, addrTy, newName);
    return success();
  }
};

// This is applied \em{only} for adaptive profile.
//
//   %1 = llvm.constant 1
//   %2 = llvm.call @llvm.qir.i64ToResult %1 : (i64) -> target(Result)
//   %3 = llvm.call @llvm.qir.getResultPtr %2 : (target(Result)) -> i1*
//   %4 = llvm.load %3
//   ──────────────────────────────────────────────────────────────────
//   %4 = call @read_result %2
struct LoadMeasureResult : public OpRewritePattern<LLVM::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp load,
                                PatternRewriter &rewriter) const override {
    if (!isResultAddrOp(load.getOperand()))
      return failure();
    auto call1 = load.getOperand().getDefiningOp<LLVM::CallOp>();
    if (!isConstructResultOp(call1.getOperand(0)))
      return failure();
    auto call2 = load.getOperand().getDefiningOp<LLVM::CallOp>();
    auto intConst = call2.getOperand(0).getDefiningOp<LLVM::ConstantOp>();
    if (!intConst)
      return failure();
    if (!isa<IntegerAttr>(intConst.getValue()))
      return failure();
    auto loc = load.getLoc();
    Value newVal = createReadResultCall(rewriter, loc, call2.getResult());
    rewriter.replaceOp(load, newVal);
    return success();
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
    auto *context = op->getContext();
    LLVM_DEBUG(llvm::dbgs() << "Before QIR profile:\n" << *op << '\n');
    RewritePatternSet patterns(context);
    // Note: LoadMeasureResult is not compliant with the Base Profile, so don't
    // add it here unless we're specifically doing the Adaptive Profile.
    patterns.insert<AddrOfCisToBase, ArrayGetElementPtrConv, CallAlloc,
                    CalleeConv, EraseArrayAlloc, EraseArrayRelease,
                    EraseDeadArrayGEP, MeasureCallConv, XCtrlOneTargetToCNot>(
        context);
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

static const std::vector<std::string> measurementFunctionNames{
    cudaq::opt::QIRMeasureBody, cudaq::opt::QIRMeasure,
    cudaq::opt::QIRMeasureToRegister};

struct QIRProfilePreparationPass
    : public cudaq::opt::QIRToQIRProfilePrepBase<QIRProfilePreparationPass> {

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

    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRReadResultBody, IntegerType::get(ctx, 1),
        {cudaq::opt::getResultType(ctx)}, module);

    // Add record functions for any
    // measurements.
    cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRRecordOutput, LLVM::LLVMVoidType::get(ctx),
        {cudaq::opt::getResultType(ctx), LLVM::LLVMPointerType::get(ctx)},
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

std::unique_ptr<Pass> cudaq::opt::createQIRProfilePreparationPass() {
  return std::make_unique<QIRProfilePreparationPass>();
}

//===----------------------------------------------------------------------===//

namespace {
/// Verify that the specific profile QIR code is sane. For now, this simply
/// checks that the QIR doesn't have any "bonus" calls to arbitrary code that is
/// not possibly defined in the QIR standard.
struct VerifyQIRProfilePass
    : public cudaq::opt::VerifyQIRProfileBase<VerifyQIRProfilePass> {
  explicit VerifyQIRProfilePass(llvm::StringRef convertTo_)
      : VerifyQIRProfileBase() {
    convertTo.setValue(convertTo_.str());
  }

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    bool passFailed = false;
    if (!func->hasAttr(cudaq::entryPointAttrName))
      return;
    auto *ctx = &getContext();
    bool isBaseProfile = convertTo.getValue() == "qir-base";
    func.walk([&](Operation *op) {
      if (auto call = dyn_cast<LLVM::CallOp>(op)) {
        auto funcName = call.getCalleeAttr().getValue();
        if (!(funcName.starts_with("__quantum_") ||
              funcName.starts_with("llvm.qir."))) {
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
                  auto i1 = isConstructQubitOp(call.getOperand(i));
                  auto j1 = isConstructQubitOp(call.getOperand(j));
                  if (i1 && j1) {
                    auto *i1Op = call.getOperand(i).getDefiningOp();
                    auto *j1Op = call.getOperand(j).getDefiningOp();
                    if (i1Op->getOperand(0) == j1Op->getOperand(0)) {
                      call.emitOpError("uses same qubit as multiple operands");
                      passFailed = true;
                      return WalkResult::interrupt();
                    }
                  }
                }
        return WalkResult::advance();
      }
      if (isBaseProfile && isa<LLVM::BrOp, LLVM::CondBrOp, LLVM::ResumeOp,
                               LLVM::UnreachableOp, LLVM::SwitchOp>(op)) {
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

std::unique_ptr<Pass>
cudaq::opt::verifyQIRProfilePass(llvm::StringRef convertTo) {
  return std::make_unique<VerifyQIRProfilePass>(convertTo);
}

namespace {
/// Verify that the QIR doesn't have any "bonus" calls to arbitrary code that is
/// not possibly defined in the QIR standard or NVQIR runtime.
struct VerifyNVQIRCallOpsPass
    : public cudaq::opt::VerifyNVQIRCallOpsBase<VerifyNVQIRCallOpsPass> {
  explicit VerifyNVQIRCallOpsPass(
      const std::vector<llvm::StringRef> &allowedFuncs)
      : VerifyNVQIRCallOpsBase(), allowedFuncs(allowedFuncs) {}

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    bool passFailed = false;
    // Check that a function name is either QIR or NVQIR registered.
    const auto isKnownFunctionName = [&](llvm::StringRef functionName) -> bool {
      if (functionName.starts_with("__quantum_"))
        return true;
      static const std::vector<llvm::StringRef> NVQIR_FUNCS = {
          cudaq::opt::NVQIRInvokeWithControlBits,
          cudaq::opt::NVQIRInvokeRotationWithControlBits,
          cudaq::opt::NVQIRInvokeWithControlRegisterOrBits,
          cudaq::opt::NVQIRPackSingleQubitInArray,
          cudaq::opt::NVQIRReleasePackedQubitArray,

          // These are required to be added to the QIR Spec.
          cudaq::opt::LLVMIntToQubit, cudaq::opt::LLVMIntToResult,
          cudaq::opt::LLVMIntToArray, cudaq::opt::LLVMResultAddr};

      // It must be either NVQIR extension functions or in the allowed list.
      return std::find(NVQIR_FUNCS.begin(), NVQIR_FUNCS.end(), functionName) !=
                 NVQIR_FUNCS.end() ||
             std::find(allowedFuncs.begin(), allowedFuncs.end(),
                       functionName) != allowedFuncs.end();
    };

    func.walk([&](Operation *op) {
      if (auto call = dyn_cast<LLVM::CallOp>(op)) {
        auto funcName = call.getCalleeAttr().getValue();
        if (!isKnownFunctionName(funcName)) {
          call.emitOpError("unexpected function call in NVQIR");
          passFailed = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    if (passFailed) {
      emitError(func.getLoc(),
                "function " + func.getName() + " not compatible with NVQIR.");
      signalPassFailure();
    }
  }

private:
  std::vector<llvm::StringRef> allowedFuncs;
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createVerifyNVQIRCallOpsPass(
    const std::vector<llvm::StringRef> &allowedFuncs) {
  return std::make_unique<VerifyNVQIRCallOpsPass>(allowedFuncs);
}

// The various passes defined here should be added as a pass pipeline.

void cudaq::opt::addQIRProfilePipeline(OpPassManager &pm,
                                       llvm::StringRef convertTo) {
  assert(convertTo == "qir-adaptive" || convertTo == "qir-base");
  pm.addPass(createQIRProfilePreparationPass());
  pm.addNestedPass<LLVM::LLVMFuncOp>(createConvertToQIRFuncPass(convertTo));
  pm.addPass(createQIRToQIRProfilePass(convertTo));
  pm.addNestedPass<LLVM::LLVMFuncOp>(verifyQIRProfilePass(convertTo));
}

namespace cudaq {

struct EraseMeasurements : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  void initialize() { setDebugName("EraseMeasurements"); }

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (auto callee = call.getCallee()) {
      if (callee->equals(cudaq::opt::QIRMeasureBody) ||
          callee->equals(cudaq::opt::QIRRecordOutput)) {
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
