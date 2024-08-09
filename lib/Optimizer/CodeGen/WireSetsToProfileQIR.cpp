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
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QuakeToCC.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "wireset-to-profile-qir"

/**
   \file

   If the Quake code is using wire sets (referencing discrete "physical" quantum
   units), then codegen should not use full QIR. Full QIR uses virtual qubits,
   so the physical mapping would be completely lost.

   This pass translates Quake that uses wire sets to QIR calls (in the
   CC dialect and FuncDialect), which can themselves be further lowered to
   LLVM-IR dialect using the CCToLLVM lowering passes.

   Prerequisites:
   The Quake IR should be
     - in DAG form (no CC control flow operations or calls)
     - using value semantics and wire_set globals
     - decomposed into single control (at most) gate form
     - negated controls must have been erased
 */

namespace cudaq::opt {
#define GEN_PASS_DEF_WIRESETTOPROFILEQIR
#define GEN_PASS_DEF_WIRESETTOPROFILEQIRPOST
#define GEN_PASS_DEF_WIRESETTOPROFILEQIRPREP
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
struct QuakeTypeConverter : public TypeConverter {
  QuakeTypeConverter() {
    addConversion([](Type ty) { return ty; });
    addConversion([](quake::WireType ty) {
      return cudaq::opt::getQubitType(ty.getContext());
    });
    addConversion([](quake::MeasureType ty) {
      return cudaq::opt::getResultType(ty.getContext());
    });
  }
};
} // namespace

static constexpr std::string_view qis_prefix = "__quantum__qis__";
static constexpr std::string_view qis_body_suffix = "__body";
static constexpr std::string_view qis_ctl_suffix = "__ctl";

static std::string toQisBodyName(std::string &&name) {
  return std::string(qis_prefix) + std::move(name) +
         std::string(qis_body_suffix);
}

static std::string toQisCtlName(std::string &&name) {
  return std::string(qis_prefix) + std::move(name) +
         std::string(qis_ctl_suffix);
}

template <typename OP>
struct GeneralRewrite : OpConversionPattern<OP> {
  using Base = OpConversionPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP qop, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto instName = qop->getName().stripDialect().str();
    if (qop.getIsAdj() && (instName == "t" || instName == "s"))
      instName += "dg";

    auto loc = qop.getLoc();
    std::string funcName = [&]() {
      if (qop.getControls().empty())
        return toQisBodyName(std::move(instName));
      if (instName == "x") {
        instName = "cnot";
        return toQisBodyName(std::move(instName));
      }
      return toQisCtlName(std::move(instName));
    }(); // NB: instName is dead
    SmallVector<Value> qubits(adaptor.getControls().begin(),
                              adaptor.getControls().end());
    qubits.append(adaptor.getTargets().begin(), adaptor.getTargets().end());
    rewriter.create<func::CallOp>(loc, std::nullopt, funcName,
                                  adaptor.getOperands());
    rewriter.replaceOp(qop, qubits);
    return success();
  }
};

namespace {
struct BorrowWireRewrite : OpConversionPattern<quake::BorrowWireOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::BorrowWireOp borrowWire, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto id = borrowWire.getIdentity();
    auto loc = borrowWire.getLoc();
    Value idCon = rewriter.create<arith::ConstantIntOp>(loc, id, 64);
    auto imTy =
        cudaq::cc::PointerType::get(NoneType::get(rewriter.getContext()));
    idCon = rewriter.create<cudaq::cc::CastOp>(loc, imTy, idCon);
    rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(
        borrowWire, cudaq::opt::getQubitType(rewriter.getContext()), idCon);
    return success();
  }
};

struct ReturnWireRewrite : OpConversionPattern<quake::ReturnWireOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::ReturnWireOp returnWire, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(returnWire);
    return success();
  }
};

struct WireSetRewrite : OpConversionPattern<quake::WireSetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::WireSetOp wireSetOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(wireSetOp);
    return success();
  }
};

struct MzRewrite : OpConversionPattern<quake::MzOp> {
  using Base = OpConversionPattern;
  explicit MzRewrite(TypeConverter &typeConverter, unsigned &counter,
                     MLIRContext *ctxt, PatternBenefit benefit = 1)
      : Base(typeConverter, ctxt, benefit), resultCount(counter) {}

  LogicalResult
  matchAndRewrite(quake::MzOp meas, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // FIXME: Must use sequentially assigned result ids
    std::string funcName = toQisBodyName(std::string("mz"));
    auto loc = meas.getLoc();
    Value idCon = rewriter.create<arith::ConstantIntOp>(loc, resultCount++, 64);
    auto imTy =
        cudaq::cc::PointerType::get(NoneType::get(rewriter.getContext()));
    idCon = rewriter.create<cudaq::cc::CastOp>(loc, imTy, idCon);
    Value resultVal = rewriter.create<cudaq::cc::CastOp>(
        loc, cudaq::opt::getResultType(rewriter.getContext()), idCon);
    rewriter.create<func::CallOp>(
        loc, std::nullopt, funcName,
        ValueRange{adaptor.getTargets()[0], resultVal});
    rewriter.replaceOp(meas, ValueRange{resultVal, adaptor.getTargets()[0]});
    return success();
  }

private:
  unsigned &resultCount;
};

struct DiscriminateRewrite : OpConversionPattern<quake::DiscriminateOp> {
  using Base = OpConversionPattern;

  explicit DiscriminateRewrite(TypeConverter &typeConverter, bool adaptive,
                               DenseMap<Operation *, StringRef> &nameMap,
                               MLIRContext *ctxt, PatternBenefit benefit = 1)
      : Base(typeConverter, ctxt, benefit), isAdaptiveProfile(adaptive),
        regNameMap(nameMap) {}

  LogicalResult
  matchAndRewrite(quake::DiscriminateOp disc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = disc.getLoc();

    auto mod = disc->getParentOfType<ModuleOp>();
    cudaq::IRBuilder irb(rewriter.getContext());
    auto iter = regNameMap.find(disc.getOperation());
    assert(iter != regNameMap.end() && "discriminate must be in map");
    // NB: This is thread safe as it should never do an insertion, just a
    // lookup.
    auto nameObj = irb.genCStringLiteralAppendNul(loc, mod, iter->second);
    auto arrI8Ty = mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(),
                                                  iter->second.size() + 1);
    auto ptrArrTy = cudaq::cc::PointerType::get(arrI8Ty);
    Value nameVal = rewriter.create<cudaq::cc::AddressOfOp>(loc, ptrArrTy,
                                                            nameObj.getName());
    auto cstrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    Value nameValCStr =
        rewriter.create<cudaq::cc::CastOp>(loc, cstrTy, nameVal);

    rewriter.create<func::CallOp>(
        loc, std::nullopt, "__quantum__rt__result_record_output",
        ValueRange{adaptor.getMeasurement(), nameValCStr});
    if (isAdaptiveProfile) {
      std::string funcName = toQisBodyName(std::string("read_result"));
      rewriter.replaceOpWithNewOp<func::CallOp>(
          disc, rewriter.getI1Type(), funcName,
          ValueRange{adaptor.getMeasurement()});
    } else {
      Value undef =
          rewriter.create<cudaq::cc::UndefOp>(loc, rewriter.getI1Type());
      rewriter.replaceOp(disc, undef);
    }
    return success();
  }

private:
  bool isAdaptiveProfile;
  DenseMap<Operation *, StringRef> &regNameMap;
};

struct WireSetToProfileQIRPass
    : public cudaq::opt::impl::WireSetToProfileQIRBase<
          WireSetToProfileQIRPass> {
  using WireSetToProfileQIRBase::WireSetToProfileQIRBase;

  void runOnOperation() override {
    auto op = getOperation();
    auto *context = &getContext();
    DenseMap<Operation *, StringRef> regNameMap;
    op.walk([&](quake::DiscriminateOp disc) {
      auto meas = disc.getMeasurement().getDefiningOp<quake::MzOp>();
      auto name = meas ? meas.getRegisterName() : std::nullopt;
      if (name)
        regNameMap[disc.getOperation()] = *name;
      else
        regNameMap[disc.getOperation()] = "?";
    });
    RewritePatternSet patterns(context);
    QuakeTypeConverter quakeTypeConverter;
    unsigned resultCounter = 0;
    patterns.insert<GeneralRewrite<quake::HOp>, GeneralRewrite<quake::XOp>,
                    GeneralRewrite<quake::YOp>, GeneralRewrite<quake::ZOp>,
                    GeneralRewrite<quake::SOp>, GeneralRewrite<quake::TOp>,
                    GeneralRewrite<quake::RxOp>, GeneralRewrite<quake::RyOp>,
                    GeneralRewrite<quake::RzOp>, GeneralRewrite<quake::R1Op>,
                    GeneralRewrite<quake::U3Op>, GeneralRewrite<quake::SwapOp>,
                    BorrowWireRewrite, ReturnWireRewrite>(quakeTypeConverter,
                                                          context);
    patterns.insert<MzRewrite>(quakeTypeConverter, resultCounter, context);
    const bool isAdaptiveProfile = convertTo == "qir-adaptive";
    patterns.insert<DiscriminateRewrite>(quakeTypeConverter, isAdaptiveProfile,
                                         regNameMap, context);
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                           func::FuncDialect, LLVM::LLVMDialect>();
    target.addIllegalDialect<quake::QuakeDialect>();
    target.addLegalOp<quake::WireSetOp>();

    LLVM_DEBUG(llvm::dbgs() << "Module before:\n"; op.dump());
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "Module after:\n"; op.dump());
  }
};

// Runs on the module. Prepare the module for conversion to QIR calls.
// We have to add the declarations of the QIR (QIS) functions and preprocess the
// names of the measurements, adding them to the Module as well as creating them
// when they are absent.
struct WireSetToProfileQIRPrepPass
    : public cudaq::opt::impl::WireSetToProfileQIRPrepBase<
          WireSetToProfileQIRPrepPass> {
  using WireSetToProfileQIRPrepBase::WireSetToProfileQIRPrepBase;

  void runOnOperation() override {
    ModuleOp op = getOperation();
    auto *ctx = &getContext();

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(op.getBody());
    auto loc = builder.getUnknownLoc();

    auto createNewDecl = [&](const std::string &name, FunctionType ty) {
      auto func = builder.create<func::FuncOp>(loc, name, ty);
      func.setPrivate();
    };
    auto addNewDecl = [&](std::string &&suffix, FunctionType ty) {
      createNewDecl(std::string(qis_prefix) + std::move(suffix), ty);
    };
    auto addBodyDecl = [&](std::string &&name, FunctionType ty) {
      addNewDecl(std::move(name) + std::string(qis_body_suffix), ty);
    };
    auto addCtlDecl = [&](std::string &&name, FunctionType ty) {
      addNewDecl(std::move(name) + std::string(qis_ctl_suffix), ty);
    };
    auto addDecls = [&](const char *name, FunctionType bodyTy,
                        FunctionType ctlTy) {
      addBodyDecl(name, bodyTy);
      addCtlDecl(name, ctlTy);
    };

    LLVM_DEBUG(llvm::dbgs() << "Module before prep:\n"; op.dump());
    // Insert declarations for all the functions we *may* be using.
    auto qbTy = cudaq::opt::getQubitType(ctx);
    auto targ1Ty = FunctionType::get(ctx, TypeRange{qbTy}, TypeRange{});
    auto targ1CtrlTy =
        FunctionType::get(ctx, TypeRange{qbTy, qbTy}, TypeRange{});
    addDecls("h", targ1Ty, targ1CtrlTy);
    addDecls("x", targ1Ty, targ1CtrlTy);
    addDecls("y", targ1Ty, targ1CtrlTy);
    addDecls("z", targ1Ty, targ1CtrlTy);
    addDecls("s", targ1Ty, targ1CtrlTy);
    addDecls("t", targ1Ty, targ1CtrlTy);
    addDecls("sdg", targ1Ty, targ1CtrlTy);
    addDecls("tdg", targ1Ty, targ1CtrlTy);

    auto f64Ty = builder.getF64Type();
    auto param1Targ1Ty =
        FunctionType::get(ctx, TypeRange{f64Ty, qbTy}, TypeRange{});
    auto param1Targ1CtrlTy =
        FunctionType::get(ctx, TypeRange{f64Ty, qbTy, qbTy}, TypeRange{});
    addDecls("rx", param1Targ1Ty, param1Targ1CtrlTy);
    addDecls("ry", param1Targ1Ty, param1Targ1CtrlTy);
    addDecls("rz", param1Targ1Ty, param1Targ1CtrlTy);
    addDecls("r1", param1Targ1Ty, param1Targ1CtrlTy);

    auto param3Targ1Ty = FunctionType::get(
        ctx, TypeRange{f64Ty, f64Ty, f64Ty, qbTy}, TypeRange{});
    auto param3Targ1CtrlTy = FunctionType::get(
        ctx, TypeRange{f64Ty, f64Ty, f64Ty, qbTy, qbTy}, TypeRange{});
    addDecls("u3", param3Targ1Ty, param3Targ1CtrlTy);

    auto targ2Ty = targ1CtrlTy;
    auto targ2CtrlTy =
        FunctionType::get(ctx, TypeRange{qbTy, qbTy, qbTy}, TypeRange{});
    addDecls("swap", targ2Ty, targ2CtrlTy);
    addBodyDecl("cnot", targ2Ty);

    auto resTy = cudaq::opt::getResultType(ctx);
    auto measTy = FunctionType::get(ctx, TypeRange{qbTy, resTy}, TypeRange{});
    addBodyDecl("mz", measTy);
    auto readResTy = FunctionType::get(ctx, TypeRange{resTy},
                                       TypeRange{builder.getI1Type()});
    createNewDecl("__quantum__qis__read_result__body", readResTy);

    auto i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());
    auto recordTy =
        FunctionType::get(ctx, TypeRange{resTy, i8PtrTy}, TypeRange{});
    createNewDecl("__quantum__rt__result_record_output", recordTy);

    unsigned counter = 0;
    op.walk([&](quake::MzOp meas) {
      auto optName = meas.getRegisterName();
      std::string name;
      if (optName) {
        name = *optName;
      } else {
        name = std::to_string(counter++);
        constexpr std::size_t padTo = 5;
        name = std::string(padTo - std::min(padTo, name.length()), '0') + name;
        meas.setRegisterName(name);
      }
      cudaq::IRBuilder irb(builder);
      irb.genCStringLiteralAppendNul(meas.getLoc(), op, name);
    });
    cudaq::IRBuilder irb(builder);
    irb.genCStringLiteralAppendNul(builder.getUnknownLoc(), op, "?");

    LLVM_DEBUG(llvm::dbgs() << "Module after prep:\n"; op->dump());
  }
};

struct WireSetToProfileQIRPostPass
    : public cudaq::opt::impl::WireSetToProfileQIRPostBase<
          WireSetToProfileQIRPostPass> {
  using WireSetToProfileQIRPostBase::WireSetToProfileQIRPostBase;

  void runOnOperation() override {
    ModuleOp op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    QuakeTypeConverter quakeTypeConverter;
    patterns.insert<WireSetRewrite>(quakeTypeConverter, ctx);
    ConversionTarget target(*ctx);
    target.addIllegalDialect<quake::QuakeDialect>();

    LLVM_DEBUG(llvm::dbgs() << "Module before:\n"; op.dump());
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "Module after:\n"; op.dump());
  }
};
} // namespace

void cudaq::opt::addWiresetToProfileQIRPipeline(OpPassManager &pm,
                                                StringRef profile) {
  pm.addPass(cudaq::opt::createWireSetToProfileQIRPrep());
  WireSetToProfileQIROptions wopt;
  if (!profile.empty())
    wopt.convertTo = profile.str();
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createWireSetToProfileQIR(wopt));
  pm.addPass(cudaq::opt::createWireSetToProfileQIRPost());
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
