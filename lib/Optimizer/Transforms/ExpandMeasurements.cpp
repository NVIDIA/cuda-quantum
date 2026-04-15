/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_EXPANDMEASUREMENTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

// Only an individual qubit measurement returns a bool.
template <typename A>
bool usesIndividualQubit(A x) {
  return x.getType() == quake::MeasureType::get(x.getContext());
}

// Generalized pattern for expanding a multiple qubit measurement (whether it is
// mx, my, or mz) to a series of individual measurements.
template <typename A>
class ExpandRewritePattern : public OpRewritePattern<A> {
public:
  using OpRewritePattern<A>::OpRewritePattern;

  LogicalResult matchAndRewrite(A measureOp,
                                PatternRewriter &rewriter) const override {
    auto loc = measureOp.getLoc();
    // 1. Determine the total number of qubits we need to measure. This
    // determines the size of the buffer of bools to create to store the results
    // in.
    unsigned numQubits = 0u;
    for (auto v : measureOp.getTargets())
      if (isa<quake::RefType>(v.getType()))
        ++numQubits;
    Value totalToRead = arith::ConstantIntOp::create(
        rewriter, loc, rewriter.getIntegerType(64), numQubits);
    auto i64Ty = rewriter.getI64Type();
    for (auto v : measureOp.getTargets())
      if (isa<quake::VeqType>(v.getType())) {
        Value vecSz = quake::VeqSizeOp::create(rewriter, loc, i64Ty, v);
        totalToRead = arith::AddIOp::create(rewriter, loc, totalToRead, vecSz);
      }

    // 2. Create the buffer.
    auto i1Ty = rewriter.getI1Type();
    auto i8Ty = rewriter.getI8Type();
    Value buff = cudaq::cc::AllocaOp::create(rewriter, loc, i8Ty, totalToRead);

    // 3. Measure each individual qubit and insert the result, in order, into
    // the buffer. For registers/vectors, loop over the entire set of qubits.
    Value buffOff = arith::ConstantIntOp::create(
        rewriter, loc, rewriter.getIntegerType(64), 0);
    Value one = arith::ConstantIntOp::create(rewriter, loc,
                                             rewriter.getIntegerType(64), 1);
    auto measTy = quake::MeasureType::get(rewriter.getContext());
    for (auto v : measureOp.getTargets()) {
      if (isa<quake::RefType>(v.getType())) {
        auto meas = A::create(rewriter, loc, measTy, v).getMeasOut();
        auto bit = quake::DiscriminateOp::create(rewriter, loc, i1Ty, meas);
        Value addr = cudaq::cc::ComputePtrOp::create(
            rewriter, loc, cudaq::cc::PointerType::get(i8Ty), buff, buffOff);
        auto bitByte = cudaq::cc::CastOp::create(
            rewriter, loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
        cudaq::cc::StoreOp::create(rewriter, loc, bitByte, addr);
        buffOff = arith::AddIOp::create(rewriter, loc, buffOff, one);
      } else {
        assert(isa<quake::VeqType>(v.getType()));
        Value vecSz = quake::VeqSizeOp::create(rewriter, loc, i64Ty, v);
        cudaq::opt::factory::createInvariantLoop(
            rewriter, loc, vecSz,
            [&](OpBuilder &builder, Location loc, Region &, Block &block) {
              Value iv = block.getArgument(0);
              Value qv = quake::ExtractRefOp::create(builder, loc, v, iv);
              auto meas = A::create(builder, loc, measTy, qv);
              auto bit = quake::DiscriminateOp::create(builder, loc, i1Ty,
                                                       meas.getMeasOut());
              if (auto registerName = measureOp.getRegisterNameAttr())
                meas.setRegisterName(registerName);
              Value offset = arith::AddIOp::create(builder, loc, iv, buffOff);
              auto addr = cudaq::cc::ComputePtrOp::create(
                  builder, loc, cudaq::cc::PointerType::get(i8Ty), buff,
                  offset);
              auto bitByte = cudaq::cc::CastOp::create(
                  rewriter, loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
              cudaq::cc::StoreOp::create(builder, loc, bitByte, addr);
            });
        buffOff = arith::AddIOp::create(rewriter, loc, buffOff, vecSz);
      }
    }

    // 4. Use the buffer as an initialization expression and create the
    // std::vec<bool> value.
    auto stdvecTy = cudaq::cc::StdvecType::get(rewriter.getContext(), i1Ty);
    for (auto *out : measureOp.getMeasOut().getUsers())
      if (auto disc = dyn_cast_if_present<quake::DiscriminateOp>(out)) {
        auto ptrArrI1Ty =
            cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i1Ty));
        auto buffCast =
            cudaq::cc::CastOp::create(rewriter, loc, ptrArrI1Ty, buff);
        rewriter.template replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
            disc, stdvecTy, buffCast, totalToRead);
      }

    rewriter.eraseOp(measureOp);
    return success();
  }
};

namespace {
using MxRewrite = ExpandRewritePattern<quake::MxOp>;
using MyRewrite = ExpandRewritePattern<quake::MyOp>;
using MzRewrite = ExpandRewritePattern<quake::MzOp>;

/// Convert a `quake.reset` with a `veq` argument into a loop over the elements
/// of the `veq` and `quake.reset` on each of them.
class ResetRewrite : public OpRewritePattern<quake::ResetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ResetOp resetOp,
                                PatternRewriter &rewriter) const override {
    auto loc = resetOp.getLoc();
    auto veqArg = resetOp.getTargets();
    auto i64Ty = rewriter.getI64Type();
    Value vecSz = quake::VeqSizeOp::create(rewriter, loc, i64Ty, veqArg);
    cudaq::opt::factory::createInvariantLoop(
        rewriter, loc, vecSz,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value iv = block.getArgument(0);
          Value qv = quake::ExtractRefOp::create(builder, loc, veqArg, iv);
          quake::ResetOp::create(builder, loc, TypeRange{}, qv);
        });
    rewriter.eraseOp(resetOp);
    return success();
  }
};

class ExpandMeasurementsPass
    : public cudaq::opt::impl::ExpandMeasurementsBase<ExpandMeasurementsPass> {
public:
  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<MxRewrite, MyRewrite, MzRewrite, ResetRewrite>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect, cudaq::cc::CCDialect,
                           arith::ArithDialect, LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<quake::MxOp>(
        [](quake::MxOp x) { return usesIndividualQubit(x.getMeasOut()); });
    target.addDynamicallyLegalOp<quake::MyOp>(
        [](quake::MyOp x) { return usesIndividualQubit(x.getMeasOut()); });
    target.addDynamicallyLegalOp<quake::MzOp>(
        [](quake::MzOp x) { return usesIndividualQubit(x.getMeasOut()); });
    target.addDynamicallyLegalOp<quake::ResetOp>([](quake::ResetOp r) {
      return !isa<quake::VeqType>(r.getTargets().getType());
    });
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      op->emitOpError("could not expand measurements");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createExpandMeasurementsPass() {
  return std::make_unique<ExpandMeasurementsPass>();
}
