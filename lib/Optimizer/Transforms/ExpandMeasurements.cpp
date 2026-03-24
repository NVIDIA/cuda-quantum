/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
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
      if (v.getType().template isa<quake::RefType>())
        ++numQubits;
    Value totalToRead =
        rewriter.template create<arith::ConstantIntOp>(loc, numQubits, 64);
    auto i64Ty = rewriter.getI64Type();
    for (auto v : measureOp.getTargets())
      if (v.getType().template isa<quake::VeqType>()) {
        Value vecSz = rewriter.template create<quake::VeqSizeOp>(loc, i64Ty, v);
        totalToRead =
            rewriter.template create<arith::AddIOp>(loc, totalToRead, vecSz);
      }

    // 2. Create the buffer.
    auto i1Ty = rewriter.getI1Type();
    auto i8Ty = rewriter.getI8Type();
    Value buff =
        rewriter.template create<cudaq::cc::AllocaOp>(loc, i8Ty, totalToRead);

    // 3. Measure each individual qubit and insert the result, in order, into
    // the buffer. For registers/vectors, loop over the entire set of qubits.
    Value buffOff = rewriter.template create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.template create<arith::ConstantIntOp>(loc, 1, 64);
    auto measTy = quake::MeasureType::get(rewriter.getContext());
    for (auto v : measureOp.getTargets()) {
      if (isa<quake::RefType>(v.getType())) {
        auto meas = rewriter.template create<A>(loc, measTy, v).getMeasOut();
        auto bit =
            rewriter.template create<quake::DiscriminateOp>(loc, i1Ty, meas);
        Value addr = rewriter.template create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(i8Ty), buff, buffOff);
        auto bitByte = rewriter.template create<cudaq::cc::CastOp>(
            loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
        rewriter.template create<cudaq::cc::StoreOp>(loc, bitByte, addr);
        buffOff = rewriter.template create<arith::AddIOp>(loc, buffOff, one);
      } else {
        assert(isa<quake::VeqType>(v.getType()));
        Value vecSz = rewriter.template create<quake::VeqSizeOp>(loc, i64Ty, v);
        cudaq::opt::factory::createInvariantLoop(
            rewriter, loc, vecSz,
            [&](OpBuilder &builder, Location loc, Region &, Block &block) {
              Value iv = block.getArgument(0);
              Value qv =
                  builder.template create<quake::ExtractRefOp>(loc, v, iv);
              auto meas = builder.template create<A>(loc, measTy, qv);
              auto bit = builder.template create<quake::DiscriminateOp>(
                  loc, i1Ty, meas.getMeasOut());
              if (auto registerName = measureOp.getRegisterNameAttr())
                meas.setRegisterName(registerName);
              Value offset =
                  builder.template create<arith::AddIOp>(loc, iv, buffOff);
              auto addr = builder.template create<cudaq::cc::ComputePtrOp>(
                  loc, cudaq::cc::PointerType::get(i8Ty), buff, offset);
              auto bitByte = rewriter.template create<cudaq::cc::CastOp>(
                  loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
              builder.template create<cudaq::cc::StoreOp>(loc, bitByte, addr);
            });
        buffOff = rewriter.template create<arith::AddIOp>(loc, buffOff, vecSz);
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
            rewriter.template create<cudaq::cc::CastOp>(loc, ptrArrI1Ty, buff);
        rewriter.template replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
            disc, stdvecTy, buffCast, totalToRead);
      }

    rewriter.eraseOp(measureOp);
    return success();
  }
};

// Expand a `quake.discriminate` on a `!quake.measurements<N>` value into
// individual `get_measure` + `discriminate` operations, producing a
// `!cc.stdvec<i1>`.
class ExpandDiscriminatePattern
    : public OpRewritePattern<quake::DiscriminateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::DiscriminateOp discOp,
                                PatternRewriter &rewriter) const override {
    auto measVal = discOp.getMeasurement();
    auto measTy = dyn_cast<quake::MeasurementsType>(measVal.getType());
    if (!measTy)
      return failure();

    auto loc = discOp.getLoc();
    auto i1Ty = rewriter.getI1Type();
    auto i8Ty = rewriter.getI8Type();
    Value totalToRead;

    if (measTy.hasSpecifiedSize()) {
      totalToRead =
          rewriter.create<arith::ConstantIntOp>(loc, measTy.getSize(), 64);
    } else {
      TODO_loc(loc, "unsized measurements in `ExpandDiscriminatePattern`");
    }

    Value buff = rewriter.create<cudaq::cc::AllocaOp>(loc, i8Ty, totalToRead);

    if (measTy.hasSpecifiedSize()) {
      std::size_t n = measTy.getSize();
      for (std::size_t i = 0; i < n; ++i) {
        Value getMeas = rewriter.create<quake::GetMeasureOp>(loc, measVal, i);
        Value bit = rewriter.create<quake::DiscriminateOp>(loc, i1Ty, getMeas);
        Value idx = rewriter.create<arith::ConstantIntOp>(loc, i, 64);
        Value addr = rewriter.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(i8Ty), buff, idx);
        auto bitByte = rewriter.create<cudaq::cc::CastOp>(
            loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
        rewriter.create<cudaq::cc::StoreOp>(loc, bitByte, addr);
      }
    }

    auto stdvecTy = cudaq::cc::StdvecType::get(rewriter.getContext(), i1Ty);
    auto ptrArrI1Ty =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i1Ty));
    auto buffCast = rewriter.create<cudaq::cc::CastOp>(loc, ptrArrI1Ty, buff);
    rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(discOp, stdvecTy,
                                                         buffCast, totalToRead);
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
    Value vecSz = rewriter.create<quake::VeqSizeOp>(loc, i64Ty, veqArg);
    cudaq::opt::factory::createInvariantLoop(
        rewriter, loc, vecSz,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value iv = block.getArgument(0);
          Value qv = builder.create<quake::ExtractRefOp>(loc, veqArg, iv);
          builder.create<quake::ResetOp>(loc, TypeRange{}, qv);
        });
    rewriter.eraseOp(resetOp);
    return success();
  }
};

class ExpandMeasurementsPass
    : public cudaq::opt::ExpandMeasurementsBase<ExpandMeasurementsPass> {
public:
  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<MxRewrite, MyRewrite, MzRewrite, ResetRewrite,
                    ExpandDiscriminatePattern>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect, cudaq::cc::CCDialect,
                           arith::ArithDialect, LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<quake::MxOp>(
        [](quake::MxOp x) { return usesIndividualQubit(x.getMeasOut()); });
    target.addDynamicallyLegalOp<quake::MyOp>(
        [](quake::MyOp x) { return usesIndividualQubit(x.getMeasOut()); });
    target.addDynamicallyLegalOp<quake::MzOp>(
        [](quake::MzOp x) { return usesIndividualQubit(x.getMeasOut()); });
    // Convert only on the `MeasurementsType` operand, otherwise legal.
    target.addDynamicallyLegalOp<quake::DiscriminateOp>(
        [](quake::DiscriminateOp d) {
          return !isa<quake::MeasurementsType>(d.getMeasurement().getType());
        });
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
