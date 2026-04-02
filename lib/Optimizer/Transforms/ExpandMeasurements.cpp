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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
// Only an individual qubit measurement returns a bool.
template <typename A>
bool usesIndividualQubit(A x) {
  return x.getType() == quake::MeasureType::get(x.getContext());
}

// Pattern for expanding a multi-qubit measurement on unsized veq<?> targets
// into a dynamic loop of individual measurements.
template <typename A>
class ExpandUnsizedMeasurePattern : public OpRewritePattern<A> {
public:
  using OpRewritePattern<A>::OpRewritePattern;

  LogicalResult matchAndRewrite(A measureOp,
                                PatternRewriter &rewriter) const override {
    if (usesIndividualQubit(measureOp.getMeasOut()))
      return failure();

    // Only handle the unsized case here.
    bool hasUnsizedTarget = false;
    for (auto v : measureOp.getTargets())
      if (auto veqTy = dyn_cast<quake::VeqType>(v.getType()))
        if (!veqTy.hasSpecifiedSize())
          hasUnsizedTarget = true;
    if (!hasUnsizedTarget)
      return failure();

    // Only expand if every user of the measurement result is a DiscriminateOp.
    for (auto *user : measureOp.getMeasOut().getUsers())
      if (!isa<quake::DiscriminateOp>(user))
        return failure();

    // Even without discriminate users we must expand, because downstream QIR
    // lowering cannot handle mz on veq<?>. When discriminate users exist we
    // additionally allocate a buffer to collect per-qubit results and build the
    // stdvec that replaces each discriminate.
    bool hasDiscriminateUsers = !measureOp.getMeasOut().use_empty();
    auto loc = measureOp.getLoc();
    auto i64Ty = rewriter.getI64Type();
    auto measTy = quake::MeasureType::get(rewriter.getContext());

    // 1. Determine the total number of qubits we need to measure. This
    // determines the size of the buffer of bools to create to store the results
    // in.
    Value buff, totalToRead, buffOff, one;
    Type elemTy, bufElemTy;
    if (hasDiscriminateUsers) {
      auto firstDisc = cast<quake::DiscriminateOp>(
          *measureOp.getMeasOut().getUsers().begin());
      auto stdvecTy =
          cast<cudaq::cc::StdvecType>(firstDisc.getResult().getType());
      elemTy = stdvecTy.getElementType();
      unsigned elemWidth = cast<IntegerType>(elemTy).getWidth();
      bufElemTy =
          elemWidth > 8 ? elemTy : static_cast<Type>(rewriter.getI8Type());

      unsigned numQubits = 0u;
      for (auto v : measureOp.getTargets())
        if (v.getType().template isa<quake::RefType>())
          ++numQubits;
      totalToRead =
          rewriter.template create<arith::ConstantIntOp>(loc, numQubits, 64);
      for (auto v : measureOp.getTargets())
        if (v.getType().template isa<quake::VeqType>()) {
          Value vecSz =
              rewriter.template create<quake::VeqSizeOp>(loc, i64Ty, v);
          totalToRead =
              rewriter.template create<arith::AddIOp>(loc, totalToRead, vecSz);
        }

      // 2. Create the buffer.
      buff = rewriter.template create<cudaq::cc::AllocaOp>(loc, bufElemTy,
                                                           totalToRead);
      buffOff = rewriter.template create<arith::ConstantIntOp>(loc, 0, 64);
      one = rewriter.template create<arith::ConstantIntOp>(loc, 1, 64);
    }

    // 3. Measure each individual qubit and insert the result, in order, into
    // the buffer. For registers/vectors, loop over the entire set of qubits.
    for (auto v : measureOp.getTargets()) {
      if (isa<quake::RefType>(v.getType())) {
        auto meas = rewriter.template create<A>(loc, measTy, v);
        if (auto registerName = measureOp.getRegisterNameAttr())
          meas.setRegisterName(registerName);
        if (hasDiscriminateUsers) {
          auto bit = rewriter.template create<quake::DiscriminateOp>(
              loc, elemTy, meas.getMeasOut());
          Value addr = rewriter.template create<cudaq::cc::ComputePtrOp>(
              loc, cudaq::cc::PointerType::get(bufElemTy), buff, buffOff);
          Value stored = (elemTy != bufElemTy)
                             ? rewriter
                                   .template create<cudaq::cc::CastOp>(
                                       loc, bufElemTy, bit,
                                       cudaq::cc::CastOpMode::Unsigned)
                                   .getResult()
                             : static_cast<Value>(bit);
          rewriter.template create<cudaq::cc::StoreOp>(loc, stored, addr);
          buffOff = rewriter.template create<arith::AddIOp>(loc, buffOff, one);
        }
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
              if (auto registerName = measureOp.getRegisterNameAttr())
                meas.setRegisterName(registerName);
              if (hasDiscriminateUsers) {
                auto bit = builder.template create<quake::DiscriminateOp>(
                    loc, elemTy, meas.getMeasOut());
                Value offset =
                    builder.template create<arith::AddIOp>(loc, iv, buffOff);
                auto addr = builder.template create<cudaq::cc::ComputePtrOp>(
                    loc, cudaq::cc::PointerType::get(bufElemTy), buff, offset);
                Value stored = (elemTy != bufElemTy)
                                   ? builder
                                         .template create<cudaq::cc::CastOp>(
                                             loc, bufElemTy, bit,
                                             cudaq::cc::CastOpMode::Unsigned)
                                         .getResult()
                                   : static_cast<Value>(bit);
                builder.template create<cudaq::cc::StoreOp>(loc, stored, addr);
              }
            });
        if (hasDiscriminateUsers)
          buffOff =
              rewriter.template create<arith::AddIOp>(loc, buffOff, vecSz);
      }
    }

    // 4. Use the buffer as an initialization expression and create the
    // std::vec<bool> value.
    if (hasDiscriminateUsers) {
      auto stdvecTy = cudaq::cc::StdvecType::get(rewriter.getContext(), elemTy);
      SmallVector<quake::DiscriminateOp> discs;
      for (auto *out : measureOp.getMeasOut().getUsers())
        if (auto disc = dyn_cast_if_present<quake::DiscriminateOp>(out))
          discs.push_back(disc);
      for (auto disc : discs) {
        auto ptrArrTy =
            cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(elemTy));
        auto buffCast =
            rewriter.template create<cudaq::cc::CastOp>(loc, ptrArrTy, buff);
        rewriter.template replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
            disc, stdvecTy, buffCast, totalToRead);
      }
    }

    rewriter.eraseOp(measureOp);
    return success();
  }
};

using MxUnsizedRewrite = ExpandUnsizedMeasurePattern<quake::MxOp>;
using MyUnsizedRewrite = ExpandUnsizedMeasurePattern<quake::MyOp>;
using MzUnsizedRewrite = ExpandUnsizedMeasurePattern<quake::MzOp>;

// Generalized pattern for expanding a multiple qubit measurement (whether it is
// mx, my, or mz) to a series of individual measurements.
template <typename A>
class ExpandRewritePattern : public OpRewritePattern<A> {
public:
  using OpRewritePattern<A>::OpRewritePattern;

  LogicalResult matchAndRewrite(A measureOp,
                                PatternRewriter &rewriter) const override {
    if (usesIndividualQubit(measureOp.getMeasOut()))
      return failure();

    // Collect all the `get_measure` ops for this measurement operation.
    SmallVector<quake::GetMeasureOp> getMeasureOps;
    for (auto *user : measureOp.getMeasOut().getUsers())
      if (auto gm = dyn_cast<quake::GetMeasureOp>(user))
        getMeasureOps.push_back(gm);

    // Can only replace `get_measure %m[i]` with per-qubit measurements, else
    // bail out.
    if (getMeasureOps.empty() && !measureOp.getMeasOut().use_empty())
      return failure();

    // Validate that all `get_measure` ops have constant indices and all the veq
    // targets have known sizes.
    for (auto gm : getMeasureOps)
      if (!gm.hasConstantIndex())
        return failure();
    std::size_t totalMeasures = 0;
    for (auto v : measureOp.getTargets()) {
      if (isa<quake::RefType>(v.getType())) {
        ++totalMeasures;
      } else {
        auto veqTy = cast<quake::VeqType>(v.getType());
        if (!veqTy.hasSpecifiedSize())
          return failure();
        totalMeasures += veqTy.getSize();
      }
    }
    // Bounds check
    for (auto gm : getMeasureOps)
      if (gm.getConstantIndex() >= totalMeasures)
        return failure();

    auto loc = measureOp.getLoc();
    auto measTy = quake::MeasureType::get(rewriter.getContext());

    // Create individual per-qubit measurements for each target.
    SmallVector<Value> individualMeasures;
    for (auto v : measureOp.getTargets()) {
      if (isa<quake::RefType>(v.getType())) {
        auto meas = rewriter.template create<A>(loc, measTy, v);
        if (auto registerName = measureOp.getRegisterNameAttr())
          meas.setRegisterName(registerName);
        individualMeasures.push_back(meas.getMeasOut());
      } else {
        auto veqTy = cast<quake::VeqType>(v.getType());
        for (std::size_t i = 0; i < veqTy.getSize(); ++i) {
          Value idx =
              rewriter.template create<arith::ConstantIntOp>(loc, i, 64);
          Value qv = rewriter.template create<quake::ExtractRefOp>(loc, v, idx);
          auto meas = rewriter.template create<A>(loc, measTy, qv);
          if (auto registerName = measureOp.getRegisterNameAttr())
            meas.setRegisterName(registerName);
          individualMeasures.push_back(meas.getMeasOut());
        }
      }
    }

    // Replace each get_measure op with the corresponding individual result.
    for (auto gm : getMeasureOps)
      rewriter.replaceOp(gm, individualMeasures[gm.getConstantIndex()]);

    if (measureOp.getMeasOut().use_empty())
      rewriter.eraseOp(measureOp);

    return success();
  }
};

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
    auto veqArg = resetOp.getTargets();
    if (!isa<quake::VeqType>(veqArg.getType()))
      return failure();
    auto loc = resetOp.getLoc();
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

// Pattern for expanding a `quake.discriminate` op on a `quake.measurements`
// with a known size into a series of `quake.discriminate` ops on individual
// `quake.measure` results via `quake.get_measure`.
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
    if (!measTy.hasSpecifiedSize())
      return failure();

    auto loc = discOp.getLoc();
    auto stdvecResTy =
        cast<cudaq::cc::StdvecType>(discOp.getResult().getType());
    auto elemTy = stdvecResTy.getElementType();
    unsigned elemWidth = cast<IntegerType>(elemTy).getWidth();
    Type bufElemTy = elemWidth > 8 ? elemTy : rewriter.getI8Type();

    Value totalToRead =
        rewriter.create<arith::ConstantIntOp>(loc, measTy.getSize(), 64);
    Value buff =
        rewriter.create<cudaq::cc::AllocaOp>(loc, bufElemTy, totalToRead);

    // TODO: For large N, consider emitting a loop to avoid IR bloat.
    std::size_t n = measTy.getSize();
    for (std::size_t i = 0; i < n; ++i) {
      Value getMeas = rewriter.create<quake::GetMeasureOp>(loc, measVal, i);
      Value bit = rewriter.create<quake::DiscriminateOp>(loc, elemTy, getMeas);
      Value idx = rewriter.create<arith::ConstantIntOp>(loc, i, 64);
      Value addr = rewriter.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(bufElemTy), buff, idx);
      Value stored =
          (elemTy != bufElemTy)
              ? rewriter
                    .create<cudaq::cc::CastOp>(loc, bufElemTy, bit,
                                               cudaq::cc::CastOpMode::Unsigned)
                    .getResult()
              : bit;
      rewriter.create<cudaq::cc::StoreOp>(loc, stored, addr);
    }

    auto ptrArrElemTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(elemTy));
    auto buffCast = rewriter.create<cudaq::cc::CastOp>(loc, ptrArrElemTy, buff);
    rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(discOp, stdvecResTy,
                                                         buffCast, totalToRead);
    return success();
  }
};

class ExpandMeasurementsPass
    : public cudaq::opt::ExpandMeasurementsBase<ExpandMeasurementsPass> {
public:
  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();

    // Step 1: Expand discriminate(measurements<N>) into individual
    // get_measure + discriminate ops. This must run first so that step 2's
    // ExpandRewritePattern can see the resulting get_measure users.
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<ExpandDiscriminatePattern>(ctx);
      ConversionTarget target(*ctx);
      target.addLegalDialect<quake::QuakeDialect, cudaq::cc::CCDialect,
                             arith::ArithDialect, LLVM::LLVMDialect>();
      target.addDynamicallyLegalOp<quake::DiscriminateOp>(
          [](quake::DiscriminateOp d) {
            auto measTy =
                dyn_cast<quake::MeasurementsType>(d.getMeasurement().getType());
            if (!measTy)
              return true;
            return !measTy.hasSpecifiedSize();
          });
      if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
        op->emitOpError("could not expand discriminate ops");
        signalPassFailure();
        return;
      }
    }

    // Step 2: Expand multi-qubit m[xyz] and reset ops.
    // ExpandRewritePattern handles sized targets (veq<N>) via unrolling.
    // ExpandUnsizedMeasurePattern handles unsized targets (veq<?>) via
    // dynamic loops using VeqSizeOp + createInvariantLoop.
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<MxRewrite, MyRewrite, MzRewrite, ResetRewrite>(ctx);
      patterns.insert<MxUnsizedRewrite, MyUnsizedRewrite, MzUnsizedRewrite>(
          ctx);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
        op->emitOpError("could not expand measurements");
        signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createExpandMeasurementsPass() {
  return std::make_unique<ExpandMeasurementsPass>();
}
