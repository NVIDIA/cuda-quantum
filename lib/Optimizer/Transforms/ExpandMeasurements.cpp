/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_EXPANDMEASUREMENTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

// Only an individual qubit measurement returns a scalar token. Both
// `!quake.measure` (legacy `bool`/`Result*` token) and `!cc.measure_handle`
// (the IR alias of `cudaq::measure_handle`, an `i64` payload) are scalar
// per-qubit measurement results, so neither requires expansion to a register.
template <typename A>
bool usesIndividualQubit(A x) {
  return isa<cudaq::quake::MeasureType, cudaq::cc::MeasureHandleType>(
      x.getType());
}

// Generalized pattern for expanding a multiple qubit measurement (whether it is
// mx, my, or mz) to a series of individual measurements.
//
// Handles both result-type families that the vector form of `quake.mz`/`mx`/
// `my` can carry:
//   - `!cc.stdvec<!quake.measure>` -- the legacy form. The only legitimate
//     consumer is `quake.discriminate`, so the rewrite folds the per-element
//     measurements straight into a `cc.stdvec_init -> !cc.stdvec<i1>`.
//   - `!cc.stdvec<!cc.measure_handle>` -- the handle-vector value can have
//     non-discriminate consumers. Those consumers expect a value of the
//     original handle-stdvec type, so the rewrite additionally builds a
//     per-element handle buffer and folds it into a `cc.stdvec_init ->
//     !cc.stdvec<!cc.measure_handle>` that replaces all remaining uses.
template <typename A>
class ExpandRewritePattern : public OpRewritePattern<A> {
public:
  using OpRewritePattern<A>::OpRewritePattern;

  LogicalResult matchAndRewrite(A measureOp,
                                PatternRewriter &rewriter) const override {
    auto loc = measureOp.getLoc();
    auto *ctx = rewriter.getContext();

    // The dynamic-legality predicate filters out the scalar forms, so by
    // construction the result type here is `!cc.stdvec<X>` for some X.
    auto stdvecResTy =
        dyn_cast<cudaq::cc::StdvecType>(measureOp.getMeasOut().getType());
    auto handleTy = cudaq::cc::MeasureHandleType::get(ctx);
    bool isHandleResult =
        isa<cudaq::cc::MeasureHandleType>(stdvecResTy.getElementType());

    // Per-element scalar result type tracks the original stdvec element
    // type. For handle inputs we measure into `!cc.measure_handle` per
    // qubit.
    Type perElemTy =
        isHandleResult ? static_cast<Type>(handleTy)
                       : static_cast<Type>(cudaq::quake::MeasureType::get(ctx));

    // Classify users so we only allocate the buffers we actually need, and
    // collect the discriminate users at the same time. The legacy
    // `!quake.measure` path has only `quake.discriminate` consumers by
    // construction; the handle path may have either, both, or none.
    SmallVector<cudaq::quake::DiscriminateOp> discUsers;
    bool hasNonDiscUser = false;
    for (auto *u : measureOp.getMeasOut().getUsers()) {
      if (auto d = dyn_cast<cudaq::quake::DiscriminateOp>(u))
        discUsers.push_back(d);
      else
        hasNonDiscUser = true;
    }
    // Allocation policy:
    //   - Legacy `!cc.stdvec<!quake.measure>` always allocates the i1 buffer.
    //   - `!cc.stdvec<!cc.measure_handle>` allocates each buffer only when a
    //   consumer in that element-type class is present.
    bool needI1Buf = !isHandleResult || !discUsers.empty();
    bool needHandleBuf = isHandleResult && hasNonDiscUser;

    // 1. Determine the total number of qubits we need to measure. This
    // determines the size of the buffer of bools to create to store the results
    // in.
    unsigned numQubits = 0u;
    for (auto v : measureOp.getTargets())
      if (isa<cudaq::quake::RefType>(v.getType()))
        ++numQubits;
    Value totalToRead =
        arith::ConstantIntOp::create(rewriter, loc, numQubits, 64);
    auto i64Ty = rewriter.getI64Type();
    for (auto v : measureOp.getTargets())
      if (isa<cudaq::quake::VeqType>(v.getType())) {
        Value vecSz = cudaq::quake::VeqSizeOp::create(rewriter, loc, i64Ty, v);
        totalToRead = arith::AddIOp::create(rewriter, loc, totalToRead, vecSz);
      }

    // 2. Create the buffers (one per output kind we actually need).
    auto i1Ty = rewriter.getI1Type();
    auto i8Ty = rewriter.getI8Type();
    Value i1Buff;
    if (needI1Buf)
      i1Buff = cudaq::cc::AllocaOp::create(rewriter, loc, i8Ty, totalToRead);
    Value handleBuff;
    if (needHandleBuf)
      handleBuff =
          cudaq::cc::AllocaOp::create(rewriter, loc, handleTy, totalToRead);

    // Per-element store helper. Each qubit is measured exactly once with
    // `perElemTy`; the resulting value is fanned out to whichever buffers we
    // allocated (i1 for discriminate consumers, handle for non-discriminate
    // consumers).
    auto storePerElement = [&](OpBuilder &builder, Location loc, Value meas,
                               Value offset) {
      if (needI1Buf) {
        auto bit =
            cudaq::quake::DiscriminateOp::create(builder, loc, i1Ty, meas);
        auto addr = cudaq::cc::ComputePtrOp::create(
            builder, loc, cudaq::cc::PointerType::get(i8Ty), i1Buff, offset);
        auto bitByte = cudaq::cc::CastOp::create(
            builder, loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
        cudaq::cc::StoreOp::create(builder, loc, bitByte, addr);
      }
      if (needHandleBuf) {
        auto addr = cudaq::cc::ComputePtrOp::create(
            builder, loc, cudaq::cc::PointerType::get(handleTy), handleBuff,
            offset);
        cudaq::cc::StoreOp::create(builder, loc, meas, addr);
      }
    };

    // 3. Measure each individual qubit and insert the result, in order, into
    // the buffer. For registers, loop over the entire set of qubits.
    Value buffOff = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    for (auto v : measureOp.getTargets()) {
      if (isa<cudaq::quake::RefType>(v.getType())) {
        auto meas = A::create(rewriter, loc, perElemTy, v).getMeasOut();
        storePerElement(rewriter, loc, meas, buffOff);
        buffOff = arith::AddIOp::create(rewriter, loc, buffOff, one);
      } else {
        assert(isa<cudaq::quake::VeqType>(v.getType()));
        Value vecSz = cudaq::quake::VeqSizeOp::create(rewriter, loc, i64Ty, v);
        cudaq::opt::factory::createInvariantLoop(
            rewriter, loc, vecSz,
            [&](OpBuilder &builder, Location loc, Region &, Block &block) {
              Value iv = block.getArgument(0);
              Value qv =
                  cudaq::quake::ExtractRefOp::create(builder, loc, v, iv);
              auto meas = A::create(builder, loc, perElemTy, qv);
              if (auto registerName = measureOp.getRegisterNameAttr())
                meas.setRegisterName(registerName);
              Value offset = arith::AddIOp::create(builder, loc, iv, buffOff);
              storePerElement(builder, loc, meas.getMeasOut(), offset);
            });
        buffOff = arith::AddIOp::create(rewriter, loc, buffOff, vecSz);
      }
    }

    // 4. Replace each `quake.discriminate` consumer with a
    // `cc.stdvec_init -> !cc.stdvec<i1>` over the i1 buffer.
    if (needI1Buf) {
      auto stdvecI1Ty = cudaq::cc::StdvecType::get(ctx, i1Ty);
      auto ptrArrI1Ty =
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i1Ty));
      for (auto disc : discUsers) {
        auto buffCast =
            cudaq::cc::CastOp::create(rewriter, loc, ptrArrI1Ty, i1Buff);
        rewriter.template replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
            disc, stdvecI1Ty, buffCast, totalToRead);
      }
    }

    // 5. For the handle path with non-discriminate consumers, build a
    // `cc.stdvec_init -> !cc.stdvec<!cc.measure_handle>` over the handle
    // buffer and route the original result's remaining users to it via
    // `replaceOp` (one atomic substitution)
    Value replacementVal;
    if (needHandleBuf) {
      auto stdvecHandleTy = cudaq::cc::StdvecType::get(ctx, handleTy);
      auto handleStdvec = cudaq::cc::StdvecInitOp::create(
          rewriter, loc, stdvecHandleTy, handleBuff, totalToRead);
      replacementVal = handleStdvec.getResult();
    }

    // The pass is scheduled before wire lowering, so the variadic `$wires`
    // result group is structurally empty here.
    assert(measureOp.getWires().empty() &&
           "`expand-measurements` runs before wire lowering");

    // Step 5 builds a handle-vector replacement exactly when the
    // user-classification scan found a non-discriminate consumer. Without
    // this, `replaceOp` below would feed a null value through to a live
    // user.
    assert((replacementVal != nullptr) == hasNonDiscUser &&
           "handle-vector replacement must exist iff a non-discriminate "
           "consumer was present");

    rewriter.replaceOp(measureOp, replacementVal);
    return success();
  }
};

namespace {
using MxRewrite = ExpandRewritePattern<cudaq::quake::MxOp>;
using MyRewrite = ExpandRewritePattern<cudaq::quake::MyOp>;
using MzRewrite = ExpandRewritePattern<cudaq::quake::MzOp>;

// Expand `quake.discriminate : !cc.stdvec<!cc.measure_handle> ->
// !cc.stdvec<i1>` when the input handle vector is *not* the direct result
// of a measurement op. The bridge emits this shape for `cudaq::to_bools`
// applied to a handle vector that has crossed an SSA boundary
// (e.g. function argument, kernel return), where the measurement-op
// pattern above cannot reach the underlying `quake.mz/mx/my`. It loops
// over the handle vector, discriminates each element, and rewraps the
// resulting bytes as a `!cc.stdvec<i1>`. The direct-from-measurement
// case stays handled by `ExpandRewritePattern` to avoid an extra
// per-element load.
class ExpandStdvecHandleDiscriminate
    : public OpRewritePattern<cudaq::quake::DiscriminateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::DiscriminateOp disc,
                                PatternRewriter &rewriter) const override {
    Value handleVec = disc.getMeasurement();
    auto stdvecTy = dyn_cast<cudaq::cc::StdvecType>(handleVec.getType());
    if (!stdvecTy ||
        !isa<cudaq::cc::MeasureHandleType>(stdvecTy.getElementType()))
      return failure();
    if (handleVec.getDefiningOp<cudaq::quake::MeasurementInterface>())
      return failure();

    auto loc = disc.getLoc();
    auto *ctx = rewriter.getContext();
    auto i1Ty = rewriter.getI1Type();
    auto i8Ty = rewriter.getI8Type();
    auto i64Ty = rewriter.getI64Type();
    auto handleTy = cudaq::cc::MeasureHandleType::get(ctx);

    Value vecSize =
        cudaq::cc::StdvecSizeOp::create(rewriter, loc, i64Ty, handleVec);
    auto handleArrPtrTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(handleTy));
    Value handleData = cudaq::cc::StdvecDataOp::create(
        rewriter, loc, handleArrPtrTy, handleVec);
    // Output is held in an i8 buffer, then bitcast to `!cc.ptr<!cc.array
    // <i1 x ?>>` for the wrap. This matches the convention used by the
    // measurement-op pattern above (steps 2 + 4) so downstream passes see
    // the same shape regardless of which path produced the i1 vector.
    Value i1Buff = cudaq::cc::AllocaOp::create(rewriter, loc, i8Ty, vecSize);

    cudaq::opt::factory::createInvariantLoop(
        rewriter, loc, vecSize,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value iv = block.getArgument(0);
          Value handleAddr = cudaq::cc::ComputePtrOp::create(
              builder, loc, cudaq::cc::PointerType::get(handleTy), handleData,
              iv);
          Value handleVal = cudaq::cc::LoadOp::create(builder, loc, handleAddr);
          Value bit = cudaq::quake::DiscriminateOp::create(builder, loc, i1Ty,
                                                           handleVal);
          Value byteAddr = cudaq::cc::ComputePtrOp::create(
              builder, loc, cudaq::cc::PointerType::get(i8Ty), i1Buff, iv);
          Value bitByte = cudaq::cc::CastOp::create(
              builder, loc, i8Ty, bit, cudaq::cc::CastOpMode::Unsigned);
          cudaq::cc::StoreOp::create(builder, loc, bitByte, byteAddr);
        });

    auto stdvecI1Ty = cudaq::cc::StdvecType::get(ctx, i1Ty);
    auto ptrArrI1Ty =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i1Ty));
    Value buffCast =
        cudaq::cc::CastOp::create(rewriter, loc, ptrArrI1Ty, i1Buff);
    rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(disc, stdvecI1Ty,
                                                         buffCast, vecSize);
    return success();
  }
};

/// Convert a `quake.reset` with a `veq` argument into a loop over the elements
/// of the `veq` and `quake.reset` on each of them.
class ResetRewrite : public OpRewritePattern<cudaq::quake::ResetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ResetOp resetOp,
                                PatternRewriter &rewriter) const override {
    auto loc = resetOp.getLoc();
    auto veqArg = resetOp.getTargets();
    auto i64Ty = rewriter.getI64Type();
    Value vecSz = cudaq::quake::VeqSizeOp::create(rewriter, loc, i64Ty, veqArg);
    cudaq::opt::factory::createInvariantLoop(
        rewriter, loc, vecSz,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value iv = block.getArgument(0);
          Value qv =
              cudaq::quake::ExtractRefOp::create(builder, loc, veqArg, iv);
          cudaq::quake::ResetOp::create(builder, loc, TypeRange{}, qv);
        });
    rewriter.eraseOp(resetOp);
    return success();
  }
};

class ExpandMeasurementsPass
    : public cudaq::opt::impl::ExpandMeasurementsBase<ExpandMeasurementsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<MxRewrite, MyRewrite, MzRewrite, ResetRewrite,
                    ExpandStdvecHandleDiscriminate>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<cudaq::quake::QuakeDialect, cudaq::cc::CCDialect,
                           arith::ArithDialect, LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<cudaq::quake::MxOp>([](cudaq::quake::MxOp x) {
      return usesIndividualQubit(x.getMeasOut());
    });
    target.addDynamicallyLegalOp<cudaq::quake::MyOp>([](cudaq::quake::MyOp x) {
      return usesIndividualQubit(x.getMeasOut());
    });
    target.addDynamicallyLegalOp<cudaq::quake::MzOp>([](cudaq::quake::MzOp x) {
      return usesIndividualQubit(x.getMeasOut());
    });
    target.addDynamicallyLegalOp<cudaq::quake::ResetOp>(
        [](cudaq::quake::ResetOp r) {
          return !isa<cudaq::quake::VeqType>(r.getTargets().getType());
        });
    target.addDynamicallyLegalOp<cudaq::quake::DiscriminateOp>(
        [](cudaq::quake::DiscriminateOp d) {
          // Scalar discriminate is always legal.
          auto stdvecTy =
              dyn_cast<cudaq::cc::StdvecType>(d.getMeasurement().getType());
          if (!stdvecTy)
            return true;
          // Vector discriminate of legacy `!quake.measure` is folded as
          // a side-effect of the measurement-op rewrite (step 4); leave
          // it legal here so the driver does not look for a standalone
          // pattern.
          if (!isa<cudaq::cc::MeasureHandleType>(stdvecTy.getElementType()))
            return true;
          // Vector discriminate of `!cc.measure_handle` whose source is
          // a measurement op is similarly folded (step 4 again). Only
          // the indirect case needs `ExpandStdvecHandleDiscriminate`.
          return d.getMeasurement()
                     .getDefiningOp<cudaq::quake::MeasurementInterface>() !=
                 nullptr;
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
