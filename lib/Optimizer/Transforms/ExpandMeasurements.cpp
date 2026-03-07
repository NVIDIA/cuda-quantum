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

// After expansion, only an individual qubit measurement is valid.
// This predicate is used by the conversion target to determine legality.
template <typename A>
bool usesIndividualQubit(A x) {
  return x.getType() == quake::MeasureType::get(x.getContext());
}

// Generalized pattern for expanding a multiple qubit measurement (whether it is
// mx, my, or mz) to a series of individual measurements.
// Two code paths exist based on whether discrimination is immediate or
// deferred:
//
// 1. With discriminate: When user code converts measurement to bool (e.g.,
//    `if (result)` or `bool b = result`), discriminate is emitted. We produce
//    `!cc.stdvec<i1>`.
//
// 2. Without discriminate: When measurement results are used as opaque
//    `measure_result` values, no discriminate exists. We produce
//    `!cc.stdvec<!quake.measure>`
template <typename A>
class ExpandRewritePattern : public OpRewritePattern<A> {
public:
  using OpRewritePattern<A>::OpRewritePattern;

  LogicalResult matchAndRewrite(A measureOp,
                                PatternRewriter &rewriter) const override {
    auto loc = measureOp.getLoc();
    auto ctx = rewriter.getContext();

    // Step 1: Determine the total number of qubits we need to measure. This
    // determines the size of the buffer to create to store the results in.
    Value totalQubits = calculateTotalQubits(rewriter, loc, measureOp);

    // Step 2: Determine processing mode and find discriminate op, if present
    quake::DiscriminateOp discriminateOp = findDiscriminateUser(measureOp);
    bool needsDiscrimination = (discriminateOp != nullptr);

    // Step 3: Allocate storage buffer
    Type storageType;
    if (needsDiscrimination)
      storageType = rewriter.getI8Type();
    else
      storageType = quake::MeasureType::get(ctx);
    Value buffer = rewriter.template create<cudaq::cc::AllocaOp>(
        loc, storageType, totalQubits);

    // Step 4: Process all measurement targets
    Value offset = rewriter.template create<arith::ConstantIntOp>(loc, 0, 64);
    Value increment =
        rewriter.template create<arith::ConstantIntOp>(loc, 1, 64);

    // Each target is processed sequentially, with offset tracking
    // to ensure each measurement result lands in its correct buffer position.
    measureAllTargets(rewriter, loc, measureOp, buffer, offset, increment,
                      needsDiscrimination);

    // Step 5: Package results into output format. The buffer becomes a stdvec,
    // replacing either the original measurement op (non-discriminated) or the
    // discriminate op (discriminated path).
    finalizeResults(rewriter, loc, measureOp, buffer, totalQubits,
                    needsDiscrimination, discriminateOp);

    return success();
  }

private:
  // Calculate how many qubits need to be measured
  Value calculateTotalQubits(PatternRewriter &rewriter, Location loc,
                             A measureOp) const {
    auto i64Ty = rewriter.getI64Type();
    unsigned refCount = 0u;
    // Count individual qubit references
    for (auto target : measureOp.getTargets())
      if (target.getType().template isa<quake::RefType>())
        ++refCount;
    Value total =
        rewriter.template create<arith::ConstantIntOp>(loc, refCount, 64);
    // Add sizes of qubit vectors
    for (auto target : measureOp.getTargets())
      if (target.getType().template isa<quake::VeqType>()) {
        Value size =
            rewriter.template create<quake::VeqSizeOp>(loc, i64Ty, target);
        total = rewriter.template create<arith::AddIOp>(loc, total, size);
      }
    return total;
  }

  // Find discriminate op if it exists for this measurement
  quake::DiscriminateOp findDiscriminateUser(A measureOp) const {
    for (auto *user : measureOp.getMeasOut().getUsers())
      if (auto discOp = dyn_cast<quake::DiscriminateOp>(user))
        return discOp;
    return nullptr;
  }

  // Stores a single measurement result into the buffer. The discrimination
  // mode determines both the storage format and whether we insert a
  // discriminate op here (eager) or preserve the raw measurement (lazy).
  void recordMeasurement(OpBuilder &builder, Location loc, Value measResult,
                         Value buffer, Value position,
                         bool hasDiscriminate) const {
    auto ctx = builder.getContext();

    if (hasDiscriminate) {
      // Convert to bit, then to byte for storage
      auto i1Ty = builder.getI1Type();
      auto i8Ty = builder.getI8Type();
      auto bitResult =
          builder.template create<quake::DiscriminateOp>(loc, i1Ty, measResult);
      Value storageAddr = builder.template create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(i8Ty), buffer, position);
      auto byteValue = builder.template create<cudaq::cc::CastOp>(
          loc, i8Ty, bitResult, cudaq::cc::CastOpMode::Unsigned);
      builder.template create<cudaq::cc::StoreOp>(loc, byteValue, storageAddr);
    } else {
      // Store measurement directly
      auto measType = quake::MeasureType::get(ctx);
      Value storageAddr = builder.template create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(measType), buffer, position);
      builder.template create<cudaq::cc::StoreOp>(loc, measResult, storageAddr);
    }
  }

  // Process a single qubit reference
  void measureSingleQubit(PatternRewriter &rewriter, Location loc, Value qubit,
                          Value &currentOffset, Value buffer, Value step,
                          A measureOp, bool discriminate) const {
    auto measType = quake::MeasureType::get(rewriter.getContext());

    auto measurement = rewriter.template create<A>(loc, measType, qubit);
    // Preserve register name for output recording in QIR
    if (auto regName = measureOp.getRegisterNameAttr())
      measurement.setRegisterName(regName);

    recordMeasurement(rewriter, loc, measurement.getMeasOut(), buffer,
                      currentOffset, discriminate);
    currentOffset =
        rewriter.template create<arith::AddIOp>(loc, currentOffset, step);
  }

  // Process a vector of qubits
  void measureQubitVector(PatternRewriter &rewriter, Location loc, Value vector,
                          Value &currentOffset, Value buffer, A measureOp,
                          bool discriminate) const {
    auto i64Ty = rewriter.getI64Type();
    auto measType = quake::MeasureType::get(rewriter.getContext());

    Value vectorSize =
        rewriter.template create<quake::VeqSizeOp>(loc, i64Ty, vector);

    cudaq::opt::factory::createInvariantLoop(
        rewriter, loc, vectorSize,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value index = block.getArgument(0);
          Value qubit =
              builder.template create<quake::ExtractRefOp>(loc, vector, index);
          auto measurement = builder.template create<A>(loc, measType, qubit);
          if (auto regName = measureOp.getRegisterNameAttr())
            measurement.setRegisterName(regName);

          // Absolute buffer position = base offset for this veq + loop index
          Value absoluteOffset =
              builder.template create<arith::AddIOp>(loc, index, currentOffset);
          recordMeasurement(builder, loc, measurement.getMeasOut(), buffer,
                            absoluteOffset, discriminate);
        });

    currentOffset =
        rewriter.template create<arith::AddIOp>(loc, currentOffset, vectorSize);
  }

  // Measure all targets in the operation
  void measureAllTargets(PatternRewriter &rewriter, Location loc, A measureOp,
                         Value buffer, Value &offset, Value increment,
                         bool discriminate) const {
    for (auto target : measureOp.getTargets()) {
      if (isa<quake::RefType>(target.getType())) {
        measureSingleQubit(rewriter, loc, target, offset, buffer, increment,
                           measureOp, discriminate);
      } else {
        assert(isa<quake::VeqType>(target.getType()));
        measureQubitVector(rewriter, loc, target, offset, buffer, measureOp,
                           discriminate);
      }
    }
  }

  // Packages the buffer into a stdvec and replaces the appropriate op:
  // - Discriminated: replaces the discriminate op (measurement op is erased)
  // - Non-discriminated: replaces the measurement op itself
  //
  // The resulting stdvec type matches what downstream code expects:
  // !cc.stdvec<i1> for booleans, !cc.stdvec<!quake.measure> for raw results.
  void finalizeResults(PatternRewriter &rewriter, Location loc, A measureOp,
                       Value buffer, Value count, bool hasDiscriminate,
                       quake::DiscriminateOp discriminateOp) const {
    auto ctx = rewriter.getContext();

    if (hasDiscriminate) {
      // Create vector of boolean results
      auto boolType = rewriter.getI1Type();
      auto vecType = cudaq::cc::StdvecType::get(ctx, boolType);
      auto arrayPtrType =
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(boolType));
      auto castedBuffer = rewriter.template create<cudaq::cc::CastOp>(
          loc, arrayPtrType, buffer);

      rewriter.template replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
          discriminateOp, vecType, castedBuffer, count);
      rewriter.eraseOp(measureOp);
    } else {
      // Create vector of measurement results
      auto measType = quake::MeasureType::get(ctx);
      auto vecType = cudaq::cc::StdvecType::get(ctx, measType);
      auto arrayPtrType =
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(measType));
      auto castedBuffer = rewriter.template create<cudaq::cc::CastOp>(
          loc, arrayPtrType, buffer);
      rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(measureOp, vecType,
                                                           castedBuffer, count);
    }
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
