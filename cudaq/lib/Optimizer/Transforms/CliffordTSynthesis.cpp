/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#ifdef CUDAQ_HAS_CLIFFORD_T_SYNTHESIS
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#endif

namespace cudaq::opt {
#define GEN_PASS_DEF_CLIFFORDTSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "clifford-t-synthesis"

using namespace mlir;

#ifdef CUDAQ_HAS_CLIFFORD_T_SYNTHESIS

namespace {

struct RotationOptions {
  double epsilon;
  int32_t diophantineTimeoutMs;
  int32_t factoringTimeoutMs;
  int32_t retryCount;
  std::string onDynamicAngle;
  double skipBelow;
};

struct SynthState {
  mlir::ModuleOp module;
  llvm::DenseMap<uint64_t, mlir::FlatSymbolRefAttr> cache;
  uint64_t hits = 0;
};

// Materialize a body of Clifford+T gates onto `qubit` from a synthesized
// `Circuit`.
void emitCircuitBody(OpBuilder &b, Location loc, Value qubit,
                     const cudaq::synth::Circuit &circuit) {
  for (cudaq::synth::Gate g : circuit) {
    switch (g) {
    case cudaq::synth::Gate::H:
      cudaq::quake::HOp::create(b, loc, ValueRange{qubit});
      break;
    case cudaq::synth::Gate::S:
      cudaq::quake::SOp::create(b, loc, ValueRange{qubit});
      break;
    case cudaq::synth::Gate::T:
      cudaq::quake::TOp::create(b, loc, ValueRange{qubit});
      break;
    case cudaq::synth::Gate::X:
      cudaq::quake::XOp::create(b, loc, ValueRange{qubit});
      break;
    case cudaq::synth::Gate::W:
      // omega = e^{i*pi/4} global phase - dropped.
      break;
    }
  }
}

// Resolve (theta, epsilon) to a private helper func that applies the
// synthesized Clifford+T sequence to its `!quake.ref` argument. On a cache
// miss we run gridsynth, materialize the helper at module top level, and
// stash a FlatSymbolRefAttr so subsequent rotations with the same angle
// just emit a func.call.
llvm::FailureOr<mlir::FlatSymbolRefAttr>
getOrCreateRzHelper(double theta, const RotationOptions &opts,
                    SynthState &state) {
  uint64_t key = llvm::bit_cast<uint64_t>(theta);
  auto it = state.cache.find(key);
  if (it != state.cache.end()) {
    ++state.hits;
    LLVM_DEBUG(llvm::dbgs() << "clifford-t-synthesis: cache hit for theta="
                            << theta << '\n');
    return it->second;
  }

  cudaq::synth::Real thetaReal(theta);
  cudaq::synth::Real epsilonReal(opts.epsilon);
  llvm::FailureOr<cudaq::synth::Circuit> circuit = llvm::failure();
  for (int32_t attempt = 0; attempt <= opts.retryCount; ++attempt) {
    circuit = cudaq::synth::gridsynth(
        thetaReal, epsilonReal,
        static_cast<int32_t>(opts.diophantineTimeoutMs << attempt),
        static_cast<int32_t>(opts.factoringTimeoutMs << attempt));
    if (llvm::succeeded(circuit))
      break;
  }
  if (llvm::failed(circuit))
    return llvm::failure();

  MLIRContext *ctx = state.module.getContext();
  std::string name = llvm::formatv("__cliffordt_rz_{0:x-16}", key).str();
  Location loc = state.module.getLoc();
  auto refType = cudaq::quake::RefType::get(ctx);

  OpBuilder b(ctx);
  b.setInsertionPointToStart(state.module.getBody());
  auto funcOp =
      func::FuncOp::create(b, loc, name, b.getFunctionType({refType}, {}));
  funcOp.setPrivate();
  Block *entry = funcOp.addEntryBlock();
  b.setInsertionPointToStart(entry);
  emitCircuitBody(b, loc, entry->getArgument(0), *circuit);
  func::ReturnOp::create(b, loc);

  auto symRef = mlir::FlatSymbolRefAttr::get(ctx, funcOp.getNameAttr());
  state.cache.try_emplace(key, symRef);
  return symRef;
}

// Outcome of validateRotationOperands. The caller maps each action to a
// LogicalResult:
//    Lower -> proceed and return success()
//    LeaveInPlace -> return failure() with the op untouched
//    Erased -> return success() with no further emission.
struct PreCheck {
  enum class Action { Lower, LeaveInPlace, Erased };
  Action action;
  double theta;
};

// Validates the common preconditions shared by every rotation lowering:
//   - controls non-empty                  -> remark, LeaveInPlace
//   - non-constant angle, on-dyn=error    -> error,  LeaveInPlace
//   - non-constant angle, on-dyn=skip     -> remark, LeaveInPlace
//   - NaN/Inf angle                       -> error,  LeaveInPlace
//   - |theta| < skipBelow                 -> erase,  Erased
//   - otherwise                           -> Lower with the constant angle
PreCheck validateRotationOperands(Operation *op, Value angleVal,
                                  ValueRange controls,
                                  PatternRewriter &rewriter,
                                  const RotationOptions &opts) {
  if (!controls.empty()) {
    op->emitRemark("clifford-t-synthesis: skipping controlled rotation; run "
                   "ApplyOpSpecialization to materialize controls before "
                   "synthesis");
    return {PreCheck::Action::LeaveInPlace, 0.0};
  }

  FloatAttr attr;
  if (!matchPattern(angleVal, m_Constant<FloatAttr>(&attr))) {
    if (opts.onDynamicAngle == "skip") {
      op->emitRemark("clifford-t-synthesis: skipping non-constant rotation "
                     "angle (--on-dynamic-angle=skip)");
    } else {
      op->emitError("clifford-t-synthesis: rotation angle is not a "
                    "compile-time constant. Fold the angle upstream or set "
                    "--on-dynamic-angle=skip to leave the op alone.");
    }
    return {PreCheck::Action::LeaveInPlace, 0.0};
  }

  double theta = attr.getValueAsDouble();
  if (!std::isfinite(theta)) {
    op->emitError("clifford-t-synthesis: rotation angle is NaN or Inf");
    return {PreCheck::Action::LeaveInPlace, 0.0};
  }

  if (std::abs(theta) < opts.skipBelow) {
    rewriter.eraseOp(op);
    return {PreCheck::Action::Erased, 0.0};
  }

  return {PreCheck::Action::Lower, theta};
}

struct RzPattern : OpRewritePattern<cudaq::quake::RzOp> {
  RzPattern(MLIRContext *ctx, RotationOptions opts, SynthState *state)
      : OpRewritePattern(ctx), opts(std::move(opts)), state(state) {}

  LogicalResult matchAndRewrite(cudaq::quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    auto check = validateRotationOperands(op, op.getParameter(),
                                          op.getControls(), rewriter, opts);
    switch (check.action) {
    case PreCheck::Action::LeaveInPlace:
      return failure();
    case PreCheck::Action::Erased:
      return success();
    case PreCheck::Action::Lower:
      break;
    }

    auto symRef = getOrCreateRzHelper(check.theta, opts, *state);
    if (llvm::failed(symRef)) {
      op.emitError("clifford-t-synthesis: gridsynth failed for theta=")
          << check.theta << " after " << (opts.retryCount + 1)
          << " attempts; raise --diophantine-timeout-ms or "
             "--factoring-timeout-ms";
      return failure();
    }

    // Replace the rotation with a call to the per-angle helper. Inlining
    // every rotation site would bloat the IR by O(rotations * gate_count).
    // Calling a shared helper keeps it at O(rotations + unique_angles *
    // gate_count).
    rewriter.replaceOpWithNewOp<func::CallOp>(op, *symRef, TypeRange{},
                                              ValueRange{op.getTarget()});
    return success();
  }

  RotationOptions opts;
  SynthState *state;
};

// Rx(theta) = H . Rz(theta) . H. The greedy driver then re-fires RzPattern
// on the inner Rz to do the actual Clifford+T expansion.
struct RxPattern : OpRewritePattern<cudaq::quake::RxOp> {
  RxPattern(MLIRContext *ctx, RotationOptions opts)
      : OpRewritePattern(ctx), opts(std::move(opts)) {}

  LogicalResult matchAndRewrite(cudaq::quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    auto check = validateRotationOperands(op, op.getParameter(),
                                          op.getControls(), rewriter, opts);
    switch (check.action) {
    case PreCheck::Action::LeaveInPlace:
      return failure();
    case PreCheck::Action::Erased:
      return success();
    case PreCheck::Action::Lower:
      break;
    }

    Location loc = op.getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    cudaq::quake::HOp::create(rewriter, loc, ValueRange{target});
    cudaq::quake::RzOp::create(rewriter, loc, ValueRange{angle}, ValueRange{},
                               ValueRange{target});
    cudaq::quake::HOp::create(rewriter, loc, ValueRange{target});
    rewriter.eraseOp(op);
    return success();
  }

  RotationOptions opts;
};

// Ry(theta) = S . H . Rz(theta) . H . S^dagger. Emitted as S.S.S . H . Rz .
// H . S in circuit order (S^dagger = S^3 since S^4 = I), keeping the output
// strictly in the Clifford+T alphabet {H, S, T, X}.
struct RyPattern : OpRewritePattern<cudaq::quake::RyOp> {
  RyPattern(MLIRContext *ctx, RotationOptions opts)
      : OpRewritePattern(ctx), opts(std::move(opts)) {}

  LogicalResult matchAndRewrite(cudaq::quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    auto check = validateRotationOperands(op, op.getParameter(),
                                          op.getControls(), rewriter, opts);
    switch (check.action) {
    case PreCheck::Action::LeaveInPlace:
      return failure();
    case PreCheck::Action::Erased:
      return success();
    case PreCheck::Action::Lower:
      break;
    }

    Location loc = op.getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    cudaq::quake::SOp::create(rewriter, loc, ValueRange{target});
    cudaq::quake::SOp::create(rewriter, loc, ValueRange{target});
    cudaq::quake::SOp::create(rewriter, loc, ValueRange{target});
    cudaq::quake::HOp::create(rewriter, loc, ValueRange{target});
    cudaq::quake::RzOp::create(rewriter, loc, ValueRange{angle}, ValueRange{},
                               ValueRange{target});
    cudaq::quake::HOp::create(rewriter, loc, ValueRange{target});
    cudaq::quake::SOp::create(rewriter, loc, ValueRange{target});
    rewriter.eraseOp(op);
    return success();
  }

  RotationOptions opts;
};

// R1(theta) = e^{i theta/2} . Rz(theta). The leading global phase is
// dropped for now and we will revisit when controlled rotations are
// reintroduced.
struct R1Pattern : OpRewritePattern<cudaq::quake::R1Op> {
  R1Pattern(MLIRContext *ctx, RotationOptions opts)
      : OpRewritePattern(ctx), opts(std::move(opts)) {}

  LogicalResult matchAndRewrite(cudaq::quake::R1Op op,
                                PatternRewriter &rewriter) const override {
    auto check = validateRotationOperands(op, op.getParameter(),
                                          op.getControls(), rewriter, opts);
    switch (check.action) {
    case PreCheck::Action::LeaveInPlace:
      return failure();
    case PreCheck::Action::Erased:
      return success();
    case PreCheck::Action::Lower:
      break;
    }

    Location loc = op.getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    cudaq::quake::RzOp::create(rewriter, loc, ValueRange{angle}, ValueRange{},
                               ValueRange{target});
    rewriter.eraseOp(op);
    return success();
  }

  RotationOptions opts;
};

} // namespace

namespace cudaq::opt::detail {

namespace {
uint64_t &mutableLastCacheHits() {
  static uint64_t v = 0;
  return v;
}
uint64_t &mutableLastCacheUniqueAngles() {
  static uint64_t v = 0;
  return v;
}
} // namespace

uint64_t lastCliffordTSynthCacheHits() { return mutableLastCacheHits(); }
uint64_t lastCliffordTSynthCacheUniqueAngles() {
  return mutableLastCacheUniqueAngles();
}

} // namespace cudaq::opt::detail

#endif // CUDAQ_HAS_CLIFFORD_T_SYNTHESIS

namespace {

class CliffordTSynthesisPass
    : public cudaq::opt::impl::CliffordTSynthesisBase<CliffordTSynthesisPass> {
public:
  using CliffordTSynthesisBase::CliffordTSynthesisBase;

  void runOnOperation() override {
#ifndef CUDAQ_HAS_CLIFFORD_T_SYNTHESIS
    getOperation().emitError(
        "clifford-t-synthesis: CUDA-Q was built without the synthesis "
        "library (GMP/MPFR missing at configure time). Rebuild with "
        "libgmp-dev and libmpfr-dev installed.");
    signalPassFailure();
    return;
#else
    LLVM_DEBUG(llvm::dbgs()
               << "clifford-t-synthesis: epsilon=" << epsilon
               << " diophantine-timeout-ms=" << diophantineTimeoutMs
               << " factoring-timeout-ms=" << factoringTimeoutMs
               << " retry-count=" << retryCount << " on-dynamic-angle="
               << onDynamicAngle << " skip-below=" << skipBelow << '\n');

    // Reject value-semantics IR.
    {
      WalkResult walk = getOperation().walk([&](Operation *op) {
        if (!isa<cudaq::quake::RxOp, cudaq::quake::RyOp, cudaq::quake::RzOp,
                 cudaq::quake::R1Op>(op))
          return WalkResult::advance();
        for (Value target :
             cast<cudaq::quake::OperatorInterface>(op).getTargets()) {
          if (!cudaq::quake::isQuantumReferenceType(target.getType())) {
            op->emitError(
                "clifford-t-synthesis: rotation target is in value-semantics "
                "form (!quake.wire/!quake.cable); run `regtomem` to convert to "
                "memory semantics (!quake.ref) before this pass.");
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (walk.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    }

    auto prec = static_cast<mpfr_prec_t>(
        std::max<double>(64.0, std::ceil(-std::log2(epsilon) * 4.0 + 64.0)));
    cudaq::synth::Real::set_default_precision(prec);

    RotationOptions opts{
        epsilon,    diophantineTimeoutMs,      factoringTimeoutMs,
        retryCount, onDynamicAngle.getValue(), skipBelow};

    MLIRContext *ctx = &getContext();
    SynthState state;
    state.module = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.add<R1Pattern, RxPattern, RyPattern>(ctx, opts);
    patterns.add<RzPattern>(ctx, opts, &state);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();

    cudaq::opt::detail::mutableLastCacheHits() = state.hits;
    cudaq::opt::detail::mutableLastCacheUniqueAngles() = state.cache.size();

    LLVM_DEBUG(llvm::dbgs() << "clifford-t-synthesis: outlined "
                            << state.cache.size() << " unique angle(s), reused "
                            << "via " << state.hits << " cache hit(s)\n");
#endif
  }
};

} // namespace
