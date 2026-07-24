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
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <limits>

namespace cudaq::opt {
#define GEN_PASS_DEF_CLIFFORDTSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "clifford-t-synthesis"

using namespace mlir;

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
  // Keyed on (theta bits, valueSemantics ? 1 : 0): a ref-form and a wire-form
  // helper for the same angle have different signatures, so they must not
  // collide.
  // Note: DenseMapInfo has no hash for `bool`, hence the unsigned tag.
  llvm::DenseMap<std::pair<uint64_t, unsigned>, mlir::FlatSymbolRefAttr> cache;
  uint64_t hits = 0;
};

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

} // namespace

// Emit a single-target, parameter-free Clifford gate on `target`. In value
// semantics `target` is a `!quake.wire`: the gate consumes it and produces a
// fresh wire, which is returned so the caller can thread it into the next
// gate. In memory semantics `target` is a `!quake.ref`: the gate mutates it in
// place and the same value is returned unchanged. Detecting the form from the
// operand type lets one code path serve both.
template <typename OpTy>
static Value emitCliffordGate(OpBuilder &b, Location loc, Value target) {
  if (isa<cudaq::quake::WireType>(target.getType())) {
    auto op =
        OpTy::create(b, loc, TypeRange{target.getType()}, /*is_adj=*/false,
                     /*parameters=*/ValueRange{},
                     /*controls=*/ValueRange{},
                     /*targets=*/ValueRange{target},
                     /*negated_qubit_controls=*/DenseBoolArrayAttr{});
    return op.getWires()[0];
  }
  OpTy::create(b, loc, ValueRange{target});
  return target;
}

// Materialize a body of Clifford+T gates onto `qubit` from a synthesized
// `Circuit`, threading value-semantics wires when `qubit` is a `!quake.wire`.
// Returns the final qubit value (the last wire for value semantics, or `qubit`
// unchanged for memory semantics).
static Value emitCircuitBody(OpBuilder &b, Location loc, Value qubit,
                             const cudaq::synth::Circuit &circuit) {
  Value cur = qubit;
  for (cudaq::synth::Gate g : circuit) {
    switch (g) {
    case cudaq::synth::Gate::H:
      cur = emitCliffordGate<cudaq::quake::HOp>(b, loc, cur);
      break;
    case cudaq::synth::Gate::S:
      cur = emitCliffordGate<cudaq::quake::SOp>(b, loc, cur);
      break;
    case cudaq::synth::Gate::T:
      cur = emitCliffordGate<cudaq::quake::TOp>(b, loc, cur);
      break;
    case cudaq::synth::Gate::X:
      cur = emitCliffordGate<cudaq::quake::XOp>(b, loc, cur);
      break;
    case cudaq::synth::Gate::W:
      // omega = e^{i*pi/4} global phase - dropped.
      // TODO: emit this phase once global-phase support lands in Quake.
      break;
    }
  }
  return cur;
}

// Compute `base << shift` as an int32_t timeout, saturating at INT32_MAX
// instead of overflowing. `base` and `shift` are validated non-negative by the
// pass before this is called, so the only hazard is the retry loop scaling the
// timeout past what int32_t can hold.
static int32_t saturatingShlToInt32(int32_t base, int32_t shift) {
  assert(base >= 0 && shift >= 0 && "timeouts and retry count must be >= 0");
  constexpr int32_t kMax = std::numeric_limits<int32_t>::max();
  // Shifting an int32_t by >= 31 always overflows a non-negative base.
  if (shift >= 31)
    return kMax;
  int64_t scaled = static_cast<int64_t>(base) << shift;
  return static_cast<int32_t>(std::min<int64_t>(scaled, kMax));
}

// Resolve (theta, epsilon) to a private helper func that applies the
// synthesized Clifford+T sequence to its qubit argument. `valueSemantics`
// selects the signature: `(!quake.wire) -> !quake.wire` (threading the wire)
// when true, `(!quake.ref) -> ()` (in-place) when false. On a cache miss we
// run gridsynth, materialize the helper at module top level, and stash a
// FlatSymbolRefAttr so subsequent rotations with the same angle and form just
// emit a func.call.
static llvm::FailureOr<mlir::FlatSymbolRefAttr>
getOrCreateRzHelper(double theta, bool valueSemantics,
                    const RotationOptions &opts, SynthState &state) {
  uint64_t bits = llvm::bit_cast<uint64_t>(theta);
  auto key = std::make_pair(bits, valueSemantics ? 1u : 0u);
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
        saturatingShlToInt32(opts.diophantineTimeoutMs, attempt),
        saturatingShlToInt32(opts.factoringTimeoutMs, attempt));
    if (llvm::succeeded(circuit))
      break;
  }
  if (llvm::failed(circuit))
    return llvm::failure();

  MLIRContext *ctx = state.module.getContext();
  std::string name =
      valueSemantics ? llvm::formatv("__cliffordt_rz_wire_{0:x-16}", bits).str()
                     : llvm::formatv("__cliffordt_rz_{0:x-16}", bits).str();
  Location loc = state.module.getLoc();
  Type qubitType = valueSemantics ? Type(cudaq::quake::WireType::get(ctx))
                                  : Type(cudaq::quake::RefType::get(ctx));

  OpBuilder b(ctx);
  b.setInsertionPointToStart(state.module.getBody());
  FunctionType fnType = valueSemantics
                            ? b.getFunctionType({qubitType}, {qubitType})
                            : b.getFunctionType({qubitType}, {});
  auto funcOp = func::FuncOp::create(b, loc, name, fnType);
  funcOp.setPrivate();
  Block *entry = funcOp.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value out = emitCircuitBody(b, loc, entry->getArgument(0), *circuit);
  if (valueSemantics)
    func::ReturnOp::create(b, loc, ValueRange{out});
  else
    func::ReturnOp::create(b, loc);

  auto symRef = mlir::FlatSymbolRefAttr::get(ctx, funcOp.getNameAttr());
  state.cache.try_emplace(key, symRef);
  return symRef;
}

// Validates the common preconditions shared by every rotation lowering:
//   - controls non-empty                  -> remark, LeaveInPlace
//   - adjoint rotation                    -> error,  LeaveInPlace (hard error)
//   - non-constant angle, on-dyn=error    -> error,  LeaveInPlace (hard error)
//   - non-constant angle, on-dyn=skip     -> remark, LeaveInPlace
//   - NaN/Inf angle                       -> error,  LeaveInPlace (hard error)
//   - |theta| < skipBelow                 -> erase,  Erased
//   - otherwise                           -> Lower with the constant angle
static PreCheck validateRotationOperands(Operation *op, Value angleVal,
                                         ValueRange controls,
                                         PatternRewriter &rewriter,
                                         const RotationOptions &opts,
                                         bool *hadHardError) {
  if (!controls.empty()) {
    op->emitRemark("clifford-t-synthesis: skipping controlled rotation; run "
                   "ApplyOpSpecialization to materialize controls before "
                   "synthesis");
    return {PreCheck::Action::LeaveInPlace, 0.0};
  }

  // Adjoint rotations must be resolved before this pass.
  if (cast<cudaq::quake::OperatorInterface>(op).isAdj()) {
    op->emitError("clifford-t-synthesis: adjoint rotation reached synthesis.");
    *hadHardError = true;
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
      *hadHardError = true;
    }
    return {PreCheck::Action::LeaveInPlace, 0.0};
  }

  double theta = attr.getValueAsDouble();
  if (!std::isfinite(theta)) {
    op->emitError("clifford-t-synthesis: rotation angle is NaN or Inf");
    *hadHardError = true;
    return {PreCheck::Action::LeaveInPlace, 0.0};
  }

  // gridsynth is never invoked for an angle that would be erased anyway.
  if (std::abs(theta) < opts.skipBelow) {
    rewriter.eraseOp(op);
    return {PreCheck::Action::Erased, 0.0};
  }

  return {PreCheck::Action::Lower, theta};
}

namespace {

struct RzPattern : OpRewritePattern<cudaq::quake::RzOp> {
  RzPattern(MLIRContext *ctx, RotationOptions opts, SynthState *state,
            bool *hadHardError)
      : OpRewritePattern(ctx), opts(std::move(opts)), state(state),
        hadHardError(hadHardError) {}

  LogicalResult matchAndRewrite(cudaq::quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    auto check = validateRotationOperands(
        op, op.getParameter(), op.getControls(), rewriter, opts, hadHardError);
    switch (check.action) {
    case PreCheck::Action::LeaveInPlace:
      return failure();
    case PreCheck::Action::Erased:
      return success();
    case PreCheck::Action::Lower:
      break;
    }

    Value target = op.getTarget();
    bool valueSemantics = isa<cudaq::quake::WireType>(target.getType());
    auto symRef =
        getOrCreateRzHelper(check.theta, valueSemantics, opts, *state);
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
    // gate_count). In value semantics the call consumes and produces the
    // qubit wire; in memory semantics it returns nothing.
    if (valueSemantics)
      rewriter.replaceOpWithNewOp<func::CallOp>(
          op, *symRef, TypeRange{target.getType()}, ValueRange{target});
    else
      rewriter.replaceOpWithNewOp<func::CallOp>(op, *symRef, TypeRange{},
                                                ValueRange{target});
    return success();
  }

  RotationOptions opts;
  SynthState *state;
  bool *hadHardError;
};

} // namespace

namespace {

class CliffordTSynthesisPass
    : public cudaq::opt::impl::CliffordTSynthesisBase<CliffordTSynthesisPass> {
public:
  using CliffordTSynthesisBase::CliffordTSynthesisBase;

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs()
               << "clifford-t-synthesis: epsilon=" << epsilon
               << " diophantine-timeout-ms=" << diophantineTimeoutMs
               << " factoring-timeout-ms=" << factoringTimeoutMs
               << " retry-count=" << retryCount << " on-dynamic-angle="
               << onDynamicAngle << " skip-below=" << skipBelow << '\n');

    // Validate the numeric options. gridsynth needs a positive epsilon
    // (-log2(epsilon) feeds the precision heuristic), and the timeouts/retry
    // count must be non-negative because the retry loop left-shifts the
    // timeouts by `attempt`.
    if (!(epsilon > 0.0) || diophantineTimeoutMs < 0 ||
        factoringTimeoutMs < 0 || retryCount < 0 || skipBelow < 0.0) {
      getOperation().emitError(
          "clifford-t-synthesis: invalid options; require epsilon > 0 and "
          "non-negative diophantine-timeout-ms, factoring-timeout-ms, "
          "retry-count, and skip-below.");
      signalPassFailure();
      return;
    }

    // Working precision (in bits) for the MPFR-backed reals gridsynth uses.
    // Representing a target of accuracy epsilon needs about log2(1/epsilon)
    // significant bits. The 4x factor supplies guard bits so rounding in
    // gridsynth's iterative arithmetic (candidate enumeration, Diophantine
    // solving) stays well below the epsilon budget, and the +64 / max(64, ...)
    // floor guarantees a sane minimum even for loose epsilon. This is an
    // empirical heuristic.
    auto prec = static_cast<mpfr_prec_t>(
        std::max<double>(64.0, std::ceil(-std::log2(epsilon) * 4.0 + 64.0)));
    const mpfr_prec_t savedPrecision =
        cudaq::synth::Real::get_default_precision();
    llvm::scope_exit precisionRestore(
        [&] { cudaq::synth::Real::set_default_precision(savedPrecision); });
    cudaq::synth::Real::set_default_precision(prec);

    RotationOptions opts{
        epsilon,    diophantineTimeoutMs,      factoringTimeoutMs,
        retryCount, onDynamicAngle.getValue(), skipBelow};

    MLIRContext *ctx = &getContext();
    SynthState state;
    state.module = getOperation();
    bool hadHardError = false;
    RewritePatternSet patterns(ctx);
    patterns.add<RzPattern>(ctx, opts, &state, &hadHardError);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        hadHardError)
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "clifford-t-synthesis: outlined "
                            << state.cache.size() << " unique angle(s), reused "
                            << "via " << state.hits << " cache hit(s)\n");
  }
};

} // namespace
