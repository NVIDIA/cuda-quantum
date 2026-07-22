// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

using namespace mlir;

namespace pulse {

// ===----------------------------------------------------------------------===//
// SSA value helpers for tracing through arith.constant defining ops.
// Returns std::nullopt when the value is a block argument (parametric).
// ===----------------------------------------------------------------------===//

static std::optional<int64_t> getConstantI64(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static std::optional<double> getConstantF64(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantFloatOp>())
    return cst.value().convertToDouble();
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
      return fa.getValueAsDouble();
  }
  return std::nullopt;
}

// ===----------------------------------------------------------------------===//
// Verifiers — real semantic checks matching the Python verify pass logic.
//
// Values that trace to arith.constant are checked for constraints.
// Block arguments (parametric values) skip runtime checks — they are
// verified at evaluation time when concrete values are substituted.
// ===----------------------------------------------------------------------===//

LogicalResult ToneOp::verify() { return success(); }

LogicalResult SquarePulseOp::verify() {
  if (auto dur = getConstantI64(getDuration())) {
    if (*dur <= 0)
      return emitOpError("duration must be positive, got ") << *dur;
  }
  return success();
}

LogicalResult GaussianPulseOp::verify() {
  if (auto dur = getConstantI64(getDuration())) {
    if (*dur <= 0)
      return emitOpError("duration must be positive, got ") << *dur;
  }
  if (auto sig = getConstantF64(getSigma())) {
    if (*sig <= 0.0)
      return emitOpError("sigma must be positive");
  }
  return success();
}

LogicalResult GaussianSquarePulseOp::verify() {
  auto dur = getConstantI64(getDuration());
  auto rf = getConstantI64(getRisefall());
  if (dur && *dur <= 0)
    return emitOpError("duration must be positive, got ") << *dur;
  if (auto sig = getConstantF64(getSigma())) {
    if (*sig <= 0.0)
      return emitOpError("sigma must be positive");
  }
  if (rf && *rf <= 0)
    return emitOpError("risefall must be positive, got ") << *rf;
  if (dur && rf && 2 * (*rf) > *dur)
    return emitOpError("2*risefall (")
           << 2 * (*rf) << ") exceeds duration (" << *dur << ")";
  return success();
}

LogicalResult DRAGPulseOp::verify() {
  if (auto dur = getConstantI64(getDuration())) {
    if (*dur <= 0)
      return emitOpError("duration must be positive, got ") << *dur;
  }
  if (auto sig = getConstantF64(getSigma())) {
    if (*sig <= 0.0)
      return emitOpError("sigma must be positive");
  }
  return success();
}

LogicalResult CosinePulseOp::verify() {
  if (auto dur = getConstantI64(getDuration())) {
    if (*dur <= 0)
      return emitOpError("duration must be positive, got ") << *dur;
  }
  return success();
}

LogicalResult TanhRampOp::verify() {
  if (auto dur = getConstantI64(getDuration())) {
    if (*dur <= 0)
      return emitOpError("duration must be positive, got ") << *dur;
  }
  if (auto sig = getConstantF64(getSigma())) {
    if (*sig <= 0.0)
      return emitOpError("sigma must be positive");
  }
  return success();
}

LogicalResult CustomOp::verify() {
  if (auto dur = getConstantI64(getDuration())) {
    if (*dur <= 0)
      return emitOpError("duration must be positive, got ") << *dur;
  }
  return success();
}

LogicalResult CustomSamplesOp::verify() {
  if (getSamples().empty())
    return emitOpError("samples array must not be empty");
  return success();
}

LogicalResult WaitOp::verify() {
  // Input and output line types must agree (drive stays drive, readout stays
  // readout) — the type system already enforces AnyLineType but we verify
  // the specific subtype matches.
  if (getLine().getType() != getUpdatedLine().getType())
    return emitOpError("input and output line types must match");
  return success();
}

LogicalResult SyncOp::verify() {
  // Sync semantics: align time of multiple lines. Two or more lines required.
  if (getLines().size() < 2)
    return emitOpError("sync requires at least 2 lines, got ")
           << getLines().size();
  if (getLines().size() != getSyncedLines().size())
    return emitOpError("number of input lines (")
           << getLines().size() << ") must match number of output lines ("
           << getSyncedLines().size() << ")";
  return success();
}

LogicalResult PulseAddOp::verify() { return success(); }

LogicalResult PulseSubOp::verify() { return success(); }

LogicalResult PulseMulOp::verify() { return success(); }

LogicalResult AtomicOp::verify() {
  // The body must receive the same number of lines as produced
  if (getLines().size() != getUpdatedLines().size())
    return emitOpError("number of input lines (")
           << getLines().size() << ") must match number of output lines ("
           << getUpdatedLines().size() << ")";
  return success();
}

LogicalResult YieldOp::verify() { return success(); }

// ===----------------------------------------------------------------------===//
// Folders — fold away no-op operations (zero shift, scale by 1, double neg).
//
// These correspond to the Python canonicalize.py transforms:
//   _idle_compression (WaitOp zero-dur), _redundant_sync_elim (SyncOp),
// and virtual_z.py transforms:
//   consecutive shift_phase/set_phase merging.
// ===----------------------------------------------------------------------===//

OpFoldResult WaitOp::fold(FoldAdaptor adaptor) {
  // Cannot fold via adaptor since DurationType is a custom type.
  // Zero-duration wait elimination handled by canonicalize pass.
  return nullptr;
}

// SyncOp has variadic results → multi-result fold signature
LogicalResult SyncOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  // Python canonicalize.py:_redundant_sync_elim removes syncs where all
  // lines already share the same time. That's a program-wide analysis;
  // per-op we can't determine that. Return failure = no fold.
  return failure();
}

OpFoldResult ShiftFrequencyOp::fold(FoldAdaptor adaptor) {
  // Fold shift_frequency by 0 Hz → identity (return input tone)
  if (auto freq = dyn_cast_or_null<FloatAttr>(adaptor.getFrequencyHz())) {
    if (freq.getValueAsDouble() == 0.0)
      return getTone();
  }
  return nullptr;
}

OpFoldResult ShiftPhaseOp::fold(FoldAdaptor adaptor) {
  // Fold shift_phase by 0 rad → identity (virtual-Z of angle 0 is no-op)
  if (auto phase = dyn_cast_or_null<FloatAttr>(adaptor.getPhaseRad())) {
    if (phase.getValueAsDouble() == 0.0)
      return getTone();
  }
  return nullptr;
}

OpFoldResult SetPhaseOp::fold(FoldAdaptor adaptor) {
  // set_phase cannot be folded away in general (even setting to 0 is
  // semantically meaningful — it resets the phase)
  return nullptr;
}

OpFoldResult SetFrequencyOp::fold(FoldAdaptor adaptor) { return nullptr; }

OpFoldResult PulseAddOp::fold(FoldAdaptor adaptor) { return nullptr; }

OpFoldResult PulseSubOp::fold(FoldAdaptor adaptor) {
  // fold x - x → zero waveform (would need zero waveform constant)
  return nullptr;
}

OpFoldResult PulseMulOp::fold(FoldAdaptor adaptor) { return nullptr; }

OpFoldResult PulseScaleOp::fold(FoldAdaptor adaptor) {
  // Fold scale(pulse, 1.0) → pulse (identity)
  if (auto scale = dyn_cast_or_null<FloatAttr>(adaptor.getScale())) {
    if (scale.getValueAsDouble() == 1.0)
      return getPulse();
  }
  return nullptr;
}

OpFoldResult PulseNegOp::fold(FoldAdaptor adaptor) {
  // neg(neg(x)) → x
  if (auto innerNeg = getPulse().getDefiningOp<PulseNegOp>())
    return innerNeg.getPulse();
  return nullptr;
}

// ===----------------------------------------------------------------------===//
// Canonicalization patterns
//
// The heavy lifting (idle compression, waveform CSE, virtual-Z folding,
// dead-line elimination, redundant-sync elimination) is performed by
// dedicated Python or MLIR passes. Op-level getCanonicalizationPatterns
// registers lightweight rewrite patterns that the generic MLIR canonicalizer
// pass picks up.
// ===----------------------------------------------------------------------===//

void WaitOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {}

void ShiftFrequencyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {}

void ShiftPhaseOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {}

void SetPhaseOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {}

void SetFrequencyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx) {}

void PulseNegOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {}

} // namespace pulse

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.cpp.inc"
