/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Move Pauli gates through a Clifford+T sequence
// so they can cancel with Paulis on the same wires. Preserve the global phase.
// For example,
//
//  ┌───┐┌───┐┌───┐               ┌────┐
//  ┤ X ├┤ T ├┤ X ├ = exp(iπ/4) × ┤ T† ├
//  └───┘└───┘└───┘               └────┘
//
// Try propagation from both ends. Emit the remaining Paulis and normalize
// the Clifford+T gates in each candidate before comparing gate counts.

#include "DecompositionPatterns.h"
#include "NormalizeSingleQubitCliffordT.h"
#include "PhaseUtilities.h"
#include "SingleQubitChain.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/OptimizationLevel.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/PauliProduct.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cmath>
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_SIGNEDPAULIPROPAGATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// The SSA values change as gates are emitted. Lane indices remain stable.
struct Gate {
  cudaq::quake::OperatorInterface op;
  unsigned target;
  std::optional<unsigned> control;
};

/// A contiguous run of supported gates in one block. Both sweeps use the
/// same first-use order for wire lanes.
struct Sequence {
  SmallVector<Value> inputs;  ///< First wire operand on each lane.
  SmallVector<Value> outputs; ///< Most recent wire result on each lane.
  /// Gates in execution order, with wires resolved to lanes.
  SmallVector<Gate> gates;
  /// Maps the current SSA value on each wire to its lane index.
  DenseMap<Value, unsigned> lanes;

  /// Advance one lane through the current gate.
  ///
  /// @param input SSA wire operand the gate consumes for a control or target.
  /// @param output Corresponding SSA wire result for that same qubit. It
  /// becomes the lane's current value for subsequent gates.
  /// @return The qubit's stable index in this sequence's `inputs` and
  /// `outputs`, assigned on first use and reused by subsequent gates.
  unsigned advanceWireLane(Value input, Value output) {
    const auto [it, fresh] = lanes.try_emplace(input, inputs.size());
    const unsigned index = it->second;
    // `fresh` is true when try_emplace inserted this SSA wire into the map.
    // Its first use in this sequence establishes a new input lane.
    if (fresh) {
      inputs.push_back(input);
      outputs.push_back(output);
    } else {
      outputs[index] = output;
    }
    // The gate consumes the old SSA value. Its next user sees the result.
    lanes.erase(it);
    lanes[output] = index;
    return index;
  }

  /// Try to extend the sequence with the next operation in the block.
  ///
  /// @param op Operation to check for exact Pauli propagation.
  /// @return True if the gate was appended and its wire lanes advanced.
  /// Otherwise, the sequence ends before `op`.
  bool tryAppendGate(Operation &op) {
    if (!isa<cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp,
             cudaq::quake::HOp, cudaq::quake::SOp, cudaq::quake::TOp>(op))
      return false;
    auto gate = cast<cudaq::quake::OperatorInterface>(&op);
    const auto flow = cudaq::quake::detail::getWireFlow(&op);
    // Only CX has a frame update for a controlled operation.
    if (!flow || gate.getTargets().size() != 1 ||
        (!gate.getControls().empty() &&
         (!isa<cudaq::quake::XOp>(op) || gate.getControls().size() != 1 ||
          llvm::is_contained(cudaq::quake::getControlPolarities(gate),
                             true))) ||
        !llvm::all_of(flow->inputs, [](Value v) { return v.hasOneUse(); }))
      return false;

    // Quake lists control wires before the target in both inputs and results.
    std::optional<unsigned> control;
    if (!gate.getControls().empty())
      control = advanceWireLane(flow->inputs.front(), flow->results.front());
    const auto target =
        advanceWireLane(flow->inputs.back(), flow->results.back());
    gates.push_back({gate, target, control});
    return true;
  }
};

/// A source gate that survived propagation. Only S and T can change adjoint.
struct RetainedGate {
  Gate gate;
  bool adjoint;
};

/// Sweep the circuit or its adjoint. After each gate the original prefix equals
/// the emitted gates followed in execution order by the pending Pauli product.
/// The additional scalar is in units of pi/4, since T can introduce an odd
/// power of exp(i pi/4). The Pauli product itself carries a power of i.
///
/// @return The additional phase modulo 8, or `std::nullopt` if a gate has no
/// propagation rule. In that case the caller must discard the partial frame
/// and retained gates.
static std::optional<unsigned>
propagatePauliFrame(const Sequence &seq, bool reverse,
                    cudaq::PauliProduct &frame,
                    SmallVectorImpl<RetainedGate> &result) {
  unsigned phase = 0;
  for (size_t i = 0, n = seq.gates.size(); i < n; ++i) {
    Gate g = seq.gates[reverse ? n - i - 1 : i];
    bool adjoint = false;
    if (!g.control &&
        isa<cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp>(g.op)) {
      //  ┌───┐┌───┐       ┌───┐
      //  ┤ P ├┤ G ├ = s × ┤ Q ├
      //  └───┘└───┘       └───┘
      //
      //  For each incoming G, the entry is the scalar s times the new Pauli Q.
      //  P \ G     X     Y     Z
      //    I       X     Y     Z
      //    X       I   -iZ    iY
      //    Y      iZ     I   -iX
      //    Z     -iY    iX     I
      frame.leftMultiplyPauli(g.target, !isa<cudaq::quake::ZOp>(g.op),
                              !isa<cudaq::quake::XOp>(g.op));
      continue;
    }
    if (!g.control && isa<cudaq::quake::HOp>(g.op)) {
      //  ┌───┐┌───┐       ┌───┐┌───┐
      //  ┤ P ├┤ H ├ = s × ┤ H ├┤ Q ├
      //  └───┘└───┘       └───┘└───┘
      //
      //   P      I   X   Y   Z
      //  s Q     I   Z  -Y   X
      //
      //  ┌───┐┌───┐┌───┐        ┌───┐
      //  ┤ H ├┤ Y ├┤ H ├ = -1 × ┤ Y ├
      //  └───┘└───┘└───┘        └───┘
      // H exchanges X and Z. When both are present, reordering ZX to XZ
      // supplies the minus sign in H Y H = -Y.
      const bool x = frame.x[g.target];
      const bool z = frame.z[g.target];
      if (x && z)
        frame.phase = (frame.phase + 2) % 4;
      frame.x.setBitVal(g.target, z);
      frame.z.setBitVal(g.target, x);
    } else if (g.control && isa<cudaq::quake::XOp>(g.op)) {
      //  ┌───┐                 ┌───┐
      //  ┤ A ├──●──       ──●──┤ C ├
      //  └───┘  │   = s ×   │  └───┘
      //  ┌───┐┌─┴─┐       ┌─┴─┐┌───┐
      //  ┤ B ├┤ X ├       ┤ X ├┤ D ├
      //  └───┘└───┘       └───┘└───┘
      //
      //  The upper wire is the control. Each entry is s times the output
      //  Pauli pair CD, control first.
      //  A \ B     I     X     Y     Z
      //    I      II    IX    ZY    ZZ
      //    X      XX    XI    YZ   -YY
      //    Y      YX    YI   -XZ    XY
      //    Z      ZI    ZX    IY    IZ
      //
      //  X on the control spreads to the target.
      //
      //  ┌───┐             ┌───┐
      //  ┤ X ├──●──   ──●──┤ X ├
      //  └───┘  │   =   │  └───┘
      //       ┌─┴─┐   ┌─┴─┐┌───┐
      //  ─────┤ X ├   ┤ X ├┤ X ├
      //       └───┘   └───┘└───┘
      //
      //  Z on the target spreads to the control.
      //
      //                    ┌───┐
      //  ───────●──   ──●──┤ Z ├
      //         │   =   │  └───┘
      //  ┌───┐┌─┴─┐   ┌─┴─┐┌───┐
      //  ┤ Z ├┤ X ├   ┤ X ├┤ Z ├
      //  └───┘└───┘   └───┘└───┘
      // CX preserves X-before-Z order in the frame. The minus signs above
      // arise when converting between the XZ factors and physical Y gates.
      if (frame.x[g.control.value()])
        frame.x.flipBit(g.target);
      if (frame.z[g.target])
        frame.z.flipBit(g.control.value());
    } else if (!g.control && isa<cudaq::quake::SOp, cudaq::quake::TOp>(g.op)) {
      adjoint = g.op.isAdj();
      //  For P = I or Z, D commutes with the frame.
      //
      //  ┌───┐┌───┐   ┌───┐┌───┐
      //  ┤ P ├┤ D ├ = ┤ D ├┤ P ├
      //  └───┘└───┘   └───┘└───┘
      //
      //  For P = X or Y, emit the adjoint and accumulate the scalar s.
      //
      //  ┌───┐┌───┐       ┌────┐┌───┐
      //  ┤ P ├┤ D ├ = s × ┤ D† ├┤ P ├
      //  └───┘└───┘       └────┘└───┘
      //
      //   D      S    S†     T           T†
      //  D†     S†    S      T†          T
      //   s      i   -i      exp(iπ/4)   exp(-iπ/4)
      if (frame.x[g.target]) {
        // For D = diag(1, omega^k), D X = omega^k X D†. The
        // exponent changes sign for an adjoint S or T.
        const unsigned exponent = isa<cudaq::quake::TOp>(g.op) ? 1 : 2;
        phase += (adjoint != reverse) ? 8 - exponent : exponent;
        adjoint = !adjoint;
      }
    } else {
      return std::nullopt;
    }
    // Equal gates on the same lanes cancel if their final forms are inverses.
    if (!result.empty()) {
      const auto &last = result.back();
      if (last.gate.op->getName() == g.op->getName() &&
          last.gate.target == g.target && last.gate.control == g.control &&
          (g.op->hasTrait<cudaq::Hermitian>() || last.adjoint != adjoint)) {
        result.pop_back();
        continue;
      }
    }
    result.push_back({g, adjoint});
  }
  return phase % 8;
}

/// Emit a gate and update its target and control wires to the corresponding
/// SSA results.
template <typename Op>
static void emitGate(OpBuilder &builder, Location loc, Value &target,
                     MutableArrayRef<Value> controls = {},
                     bool adjoint = false) {
  cudaq::quake::createAndThreadGate<Op>(
      builder, loc, adjoint ? builder.getUnitAttr() : UnitAttr{}, {}, controls,
      MutableArrayRef<Value>(target));
}

static void emitPauli(cudaq::quake::Pauli pauli, OpBuilder &builder,
                      Location loc, Value &wire) {
  switch (pauli) {
  case cudaq::quake::Pauli::X:
    return emitGate<cudaq::quake::XOp>(builder, loc, wire);
  case cudaq::quake::Pauli::Y:
    return emitGate<cudaq::quake::YOp>(builder, loc, wire);
  case cudaq::quake::Pauli::Z:
    return emitGate<cudaq::quake::ZOp>(builder, loc, wire);
  case cudaq::quake::Pauli::I:
    llvm_unreachable("identity has no gate");
  }
}

/// Anchor exp(i pi exponent/4) on a live wire to preserve the exact unitary.
static Value applyGlobalPhase(OpBuilder &builder, Location loc, Value wire,
                              unsigned exponent) {
  exponent %= 8;
  if (!exponent)
    return wire;
  const Value angle =
      cudaq::opt::factory::createF64Constant(loc, builder, exponent * M_PI_4);
  return cudaq::opt::emitPhaseCorrection(builder, loc, angle, {}, {}, wire)
      .anchor;
}

/// Build an exactly equivalent candidate by propagating Paulis from one end
/// of the sequence. The two directions can expose different cancellations.
///
/// For example, start with a CX followed by an X on each wire. Both circuits
/// below are drawn in execution order, from left to right.
///
///     Forward (3 gates)     Reverse (2 gates)
///            ┌───┐            ┌───┐
///     c ──●──┤ X ├          c ┤ X ├──●──
///         │  └───┘            └───┘  │
///       ┌─┴─┐┌───┐                 ┌─┴─┐
///     t ┤ X ├┤ X ├          t ─────┤ X ├
///       └───┘└───┘                 └───┘
///
/// Forward, the sweep emits the CX first and both pending X gates at the end.
/// In reverse, moving the control X back through the CX produces a target X
/// that cancels the target X already in the frame.
///
/// @param seq Nonempty run of supported gates in execution order. Its input
/// and output wires define the lane order of the candidate.
/// @param reverse If false, sweep from input to output. If true, sweep the
/// adjoint sequence and take the adjoint of the result.
/// @return A detached block in the original execution direction, with the
/// Pauli frame fully emitted. Block argument i corresponds to `seq.inputs[i]`
/// and operand i of its `func.return` corresponds to `seq.outputs[i]`.
/// A null block means a gate lacks a propagation or emission rule and the
/// caller must keep the original sequence.
static std::unique_ptr<Block> materializeCandidate(const Sequence &seq,
                                                   bool reverse) {
  const unsigned width = seq.inputs.size();
  cudaq::PauliProduct frame(width);
  SmallVector<RetainedGate> retained;
  const auto sweepPhase = propagatePauliFrame(seq, reverse, frame, retained);
  if (!sweepPhase)
    return nullptr;
  unsigned phase = sweepPhase.value() + 2 * frame.phase;

  auto block = std::make_unique<Block>();
  OpBuilder b(seq.inputs.front().getContext());
  b.setInsertionPointToEnd(block.get());
  SmallVector<Value> wires;
  for (Value input : seq.inputs)
    wires.push_back(block->addArgument(input.getType(), input.getLoc()));
  const Location loc = seq.gates.back().op->getLoc();
  const auto emitFrame = [&] {
    for (unsigned i = 0; i < width; ++i) {
      const unsigned lane = reverse ? width - i - 1 : i;
      const bool x = frame.x[lane], z = frame.z[lane];
      if (!x && !z)
        continue;
      if (x && z)
        phase += 6; // XZ = -iY.
      emitPauli(x && z ? cudaq::quake::Pauli::Y
                : x    ? cudaq::quake::Pauli::X
                       : cudaq::quake::Pauli::Z,
                b, reverse ? seq.gates.front().op->getLoc() : loc, wires[lane]);
    }
  };
  // Taking the adjoint of the reverse candidate moves its residual frame to
  // the front and restores the original order of the surviving source gates.
  if (reverse) {
    emitFrame();
    std::reverse(retained.begin(), retained.end());
  }
  for (const auto &[g, adjoint] : retained) {
    MutableArrayRef<Value> controls;
    if (g.control)
      controls = MutableArrayRef<Value>(wires[g.control.value()]);
    const auto emitted =
        llvm::TypeSwitch<Operation *, LogicalResult>(g.op)
            .Case<cudaq::quake::HOp, cudaq::quake::XOp, cudaq::quake::SOp,
                  cudaq::quake::TOp>([&](auto op) {
              emitGate<decltype(op)>(b, op.getLoc(), wires[g.target], controls,
                                     adjoint);
              return success();
            })
            .Default(failure());
    if (failed(emitted))
      return nullptr;
  }
  if (!reverse)
    emitFrame();
  wires.front() =
      applyGlobalPhase(b, loc, wires.front(), reverse ? 0u - phase : phase);
  func::ReturnOp::create(b, loc, wires);
  // Propagation can expose a simpler Clifford+T word. For example, moving X
  // through T† in T X T† X leaves two T gates and a phase.
  // Normalize before comparing candidates so those two gates count as one S.
  //  ┌───┐┌───┐┌────┐┌───┐                ┌───┐
  //  ┤ T ├┤ X ├┤ T† ├┤ X ├ = exp(-iπ/4) × ┤ S ├
  //  └───┘└───┘└────┘└───┘                └───┘
  cudaq::opt::normalizeSingleQubitCliffordT(*block);
  return block;
}

/// Physical gate counts used to compare an original sequence and its rewrites.
struct GateCounts {
  /// A candidate must strictly reduce the total to replace the original.
  size_t total = 0;
  /// T and T-adjoint gates both contribute to the T-count.
  size_t tCount = 0;
  /// Gates acting on two qubits, counting both controls and targets.
  size_t twoQubit = 0;
};

/// Count a physical Quake operation identically in the input and both sweeps.
static void countGate(GateCounts &counts,
                      cudaq::quake::OperatorInterface gate) {
  ++counts.total;
  counts.tCount += isa<cudaq::quake::TOp>(gate);
  counts.twoQubit += gate.getControls().size() + gate.getTargets().size() == 2;
}

/// Legalize the normalized candidate before comparing its physical gate count.
static std::optional<GateCounts>
legalizeAndCount(Block &block, const ConversionTarget *target) {
  GateCounts counts;
  unsigned phase = 0;
  for (Operation &op : llvm::make_early_inc_range(block.without_terminator())) {
    auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op);
    if (!gate)
      continue;
    if (isa<cudaq::quake::PhaseOp>(op)) {
      if (!gate.getControls().empty())
        return std::nullopt;
      continue;
    }
    if (!target || target->isLegal(&op)) {
      countGate(counts, gate);
      continue;
    }

    if (!gate.getControls().empty() ||
        !isa<cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp>(op))
      return std::nullopt;
    constexpr cudaq::quake::Pauli axes[] = {
        cudaq::quake::Pauli::X, cudaq::quake::Pauli::Y, cudaq::quake::Pauli::Z};
    const unsigned missing = isa<cudaq::quake::XOp>(op)   ? 0
                             : isa<cudaq::quake::YOp>(op) ? 1
                                                          : 2;
    OpBuilder builder(&op);
    Value wire = gate.getTargets().front();
    for (const unsigned offset : {1u, 2u}) {
      const cudaq::quake::Pauli axis = axes[(missing + offset) % 3];
      emitPauli(axis, builder, op.getLoc(), wire);
      Operation *emitted = wire.getDefiningOp();
      if (!target->isLegal(emitted))
        return std::nullopt;
      countGate(counts, cast<cudaq::quake::OperatorInterface>(emitted));
    }
    // In cyclic order the emitted pair equals -i times the missing Pauli
    // (ZY = -iX, XZ = -iY, YX = -iZ). Correct by +i, or two units of pi/4.
    phase += 2;
    op.getResult(0).replaceAllUsesWith(wire);
    op.erase();
  }
  if (phase % 8) {
    Operation &ret = block.back();
    OpBuilder builder(&ret);
    ret.setOperand(
        0, applyGlobalPhase(builder, ret.getLoc(), ret.getOperand(0), phase));
  }
  return counts;
}

/// Compare both sweeps against the input and splice the best qualifying
/// candidate into the original block, preserving its external wire uses.
static void optimizeSequence(Sequence &seq, const ConversionTarget *target) {
  if (!llvm::any_of(seq.gates, [](const Gate &g) {
        return !g.control &&
               isa<cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp>(
                   g.op);
      }))
    return;
  GateCounts original;
  for (const Gate &gate : seq.gates)
    countGate(original, gate.op);
  size_t bestCount = original.total;
  std::unique_ptr<Block> best;
  // Score each normalized candidate before changing the original block.
  for (const bool reverse : {false, true}) {
    auto block = materializeCandidate(seq, reverse);
    if (!block)
      return;
    // NOTE: Use the total gate count after flushing the frame and normalizing.
    // In the circuit below, the forward sweep records the middle X as pending
    // on the control wire. Moving it through the second CX makes an X pending
    // on each wire. The two identical CX gates are now adjacent and cancel.
    // At the end of the sequence the sweep emits the two pending X gates.
    // Counting only X gates would see an increase from one to two and miss
    // the reduction from three total gates to two.
    //
    //       ┌───┐        ┌───┐
    //  ──●──┤ X ├──●──   ┤ X ├
    //    │  └───┘  │   = └───┘
    //  ┌─┴─┐     ┌─┴─┐   ┌───┐
    //  ┤ X ├─────┤ X ├   ┤ X ├
    //  └───┘     └───┘   └───┘
    const auto candidate = legalizeAndCount(*block, target);
    // Select this sweep only if legalization succeeded and its total beats
    // bestCount (the input total, or the forward sweep's total if selected).
    // A tie keeps the current choice. Both the T count and the two-qubit
    // count must also be at most their respective counts in the original.
    if (candidate && candidate->total < bestCount &&
        candidate->tCount <= original.tCount &&
        candidate->twoQubit <= original.twoQubit) {
      bestCount = candidate->total;
      best = std::move(block);
    }
  }
  if (!best)
    return;
  // Bind the candidate's arguments first. If an entire lane cancelled, its
  // return operand then points directly to the original input wire.
  for (auto [arg, input] : llvm::zip_equal(best->getArguments(), seq.inputs))
    arg.replaceAllUsesWith(input);
  for (auto [output, replacement] :
       llvm::zip_equal(seq.outputs, best->back().getOperands()))
    output.replaceAllUsesWith(replacement);
  // Splice the gates and any phase correction, then erase the replaced
  // operations from users to definitions.
  best->back().erase();
  seq.gates.front().op->getBlock()->getOperations().splice(
      seq.gates.front().op->getIterator(), best->getOperations());
  for (const Gate &gate : llvm::reverse(seq.gates))
    gate.op->erase();
}

/// Run signed Pauli propagation on the straight-line sequences in a function.
struct SignedPauliPropagationPass
    : cudaq::opt::impl::SignedPauliPropagationBase<SignedPauliPropagationPass> {
  using SignedPauliPropagationBase::SignedPauliPropagationBase;

  void runOnOperation() override {
    if (cudaq::runtime::isQuantumOptimizationDisabled(getOperation()))
      return;
    const auto target =
        basis.empty() ? nullptr : cudaq::createBasisTarget(getContext(), basis);

    cudaq::opt::forEachNestedBlock(getOperation().getBody(), [&](Block &block) {
      Sequence seq;
      for (Operation &op : block) {
        if (seq.tryAppendGate(op) || seq.gates.empty())
          continue;
        // A later CX can connect this op's wire to the run. Stopping here
        // keeps each rewrite contiguous and avoids tracking which wires a
        // skipped operation touched.
        optimizeSequence(seq, target.get());
        seq = Sequence();
      }
      optimizeSequence(seq, target.get());
    });
  }
};
} // namespace
