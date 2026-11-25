/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Move Pauli gates through the Clifford+T gates left by rotation synthesis
// so they can cancel with Paulis on the same wires. Preserve the global phase.
// For example,
//
//  ┌───┐┌───┐┌───┐               ┌────┐
//  ┤ X ├┤ T ├┤ X ├ = exp(iπ/4) × ┤ T† ├
//  └───┘└───┘└───┘               └────┘
//
// Try propagation from both ends. Emit the remaining Paulis and normalize
// the Clifford+T gates in each candidate before comparing gate counts.

#include "PhaseUtilities.h"
#include "SingleQubitChain.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/OptimizationLevel.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
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
/// The Quake gates for which propagatePauliFrame() implements an exact frame
/// update.
enum class Kind { X, Y, Z, H, CX, S, T };

/// A gate in execution order. Lane indices identify qubits even as each
/// emitted gate produces a new SSA wire for its target and controls.
struct Gate {
  /// Selects the frame update and the Quake operation to emit.
  Kind kind;
  /// Index of the target in the sequence's wire arrays.
  unsigned target;
  /// Index of the control wire for CX. Empty for single-qubit gates.
  std::optional<unsigned> control;
  /// Source location carried into the replacement IR.
  Location loc;
  /// Whether to emit the adjoint of the named gate.
  bool adj;
  /// Quake's Hermitian trait identifies gates that are their own inverses.
  bool hermitian;

  /// Only S and T have distinct adjoints in this gate set.
  void takeAdjoint() {
    if (!hermitian)
      adj = !adj;
  }
};

/// A contiguous run of supported operations in one block. The lane order is
/// fixed by the first use of each wire and shared by both candidate sweeps.
struct Sequence {
  /// Original operations to erase if a candidate wins, in execution order.
  SmallVector<Operation *> operations;
  SmallVector<Value> inputs;  ///< First wire operand on each lane.
  SmallVector<Value> outputs; ///< Most recent wire result on each lane.
  /// Gates in the same order as operations, with wires resolved to lanes.
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
    auto [it, fresh] = lanes.try_emplace(input, inputs.size());
    unsigned index = it->second;
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
  /// False leaves the sequence unchanged so the caller can finish it before
  /// `op`.
  bool tryAppendGate(Operation &op) {
    auto kind = llvm::TypeSwitch<Operation *, std::optional<Kind>>(&op)
                    .Case<cudaq::quake::XOp>([](auto g) {
                      return g.getControls().empty() ? Kind::X : Kind::CX;
                    })
                    .Case<cudaq::quake::YOp>([](auto) { return Kind::Y; })
                    .Case<cudaq::quake::ZOp>([](auto) { return Kind::Z; })
                    .Case<cudaq::quake::HOp>([](auto) { return Kind::H; })
                    .Case<cudaq::quake::SOp>([](auto) { return Kind::S; })
                    .Case<cudaq::quake::TOp>([](auto) { return Kind::T; })
                    .Default(std::nullopt);
    // Only the gate kinds above have frame-update rules. End the sequence
    // here so pending Paulis are flushed before this operation.
    if (!kind)
      return false;
    auto gate = cast<cudaq::quake::OperatorInterface>(&op);
    auto flow = cudaq::quake::detail::getWireFlow(&op);
    // Each input must have a unique use and a corresponding wire result.
    // The CX update assumes one positive control.
    if (!flow || gate.getTargets().size() != 1 ||
        gate.getControls().size() != (*kind == Kind::CX ? 1u : 0u) ||
        llvm::is_contained(cudaq::quake::getControlPolarities(gate), true) ||
        !llvm::all_of(flow->inputs, [](Value v) { return v.hasOneUse(); }))
      return false;
    // Quake lists control wires before the target in both inputs and results.
    std::optional<unsigned> control;
    if (!gate.getControls().empty())
      control = advanceWireLane(flow->inputs.front(), flow->results.front());
    auto target = advanceWireLane(flow->inputs.back(), flow->results.back());
    gates.push_back({*kind, target, control, op.getLoc(), gate.isAdj(),
                     op.hasTrait<cudaq::Hermitian>()});
    operations.push_back(&op);
    return true;
  }
};

/// Take the adjoint of the entire sequence in place, reversing execution order.
static void takeSequenceAdjoint(SmallVectorImpl<Gate> &gates) {
  std::reverse(gates.begin(), gates.end());
  for (Gate &g : gates)
    g.takeAdjoint();
}

/// One binary symplectic tableau row (x | z) and its scalar phase represent
/// the pending product exp(i pi phase/4) prod_j X_j^x[j] Z_j^z[j].
struct PauliFrame {
  /// Start with the identity on every wire.
  explicit PauliFrame(unsigned width) : x(width), z(width) {}

  llvm::BitVector x;  ///< One packed X bit per wire.
  llvm::BitVector z;  ///< One packed Z bit per wire.
  unsigned phase = 0; ///< Exponent of exp(i pi/4), modulo eight.

  /// Multiply the entire row by exp(i pi exponent/4). Unsigned reduction
  /// modulo eight also handles negative exponents.
  void addPhase(int exponent) { phase = (phase + exponent) % 8; }
};

/// Sweep forward on `width` wires. After each input gate, the original prefix
/// equals the emitted gates followed in execution order by the pending frame.
/// In matrix order the frame is F = omega^phase prod_j X_j^x Z_j^z, where
/// omega = exp(i pi/4). For each gate G, solve G F = F' G' and emit G'.
/// The diagrams below give each rule in execution order. I is an idle wire.
///
/// @param gates Sequence to rewrite in place, including the final Paulis.
/// @param width Number of wire lanes in the sequence.
/// @return Remaining scalar exponent modulo eight after emitting the frame.
static unsigned propagatePauliFrame(SmallVectorImpl<Gate> &gates,
                                    unsigned width) {
  PauliFrame frame(width);
  SmallVector<Gate> result;
  for (Gate g : gates) {
    switch (g.kind) {
    case Kind::X:
    case Kind::Y:
    case Kind::Z:
      /*
       * Absorb any single-qubit Pauli gate G (X, Y, or Z) into the frame.
       * If P is pending on its target wire, the product in matrix order is
       * G P = s Q. The frame retains Q and the scalar phase for later emission.
       */
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
      // Left-multiply by the incoming Pauli. Y supplies i = omega^2.
      // Moving its Z factor (or an incoming Z) past a pending X supplies -1.
      if (g.kind == Kind::Y)
        frame.addPhase(2);
      if (g.kind != Kind::X) {
        if (frame.x.test(g.target))
          frame.addPhase(4);
        frame.z.flip(g.target);
      }
      if (g.kind != Kind::Z)
        frame.x.flip(g.target);
      continue;
    case Kind::H: {
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
      const bool x = frame.x.test(g.target);
      const bool z = frame.z.test(g.target);
      if (x && z)
        frame.addPhase(4);
      frame.x[g.target] = z;
      frame.z[g.target] = x;
      break;
    }
    case Kind::CX:
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
      if (frame.x.test(*g.control))
        frame.x.flip(g.target);
      if (frame.z.test(g.target))
        frame.z.flip(*g.control);
      break;
    case Kind::S:
    case Kind::T:
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
      if (frame.x.test(g.target)) {
        // For D = diag(1, omega^k), D X = omega^k X D†. The
        // exponent changes sign for an adjoint S or T.
        int exponent = g.kind == Kind::T ? 1 : 2;
        frame.addPhase(g.adj ? -exponent : exponent);
        g.adj = !g.adj;
      }
      break;
    }
    // Absorbing a Pauli can expose adjacent inverse gates on the same wires.
    if (!result.empty()) {
      const Gate &last = result.back();
      if (last.kind == g.kind && last.target == g.target &&
          last.control == g.control && (g.hermitian || last.adj != g.adj)) {
        result.pop_back();
        continue;
      }
    }
    result.push_back(g);
  }
  // Flush at most one Pauli per wire in the sequence's original lane order.
  for (unsigned lane = 0; lane < width; ++lane) {
    const bool x = frame.x.test(lane);
    const bool z = frame.z.test(lane);
    if (!x && !z)
      continue;
    // Y = iXZ, so emitting Y for XZ leaves a scalar of -i.
    if (x && z)
      frame.addPhase(-2);
    result.push_back({x ? (z ? Kind::Y : Kind::X) : Kind::Z, lane, std::nullopt,
                      gates.back().loc,
                      /*adj=*/false,
                      /*hermitian=*/true});
  }
  gates = std::move(result);
  return frame.phase;
}

/// Emit a gate and advance the caller's target and control wires to its SSA
/// results using Quake's operand and result ordering.
template <typename Op>
static void emitGate(OpBuilder &builder, Location loc, Value &target,
                     MutableArrayRef<Value> controls = {},
                     bool adjoint = false) {
  cudaq::quake::createAndThreadGate<Op>(
      builder, loc, adjoint ? builder.getUnitAttr() : UnitAttr{}, {}, controls,
      MutableArrayRef<Value>(target));
}

/// Build one candidate by propagating Paulis either forward (input to output)
/// or in reverse (output to input). The caller tries both directions because
/// they can expose different cancellations. Both candidates implement the
/// original circuit, including its global phase.
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
/// @return A caller-owned, detached block in the original execution direction,
/// with the Pauli frame fully emitted. Block argument i represents
/// `seq.inputs[i]` and operand i of its `func.return` replaces
/// `seq.outputs[i]` if the caller selects this candidate.
static std::unique_ptr<Block> materializeCandidate(const Sequence &seq,
                                                   bool reverse) {
  auto gates = seq.gates;
  // For a reverse sweep, reverse the gate order and invert each gate. The
  // forward routine then visits the original circuit from output to input.
  if (reverse)
    takeSequenceAdjoint(gates);
  unsigned phase = propagatePauliFrame(gates, seq.inputs.size());
  if (reverse) {
    // The sweep just optimized the inverse circuit. Invert the result so the
    // candidate implements the original circuit in its input-to-output order.
    takeSequenceAdjoint(gates);
    // Taking the adjoint also conjugates the scalar returned by the sweep.
    phase = (8 - phase) % 8;
  }
  // Build each candidate off to the side so cleanup can follow unique wire
  // uses without modifying the original. The terminator holds its live-outs.
  auto block = std::make_unique<Block>();
  OpBuilder b(seq.operations.front()->getContext());
  b.setInsertionPointToEnd(block.get());
  SmallVector<Value> wires;
  for (Value input : seq.inputs)
    wires.push_back(block->addArgument(input.getType(), input.getLoc()));
  for (const Gate &g : gates) {
    Value &q = wires[g.target];
    switch (g.kind) {
    case Kind::X:
      emitGate<cudaq::quake::XOp>(b, g.loc, q);
      break;
    case Kind::Y:
      emitGate<cudaq::quake::YOp>(b, g.loc, q);
      break;
    case Kind::Z:
      emitGate<cudaq::quake::ZOp>(b, g.loc, q);
      break;
    case Kind::H:
      emitGate<cudaq::quake::HOp>(b, g.loc, q);
      break;
    case Kind::CX:
      emitGate<cudaq::quake::XOp>(b, g.loc, q,
                                  MutableArrayRef<Value>(wires[*g.control]));
      break;
    case Kind::S:
      emitGate<cudaq::quake::SOp>(b, g.loc, q, {}, g.adj);
      break;
    case Kind::T:
      emitGate<cudaq::quake::TOp>(b, g.loc, q, {}, g.adj);
      break;
    }
  }
  Location loc = seq.operations.back()->getLoc();
  if (phase) {
    // A global phase multiplies the whole sequence. Anchor it on one live
    // wire and forward that wire through the correction.
    Value angle =
        cudaq::opt::factory::createF64Constant(loc, b, phase * M_PI_4);
    wires.front() =
        cudaq::opt::emitPhaseCorrection(b, loc, angle, {}, {}, wires.front())
            .anchor;
  }
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

/// Count gates in a materialized sequence. Return no count if phase lowering
/// is needed to determine how many physical gates a controlled phase emits.
static std::optional<GateCounts>
countGates(llvm::iterator_range<Block::iterator> operations) {
  GateCounts counts;
  for (Operation &op : operations) {
    auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op);
    if (!gate)
      continue;
    if (isa<cudaq::quake::PhaseOp>(op)) {
      // LowerPhase erases a global phase. A controlled phase needs gates
      // on its controls, so its gate count is only known after lowering.
      if (!gate.getControls().empty())
        return std::nullopt;
      continue;
    }
    ++counts.total;
    counts.tCount += isa<cudaq::quake::TOp>(op);
    counts.twoQubit +=
        gate.getControls().size() + gate.getTargets().size() == 2;
  }
  return counts;
}

/// Compare both sweeps against the input and splice the best qualifying
/// candidate into the original block, preserving its external wire uses.
static void optimizeSequence(Sequence &seq) {
  // An empty sequence or one without explicit Paulis has no frame to move.
  if (!llvm::any_of(seq.gates, [](const Gate &g) {
        return g.kind == Kind::X || g.kind == Kind::Y || g.kind == Kind::Z;
      }))
    return;
  auto original = countGates(
      llvm::make_range(seq.operations.front()->getIterator(),
                       std::next(seq.operations.back()->getIterator())));
  if (!original)
    return;
  size_t bestCount = original->total;
  std::unique_ptr<Block> best;
  // Build each sweep in a temporary block so we can flush its frame, run the
  // existing Clifford+T normalizer, and compare the gates we would emit.
  for (bool reverse : {false, true}) {
    auto block = materializeCandidate(seq, reverse);
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
    auto candidate = countGates(block->without_terminator());
    // Select this sweep only if countGates succeeded and its total beats
    // bestCount (the input total, or the forward sweep's total if selected).
    // A tie keeps the current choice. Both the T count and the two-qubit
    // count must also be at most their respective counts in the original.
    if (candidate && candidate->total < bestCount &&
        candidate->tCount <= original->tCount &&
        candidate->twoQubit <= original->twoQubit) {
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
  seq.operations.front()->getBlock()->getOperations().splice(
      seq.operations.front()->getIterator(), best->getOperations());
  for (auto *op : llvm::reverse(seq.operations))
    op->erase();
}

/// Run signed Pauli propagation on the straight-line sequences in a function.
struct SignedPauliPropagationPass
    : cudaq::opt::impl::SignedPauliPropagationBase<SignedPauliPropagationPass> {
  /// Visit nested blocks independently and flush the frame at each unsupported
  /// operation, including measurements, resets, and region boundaries.
  void runOnOperation() override {
    if (cudaq::runtime::isQuantumOptimizationDisabled(getOperation()))
      return;
    cudaq::opt::forEachNestedBlock(getOperation().getBody(), [](Block &block) {
      Sequence seq;
      for (Operation &op : block) {
        // Collect consecutive supported gates in circuit order.
        if (seq.tryAppendGate(op))
          continue;
        // This operation ends the run. Leave it in place, optimize any gates
        // collected before it, and resume collection at the next operation.
        if (!seq.operations.empty()) {
          optimizeSequence(seq);
          seq = Sequence();
        }
      }
      // The final run can reach the end of the block.
      optimizeSequence(seq);
    });
  }
};
} // namespace
