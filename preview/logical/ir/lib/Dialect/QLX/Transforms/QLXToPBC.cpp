/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#include "qlx/Dialect/QLX/Transforms/QLXToPBC.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"
#include "qlx/Dialect/QLX/Transforms/Passes.h"
#include "qlx/Dialect/QLX/Transforms/QLXVerifyPBC.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include <cmath>
#include <limits>

namespace qlx {
#define GEN_PASS_DEF_QLXTOPBC
#include "qlx/Dialect/QLX/Transforms/Passes.h.inc"
} // namespace qlx

using namespace mlir;
using namespace qlx;

namespace {

/// One symplectic generator over `n` qubits: Pauli `(x,z)` bits per qubit plus
/// a sign. `(x,z)=(0,0)=I, (1,0)=X, (0,1)=Z, (1,1)=Y`.
struct Generator {
  llvm::SmallVector<bool> x, z;
  bool sign = false;
  explicit Generator(unsigned n) : x(n, false), z(n, false) {}
};

/// A stabilizer frame carried through the circuit under Clifford conjugation.
/// Each Clifford right-updates every generator via the standard Aaronson-
/// Gottesman rules (`P -> g P g^dagger`). Generators are added on demand: the
/// measurement operators up front, one T-stab column per T/T-dagger.
struct Frame {
  unsigned n;
  llvm::SmallVector<Generator> gens;
  explicit Frame(unsigned n) : n(n) {}

  unsigned add() {
    gens.emplace_back(n);
    return gens.size() - 1;
  }

  void h(unsigned q) {
    for (auto &g : gens) {
      g.sign ^= (g.x[q] && g.z[q]);
      std::swap(g.x[q], g.z[q]);
    }
  }
  void s(unsigned q) {
    for (auto &g : gens) {
      g.sign ^= (g.x[q] && g.z[q]);
      g.z[q] = g.z[q] != g.x[q];
    }
  }
  void sdg(unsigned q) {
    s(q);
    s(q);
    s(q);
  } // S^dagger = S^3 (S^4 = I)
  void px(unsigned q) {
    for (auto &g : gens)
      g.sign ^= g.z[q]; // X anticommutes with Z
  }
  void pz(unsigned q) {
    for (auto &g : gens)
      g.sign ^= g.x[q]; // Z anticommutes with X
  }
  void py(unsigned q) {
    for (auto &g : gens)
      g.sign ^= (g.x[q] != g.z[q]); // Y anticommutes with X and Z
  }
  void cx(unsigned c, unsigned t) {
    for (auto &g : gens) {
      // r ^= x_c & z_t & (x_t == z_c), computed before updating x_t / z_c.
      g.sign ^= (g.x[c] && g.z[t] && (g.x[t] == g.z[c]));
      g.x[t] = g.x[t] != g.x[c];
      g.z[c] = g.z[c] != g.z[t];
    }
  }
  void cz(unsigned c, unsigned t) {
    h(t);
    cx(c, t);
    h(t);
  }

  void seedCanonicalBasis() {
    for (unsigned q = 0; q < n; ++q) {
      unsigned xGen = add();
      gens[xGen].x[q] = true;
      unsigned zGen = add();
      gens[zGen].z[q] = true;
    }
  }

  bool isCanonicalBasis() const {
    if (gens.size() != 2 * n)
      return false;
    for (unsigned q = 0; q < n; ++q) {
      const Generator &xGen = gens[2 * q];
      const Generator &zGen = gens[2 * q + 1];
      if (xGen.sign || zGen.sign)
        return false;
      for (unsigned k = 0; k < n; ++k)
        if (xGen.x[k] != (k == q) || xGen.z[k] || zGen.x[k] ||
            zGen.z[k] != (k == q))
          return false;
    }
    return true;
  }
};

/// Map a QLX Pauli-basis enum (measurement basis) onto (x,z) bits.
static void basisBits(Pauli p, bool &x, bool &z) {
  switch (p) {
  case Pauli::X:
    x = true;
    z = false;
    break;
  case Pauli::Y:
    x = true;
    z = true;
    break;
  case Pauli::Z:
    x = false;
    z = true;
    break;
  }
}

/// Build the `{x_mask, z_mask, sign}` parameter dict over a fixed operand order
/// (bit k = the k-th operand), optionally stamping the exact pi/4 angle.
static DictionaryAttr pauliParams(OpBuilder &b,
                                  llvm::ArrayRef<unsigned> support,
                                  const Generator &g, bool withAngle) {
  MLIRContext *ctx = b.getContext();
  auto i64 = IntegerType::get(ctx, 64);
  uint64_t xm = 0, zm = 0;
  for (auto [bit, q] : llvm::enumerate(support)) {
    if (g.x[q])
      xm |= (uint64_t{1} << bit);
    if (g.z[q])
      zm |= (uint64_t{1} << bit);
  }
  llvm::SmallVector<NamedAttribute> fields;
  fields.emplace_back(b.getStringAttr("x_mask"),
                      IntegerAttr::get(i64, static_cast<int64_t>(xm)));
  fields.emplace_back(b.getStringAttr("z_mask"),
                      IntegerAttr::get(i64, static_cast<int64_t>(zm)));
  fields.emplace_back(b.getStringAttr("sign"),
                      IntegerAttr::get(i64, g.sign ? -1 : 1));
  if (withAngle) {
    fields.emplace_back(b.getStringAttr("angle_pi_numer"),
                        IntegerAttr::get(i64, 1));
    fields.emplace_back(b.getStringAttr("angle_pi_denom"),
                        IntegerAttr::get(i64, 4));
  }
  return DictionaryAttr::get(ctx, fields);
}

/// The canonical Pauli masks are nonnegative i64 attributes. Consequently,
/// only bits 0..62 are representable; bit 63 would print/read as a negative
/// mask and is outside the model contract. Check this before mask assembly so
/// every shift is both defined and semantically representable.
static LogicalResult requireRepresentableMask(Operation *source,
                                              size_t supportSize,
                                              llvm::StringRef kind) {
  constexpr size_t maxSupport = std::numeric_limits<int64_t>::digits;
  if (supportSize <= maxSupport)
    return success();
  return source->emitOpError()
         << "qlx-to-pbc cannot encode " << kind << " support of " << supportSize
         << " operands in nonnegative i64 Pauli masks; maximum is "
         << maxSupport;
}

/// Qubits in a generator's support, in increasing qubit-index order.
static llvm::SmallVector<unsigned> supportOf(const Generator &g) {
  llvm::SmallVector<unsigned> support;
  for (unsigned q = 0; q < g.x.size(); ++q)
    if (g.x[q] || g.z[q])
      support.push_back(q);
  return support;
}

/// Period materialization is a bounded compiler normalization, never an
/// invitation to unroll the complete source trip count. Small counts can be
/// represented as one count-one phase chunk. Large counts require residual
/// signed-Clifford closure within the same bound. A second operation budget
/// protects nested or unusually large bodies from multiplicative cloning.
static constexpr int64_t maxDirectRepeatPhases = 64;
static constexpr int64_t maxRepeatFramePeriod = 64;
static constexpr size_t maxPeriodicClonedOperations = 4096;

struct Measurement {
  MeasureOp op;
  unsigned qubit;
  Pauli basis;
  unsigned generator = 0;
};

struct Disposition {
  DiscardOp op;
  llvm::SmallVector<unsigned> qubits;
  StringAttr reason;
};

/// Analyze and rewrite one qlx.program. Qubit identities are global to the
/// program, while SSA values remain local to their containing repeat region.
/// This lets a lowered rotation keep its folded region structure without
/// losing the global Pauli axis computed by the stabilizer frame.
class ProgramLowering {
public:
  explicit ProgramLowering(ProgramOp program)
      : program(program), ctx(program.getContext()),
        lqbit(LogicalQubitType::get(ctx)), i1(IntegerType::get(ctx, 1)) {}

  LogicalResult run() {
    Block &body = program.getBody().front();
    resetAnalysis();
    llvm::DenseMap<unsigned, Value> currentOwner;
    if (failed(indexBlock(body, /*carried=*/nullptr, /*insideRepeat=*/false,
                          currentOwner)))
      return failure();
    FailureOr<bool> normalized = normalizePeriodicRepeats(body);
    if (failed(normalized))
      return failure();
    if (*normalized) {
      resetAnalysis();
      currentOwner.clear();
      if (failed(indexBlock(body, /*carried=*/nullptr, /*insideRepeat=*/false,
                            currentOwner)))
        return failure();
    }

    Frame frame(initValues.size());
    for (Measurement &measurement : measurements) {
      measurement.generator = frame.add();
      bool x = false, z = false;
      basisBits(measurement.basis, x, z);
      frame.gens[measurement.generator].x[measurement.qubit] = x;
      frame.gens[measurement.generator].z[measurement.qubit] = z;
    }
    if (failed(walkReverse(body, frame, /*recordRotations=*/true)))
      return failure();
    if (failed(validateGeneratedSupports(body, frame)))
      return failure();
    for (Measurement &measurement : measurements) {
      llvm::SmallVector<unsigned> support =
          supportOf(frame.gens[measurement.generator]);
      if (failed(requireRepresentableMask(measurement.op, support.size(),
                                          "Pauli-product measurement")))
        return failure();
    }

    activeFrame = &frame;
    llvm::SmallVector<Value> current(initValues.begin(), initValues.end());
    if (failed(rewriteBlock(body, current)))
      return failure();
    return emitTerminalMeasurements(body, current, frame);
  }

private:
  ProgramOp program;
  MLIRContext *ctx;
  Type lqbit;
  Type i1;
  llvm::DenseMap<Value, unsigned> qidx;
  llvm::SmallVector<Value> initValues;
  llvm::SmallVector<Measurement> measurements;
  llvm::SmallVector<Disposition> dispositions;
  llvm::DenseMap<Operation *, unsigned> rotationGenerator;
  llvm::DenseMap<Operation *, llvm::SmallVector<unsigned>> repeatScope;
  llvm::DenseMap<Operation *, bool> repeatIdentity;

  void resetAnalysis() {
    qidx.clear();
    initValues.clear();
    measurements.clear();
    dispositions.clear();
    rotationGenerator.clear();
    repeatScope.clear();
    repeatIdentity.clear();
    activeFrame = nullptr;
  }

  LogicalResult requireLogicalGateSignature(ApplyOp apply, unsigned arity) {
    if (apply.getInputs().size() != arity || apply.getResults().size() != arity)
      return apply.emitOpError()
             << "qlx-to-pbc requires this builtin action to have exactly "
             << arity
             << " logical-qubit inputs and results and no classical payloads";
    for (Value input : apply.getInputs())
      if (!isa<LogicalQubitType>(input.getType()))
        return apply.emitOpError()
               << "qlx-to-pbc requires this builtin action to have exactly "
               << arity
               << " logical-qubit inputs and results and no classical "
                  "payloads";
    for (Value result : apply.getResults())
      if (!isa<LogicalQubitType>(result.getType()))
        return apply.emitOpError()
               << "qlx-to-pbc requires this builtin action to have exactly "
               << arity
               << " logical-qubit inputs and results and no classical "
                  "payloads";
    return success();
  }

  LogicalResult validateAction(ApplyOp apply) {
    auto builtin = dyn_cast<BuiltinActionAttr>(apply.getActionAttr());
    if (!builtin)
      return apply.emitOpError("qlx-to-pbc expects builtin actions");
    if (builtin.getValue() != BuiltinAction::pauli_rotation &&
        apply.getParameters() && !apply.getParameters()->empty())
      return apply.emitOpError(
          "qlx-to-pbc fixed builtin actions do not accept parameter bindings");
    switch (builtin.getValue()) {
    case BuiltinAction::pauli_rotation:
      return apply.emitOpError(
          "qlx-to-pbc requires synthesized gates; run qlx-synthesize first");
    case BuiltinAction::ccz:
    case BuiltinAction::ccx:
      return apply.emitOpError("qlx-to-pbc does not yet lower ccz/ccx");
    case BuiltinAction::idle:
      return apply.emitOpError(
          "qlx-to-pbc cannot erase workload-bearing idle actions; PBC idle "
          "preservation is not yet supported");
    case BuiltinAction::h:
    case BuiltinAction::s:
    case BuiltinAction::sdg:
    case BuiltinAction::x:
    case BuiltinAction::y:
    case BuiltinAction::z:
      return requireLogicalGateSignature(apply, 1);
    case BuiltinAction::cx:
    case BuiltinAction::cz:
      return requireLogicalGateSignature(apply, 2);
    case BuiltinAction::t:
    case BuiltinAction::tdg:
      return requireLogicalGateSignature(apply, 1);
    default:
      return apply.emitOpError("qlx-to-pbc: unsupported action");
    }
  }

  LogicalResult requireTracked(Value value, Operation *owner,
                               const llvm::DenseSet<unsigned> *carried,
                               llvm::StringRef what) {
    if (!isa<LogicalQubitType>(value.getType()))
      return success();
    auto it = qidx.find(value);
    if (it == qidx.end())
      return owner->emitOpError() << what << " has no tracked qubit identity";
    if (carried && !carried->contains(it->second))
      return owner->emitOpError()
             << "qlx-to-pbc repeat bodies may use only explicitly carried "
                "logical qubits; captured qubit identity "
             << it->second << " is not a repeat carry";
    return success();
  }

  LogicalResult
  requireCurrentOwner(Value value, Operation *owner,
                      const llvm::DenseSet<unsigned> *carried,
                      const llvm::DenseMap<unsigned, Value> &currentOwner,
                      llvm::StringRef what) {
    if (!isa<LogicalQubitType>(value.getType()))
      return success();
    if (failed(requireTracked(value, owner, carried, what)))
      return failure();
    unsigned identity = qidx.lookup(value);
    auto current = currentOwner.find(identity);
    if (current == currentOwner.end() || current->second != value)
      return owner->emitOpError()
             << what
             << " must consume the current logical-qubit SSA owner for "
                "identity "
             << identity
             << "; stale values and hidden captures are unsupported";
    return success();
  }

  LogicalResult indexBlock(Block &block,
                           const llvm::DenseSet<unsigned> *carried,
                           bool insideRepeat,
                           llvm::DenseMap<unsigned, Value> &currentOwner,
                           bool collectScopes = true) {
    for (Operation &operation : block) {
      if (auto prepare = dyn_cast<PrepareOp>(operation)) {
        if (insideRepeat)
          return prepare.emitOpError(
              "qlx-to-pbc repeat bodies do not support local preparation");
        unsigned index = initValues.size();
        qidx[prepare.getResult()] = index;
        initValues.push_back(prepare.getResult());
        currentOwner[index] = prepare.getResult();
        continue;
      }
      if (auto apply = dyn_cast<ApplyOp>(operation)) {
        if (failed(validateAction(apply)))
          return failure();
        if (carried && collectScopes) {
          auto &scope = repeatScope[apply.getOperation()];
          scope.append(carried->begin(), carried->end());
        }
        llvm::DenseSet<unsigned> consumed;
        for (Value input : apply.getInputs()) {
          if (failed(requireCurrentOwner(input, apply, carried, currentOwner,
                                         "apply input")))
            return failure();
          if (isa<LogicalQubitType>(input.getType()) &&
              !consumed.insert(qidx.lookup(input)).second)
            return apply.emitOpError(
                "qlx-to-pbc requires distinct logical-qubit action operands");
        }
        for (auto [input, output] :
             llvm::zip(apply.getInputs(), apply.getResults())) {
          if (!isa<LogicalQubitType>(output.getType()))
            continue;
          auto it = qidx.find(input);
          if (it == qidx.end())
            return apply.emitOpError(
                "logical-qubit result has no positionally tracked input");
          unsigned identity = it->second;
          qidx[output] = identity;
          currentOwner[identity] = output;
        }
        continue;
      }
      if (auto repeat = dyn_cast<cflow::RepeatOp>(operation)) {
        if (repeat.getEventIdAttr())
          return repeat.emitOpError(
              "qlx-to-pbc does not accept physical schedule event_id on a "
              "P0 repeat");
        if (repeat.getInits().size() != repeat.getResults().size())
          return repeat.emitOpError(
              "qlx-to-pbc requires one result for every repeat carry");
        llvm::DenseSet<unsigned> bodyCarries;
        llvm::DenseMap<unsigned, Value> bodyCurrentOwner;
        Block &repeatBody = repeat.getBody().front();
        for (auto [init, argument, result] :
             llvm::zip(repeat.getInits(), repeatBody.getArguments(),
                       repeat.getResults())) {
          if (!isa<LogicalQubitType>(init.getType()) ||
              !isa<LogicalQubitType>(argument.getType()) ||
              !isa<LogicalQubitType>(result.getType()))
            return repeat.emitOpError(
                "qlx-to-pbc currently supports only logical-qubit repeat "
                "carries and results");
          if (failed(requireCurrentOwner(init, repeat, carried, currentOwner,
                                         "repeat init")))
            return failure();
          unsigned index = qidx.lookup(init);
          if (!bodyCarries.insert(index).second)
            return repeat.emitOpError(
                "qlx-to-pbc requires distinct logical-qubit repeat carries");
          qidx[argument] = index;
          qidx[result] = index;
          bodyCurrentOwner[index] = argument;
        }
        if (failed(indexBlock(repeatBody, &bodyCarries,
                              /*insideRepeat=*/true, bodyCurrentOwner,
                              collectScopes)))
          return failure();
        auto yield = cast<cflow::YieldOp>(repeatBody.getTerminator());
        if (yield.getOperands().size() != repeat.getInits().size())
          return repeat.emitOpError(
              "qlx-to-pbc requires every repeat carry to be yielded");
        for (auto [index, yielded] : llvm::enumerate(yield.getOperands())) {
          if (!isa<LogicalQubitType>(yielded.getType()))
            return repeat.emitOpError(
                "qlx-to-pbc currently supports only logical-qubit repeat "
                "yields");
          auto it = qidx.find(yielded);
          if (it == qidx.end() ||
              it->second != qidx.lookup(repeatBody.getArgument(index)) ||
              bodyCurrentOwner.lookup(it->second) != yielded)
            return repeat.emitOpError(
                "qlx-to-pbc requires position-preserving repeat yields of "
                "the current logical-qubit SSA owners");
          currentOwner[it->second] = repeat.getResult(index);
        }
        continue;
      }
      if (auto measure = dyn_cast<MeasureOp>(operation)) {
        if (insideRepeat)
          return measure.emitOpError(
              "qlx-to-pbc repeat bodies do not support measurement or "
              "feed-forward");
        if (failed(requireCurrentOwner(measure.getInput(), measure, carried,
                                       currentOwner, "measured qubit")))
          return failure();
        currentOwner.erase(qidx.lookup(measure.getInput()));
        measurements.push_back(
            {measure, qidx.lookup(measure.getInput()), measure.getBasis()});
        continue;
      }
      if (auto discard = dyn_cast<DiscardOp>(operation)) {
        if (insideRepeat)
          return discard.emitOpError(
              "qlx-to-pbc repeat bodies do not support local discard");
        Disposition disposition{discard, {}, discard.getReasonAttr()};
        for (Value input : discard.getInputs()) {
          if (failed(requireCurrentOwner(input, discard, carried, currentOwner,
                                         "discard input")))
            return failure();
          unsigned qubit = qidx.lookup(input);
          disposition.qubits.push_back(qubit);
          currentOwner.erase(qubit);
        }
        dispositions.push_back(std::move(disposition));
        continue;
      }
      if (isa<cflow::YieldOp>(operation)) {
        if (!insideRepeat)
          return operation.emitOpError(
              "qlx-to-pbc found cflow.yield outside a repeat body");
        continue;
      }
      if (auto ret = dyn_cast<ReturnOp>(operation)) {
        for (Value operand : ret.getOperands())
          if (isa<LogicalQubitType>(operand.getType()))
            return ret.emitOpError(
                "qlx-to-pbc does not support logical-qubit program returns; "
                "terminal logical owners must be measured or discarded");
        if (!currentOwner.empty())
          return ret.emitOpError(
              "qlx-to-pbc requires every source logical-qubit owner to be "
              "measured or explicitly discarded before return");
        continue;
      }
      if (isa<arith::ConstantOp>(operation))
        continue;
      if (insideRepeat)
        return operation.emitOpError(
            "qlx-to-pbc repeat bodies support only synthesized unitary "
            "actions, nested supported repeats, constants, and cflow.yield");
      return operation.emitOpError(
          "qlx-to-pbc cannot lower this op in a PBC body");
    }
    return success();
  }

  LogicalResult validateGeneratedSupports(Block &block, Frame &frame) {
    for (Operation &operation : block) {
      if (auto apply = dyn_cast<ApplyOp>(operation)) {
        auto found = rotationGenerator.find(apply.getOperation());
        if (found == rotationGenerator.end())
          continue;
        llvm::SmallVector<unsigned> support =
            supportOf(frame.gens[found->second]);
        if (failed(requireRepresentableMask(apply, support.size(),
                                            "Pauli-product rotation")))
          return failure();
        auto scoped = repeatScope.find(apply.getOperation());
        if (scoped == repeatScope.end())
          continue;
        for (unsigned qubit : support) {
          bool carried = false;
          for (unsigned candidate : scoped->second)
            carried |= (candidate == qubit);
          if (!carried)
            return apply.emitOpError()
                   << "qlx-to-pbc rotation support escapes the repeat carry "
                      "set at logical-qubit identity "
                   << qubit;
        }
        continue;
      }
      if (auto repeat = dyn_cast<cflow::RepeatOp>(operation))
        if (failed(validateGeneratedSupports(repeat.getBody().front(), frame)))
          return failure();
    }
    return success();
  }

  void applyClifford(Frame &frame, ApplyOp apply, BuiltinAction action) {
    auto qOf = [&](unsigned index) {
      return qidx.lookup(apply.getInputs()[index]);
    };
    switch (action) {
    case BuiltinAction::h:
      frame.h(qOf(0));
      break;
    case BuiltinAction::s:
      frame.sdg(qOf(0));
      break;
    case BuiltinAction::sdg:
      frame.s(qOf(0));
      break;
    case BuiltinAction::x:
      frame.px(qOf(0));
      break;
    case BuiltinAction::y:
      frame.py(qOf(0));
      break;
    case BuiltinAction::z:
      frame.pz(qOf(0));
      break;
    case BuiltinAction::cx:
      frame.cx(qOf(0), qOf(1));
      break;
    case BuiltinAction::cz:
      frame.cz(qOf(0), qOf(1));
      break;
    default:
      break;
    }
  }

  FailureOr<bool> hasIdentityResidual(cflow::RepeatOp repeat) {
    auto cached = repeatIdentity.find(repeat.getOperation());
    if (cached != repeatIdentity.end())
      return cached->second;
    Frame basis(initValues.size());
    basis.seedCanonicalBasis();
    if (failed(walkReverse(repeat.getBody().front(), basis,
                           /*recordRotations=*/false)))
      return failure();
    bool identity = basis.isCanonicalBasis();
    repeatIdentity[repeat.getOperation()] = identity;
    return identity;
  }

  FailureOr<int64_t> findResidualPeriod(cflow::RepeatOp repeat) {
    Frame basis(initValues.size());
    basis.seedCanonicalBasis();
    for (int64_t period = 1; period <= maxRepeatFramePeriod; ++period) {
      if (failed(walkReverse(repeat.getBody().front(), basis,
                             /*recordRotations=*/false)))
        return failure();
      if (basis.isCanonicalBasis())
        return period;
    }
    repeat.emitOpError()
        << "qlx-to-pbc residual Clifford period exceeds the bounded search "
           "limit of "
        << maxRepeatFramePeriod
        << "; use a smaller static count or a future recurrence "
           "representation";
    return failure();
  }

  size_t countBodyOperations(cflow::RepeatOp repeat) {
    size_t count = 0;
    Operation *phaseTerminator = repeat.getBody().front().getTerminator();
    repeat.getBody().walk([&](Operation *operation) {
      // The phase's outer yield is threaded rather than cloned. Nested region
      // terminators are cloned with their parent operation and count against
      // the materialization budget.
      if (operation != phaseTerminator)
        ++count;
    });
    return count;
  }

  FailureOr<cflow::RepeatOp>
  createPhaseChunk(OpBuilder &builder, cflow::RepeatOp source,
                   int64_t repeatCount, int64_t phaseCopies, ValueRange inits) {
    auto chunk = cflow::RepeatOp::create(
        builder, source.getLoc(), source.getResultTypes(),
        builder.getI64IntegerAttr(repeatCount), inits, StringAttr{});
    for (NamedAttribute attribute : source->getAttrs())
      if (attribute.getName().getValue() != "count" &&
          attribute.getName().getValue() != "event_id")
        chunk->setAttr(attribute.getName(), attribute.getValue());

    Block &sourceBody = source.getBody().front();
    llvm::SmallVector<Type> argumentTypes;
    llvm::SmallVector<Location> argumentLocations;
    for (BlockArgument argument : sourceBody.getArguments()) {
      argumentTypes.push_back(argument.getType());
      argumentLocations.push_back(argument.getLoc());
    }

    OpBuilder::InsertionGuard guard(builder);
    Block *chunkBody =
        builder.createBlock(&chunk.getBody(), chunk.getBody().end(),
                            argumentTypes, argumentLocations);
    builder.setInsertionPointToEnd(chunkBody);
    llvm::SmallVector<Value> current(chunkBody->getArguments().begin(),
                                     chunkBody->getArguments().end());
    auto sourceYield = cast<cflow::YieldOp>(sourceBody.getTerminator());
    for (int64_t phase = 0; phase < phaseCopies; ++phase) {
      IRMapping mapping;
      for (auto [argument, value] :
           llvm::zip(sourceBody.getArguments(), current))
        mapping.map(argument, value);
      for (Operation &operation : sourceBody.without_terminator())
        builder.clone(operation, mapping);
      current.clear();
      for (Value yielded : sourceYield.getOperands()) {
        Value mapped = mapping.lookupOrNull(yielded);
        if (!mapped) {
          source.emitOpError()
              << "qlx-to-pbc periodic normalization could not map a yielded "
                 "SSA owner in phase "
              << phase;
          return failure();
        }
        current.push_back(mapped);
      }
    }
    cflow::YieldOp::create(builder, sourceYield.getLoc(), current);
    return chunk;
  }

  /// Index only a newly cloned chunk. The initial preflight already validated
  /// the whole program; the final preflight rebuilds all analysis state once.
  LogicalResult indexNewChunk(cflow::RepeatOp chunk) {
    llvm::DenseSet<unsigned> carried;
    llvm::DenseMap<unsigned, Value> currentOwner;
    Block &body = chunk.getBody().front();
    for (auto [init, argument, result] :
         llvm::zip(chunk.getInits(), body.getArguments(), chunk.getResults())) {
      auto found = qidx.find(init);
      if (found == qidx.end())
        return chunk.emitOpError(
            "periodic normalization lost a repeat init identity");
      unsigned identity = found->second;
      if (!carried.insert(identity).second)
        return chunk.emitOpError(
            "periodic normalization produced duplicate repeat carries");
      qidx[argument] = identity;
      qidx[result] = identity;
      currentOwner[identity] = argument;
    }
    return indexBlock(body, &carried, /*insideRepeat=*/true, currentOwner,
                      /*collectScopes=*/false);
  }

  LogicalResult expandPeriodicRepeat(cflow::RepeatOp repeat) {
    int64_t sourceCount = repeat.getCount();
    int64_t chunkCount = 1;
    int64_t chunkPhases = sourceCount;
    int64_t remainderPhases = 0;
    if (sourceCount > maxDirectRepeatPhases) {
      FailureOr<int64_t> period = findResidualPeriod(repeat);
      if (failed(period))
        return failure();
      chunkPhases = *period;
      chunkCount = sourceCount / *period;
      remainderPhases = sourceCount % *period;
    }

    uint64_t materializedPhases =
        static_cast<uint64_t>(chunkPhases + remainderPhases);
    size_t bodyOperations = countBodyOperations(repeat);
    if (bodyOperations != 0 &&
        materializedPhases > maxPeriodicClonedOperations / bodyOperations) {
      return repeat.emitOpError()
             << "qlx-to-pbc periodic normalization would clone "
             << bodyOperations << " body operations across "
             << materializedPhases << " phases, exceeding the bounded "
             << maxPeriodicClonedOperations << "-operation budget";
    }

    OpBuilder builder(repeat);
    llvm::SmallVector<Value> current(repeat.getInits().begin(),
                                     repeat.getInits().end());
    FailureOr<cflow::RepeatOp> chunk =
        createPhaseChunk(builder, repeat, chunkCount, chunkPhases, current);
    if (failed(chunk))
      return failure();
    if (failed(indexNewChunk(*chunk)))
      return failure();
    current.assign(chunk->getResults().begin(), chunk->getResults().end());
    if (remainderPhases != 0) {
      FailureOr<cflow::RepeatOp> remainder = createPhaseChunk(
          builder, repeat, /*repeatCount=*/1, remainderPhases, current);
      if (failed(remainder))
        return failure();
      if (failed(indexNewChunk(*remainder)))
        return failure();
      current.assign(remainder->getResults().begin(),
                     remainder->getResults().end());
    }
    for (auto [result, replacement] : llvm::zip(repeat.getResults(), current))
      result.replaceAllUsesWith(replacement);
    repeat.getBody().walk([&](cflow::RepeatOp nested) {
      repeatIdentity.erase(nested.getOperation());
    });
    repeatIdentity.erase(repeat.getOperation());
    repeat.erase();
    return success();
  }

  FailureOr<bool> normalizePeriodicRepeats(Block &block) {
    llvm::SmallVector<cflow::RepeatOp> repeats;
    for (Operation &operation : block)
      if (auto repeat = dyn_cast<cflow::RepeatOp>(operation))
        repeats.push_back(repeat);

    bool changed = false;
    for (cflow::RepeatOp repeat : repeats) {
      FailureOr<bool> nested =
          normalizePeriodicRepeats(repeat.getBody().front());
      if (failed(nested))
        return failure();
      changed |= *nested;
      if (repeat.getCount() <= 1)
        continue;
      FailureOr<bool> identity = hasIdentityResidual(repeat);
      if (failed(identity))
        return failure();
      if (*identity)
        continue;
      if (failed(expandPeriodicRepeat(repeat)))
        return failure();
      changed = true;
    }
    return changed;
  }

  LogicalResult walkReverse(Block &block, Frame &frame, bool recordRotations) {
    for (Operation &operation : llvm::reverse(block)) {
      if (auto apply = dyn_cast<ApplyOp>(operation)) {
        auto action = cast<BuiltinActionAttr>(apply.getActionAttr()).getValue();
        if (action == BuiltinAction::t || action == BuiltinAction::tdg) {
          if (recordRotations) {
            unsigned generator = frame.add();
            frame.gens[generator].z[qidx.lookup(apply.getInputs().front())] =
                true;
            frame.gens[generator].sign = (action == BuiltinAction::tdg);
            rotationGenerator[apply.getOperation()] = generator;
          }
          continue;
        }
        applyClifford(frame, apply, action);
        continue;
      }
      if (auto repeat = dyn_cast<cflow::RepeatOp>(operation)) {
        if (repeat.getCount() == 0) {
          if (!recordRotations)
            continue;
          // A zero-count body has no effect on the surrounding frame, but its
          // syntactic T sites remain part of the canonical PBC template. Walk
          // it once to derive those rotation columns, then restore every
          // generator that existed before entering the dead region.
          llvm::SmallVector<Generator> outerGenerators = frame.gens;
          unsigned outerCount = outerGenerators.size();
          if (failed(walkReverse(repeat.getBody().front(), frame,
                                 /*recordRotations=*/true)))
            return failure();
          llvm::SmallVector<Generator> bodyGenerators(
              frame.gens.begin() + outerCount, frame.gens.end());
          frame.gens = std::move(outerGenerators);
          frame.gens.append(bodyGenerators.begin(), bodyGenerators.end());
          continue;
        }
        if (repeat.getCount() > 1) {
          FailureOr<bool> identity = hasIdentityResidual(repeat);
          if (failed(identity))
            return failure();
          if (!*identity)
            return repeat.emitOpError()
                   << "qlx-to-pbc periodic normalization left folded repeat "
                      "count "
                   << repeat.getCount()
                   << " with a non-identity residual Clifford frame; every "
                      "count-greater-than-one output chunk must close its "
                      "signed frame";
        }
        if (failed(
                walkReverse(repeat.getBody().front(), frame, recordRotations)))
          return failure();
      }
    }
    return success();
  }

  LogicalResult rewriteApply(ApplyOp apply,
                             llvm::SmallVectorImpl<Value> &current) {
    for (Value result : apply.getResults())
      if (!isa<LogicalQubitType>(result.getType()))
        return apply.emitOpError(
            "qlx-to-pbc internal error: a nonlogical action result escaped "
            "signature preflight");

    auto action = cast<BuiltinActionAttr>(apply.getActionAttr()).getValue();
    if (action == BuiltinAction::t || action == BuiltinAction::tdg) {
      auto found = rotationGenerator.find(apply.getOperation());
      if (found == rotationGenerator.end())
        return apply.emitOpError(
            "qlx-to-pbc internal error: missing repeat rotation summary");
      const Generator &column = activeFrame->gens[found->second];
      llvm::SmallVector<unsigned> support = supportOf(column);
      if (failed(requireRepresentableMask(apply, support.size(),
                                          "Pauli-product rotation")))
        return failure();
      llvm::SmallVector<Value> operands;
      llvm::SmallVector<Type> results;
      for (unsigned qubit : support) {
        if (!current[qubit])
          return apply.emitOpError()
                 << "qlx-to-pbc rotation support escapes the repeat carry "
                    "set at logical-qubit identity "
                 << qubit;
        operands.push_back(current[qubit]);
        results.push_back(lqbit);
      }
      OpBuilder builder(apply);
      auto angle = arith::ConstantOp::create(
          builder, apply.getLoc(), builder.getF64FloatAttr(M_PI / 4.0));
      operands.push_back(angle);
      auto rotation = ApplyOp::create(
          builder, apply.getLoc(), TypeRange(results),
          BuiltinActionAttr::get(ctx, BuiltinAction::pauli_rotation), operands,
          pauliParams(builder, support, column, /*withAngle=*/true));
      for (auto [index, qubit] : llvm::enumerate(support)) {
        current[qubit] = rotation.getResult(index);
        qidx[rotation.getResult(index)] = qubit;
      }
    }

    for (Value result : apply.getResults()) {
      if (!isa<LogicalQubitType>(result.getType()))
        continue;
      unsigned qubit = qidx.lookup(result);
      if (!current[qubit])
        return apply.emitOpError(
            "qlx-to-pbc lost the current SSA value for a logical qubit");
      result.replaceAllUsesWith(current[qubit]);
    }
    for (Value result : apply.getResults())
      if (!result.use_empty())
        return apply.emitOpError(
            "qlx-to-pbc internal error: an action result remains live before "
            "source erasure");
    apply.erase();
    return success();
  }

  LogicalResult rewriteBlock(Block &block,
                             llvm::SmallVectorImpl<Value> &current) {
    llvm::SmallVector<Operation *> original;
    for (Operation &operation : block)
      original.push_back(&operation);

    for (Operation *operation : original) {
      if (auto apply = dyn_cast<ApplyOp>(operation)) {
        if (failed(rewriteApply(apply, current)))
          return failure();
        continue;
      }
      auto repeat = dyn_cast<cflow::RepeatOp>(operation);
      if (!repeat)
        continue;

      llvm::SmallVector<Value> nested(initValues.size());
      Block &repeatBody = repeat.getBody().front();
      for (auto [index, argument] :
           llvm::enumerate(repeatBody.getArguments())) {
        unsigned qubit = qidx.lookup(argument);
        if (!current[qubit])
          return repeat.emitOpError(
              "qlx-to-pbc lost the current SSA value for a repeat init");
        repeat->setOperand(index, current[qubit]);
        nested[qubit] = argument;
      }
      if (failed(rewriteBlock(repeatBody, nested)))
        return failure();

      auto yield = cast<cflow::YieldOp>(repeatBody.getTerminator());
      for (auto [index, argument] :
           llvm::enumerate(repeatBody.getArguments())) {
        unsigned qubit = qidx.lookup(argument);
        if (!nested[qubit])
          return repeat.emitOpError("qlx-to-pbc lost a yielded repeat carry");
        yield->setOperand(index, nested[qubit]);
        current[qubit] = repeat.getResult(index);
      }
    }
    return success();
  }

  LogicalResult emitTerminalMeasurements(Block &block,
                                         llvm::SmallVectorImpl<Value> &current,
                                         Frame &frame) {
    Operation *terminator = block.getTerminator();
    OpBuilder builder(terminator);
    Location location = program.getLoc();

    for (Measurement &measurement : measurements) {
      const Generator &column = frame.gens[measurement.generator];
      llvm::SmallVector<unsigned> support = supportOf(column);
      if (failed(requireRepresentableMask(measurement.op, support.size(),
                                          "Pauli-product measurement")))
        return failure();
      llvm::SmallVector<Value> operands;
      llvm::SmallVector<Type> results;
      for (unsigned qubit : support) {
        operands.push_back(current[qubit]);
        results.push_back(lqbit);
      }
      results.push_back(i1);
      auto mpp = InstrumentOp::create(
          builder, location, TypeRange(results),
          BuiltinInstrumentAttr::get(ctx, BuiltinInstrument::mpp), operands,
          pauliParams(builder, support, column, /*withAngle=*/false));
      for (auto [index, qubit] : llvm::enumerate(support))
        current[qubit] = mpp.getResult(index);
      measurement.op.getResult().replaceAllUsesWith(mpp.getResults().back());
    }

    for (Disposition &disposition : dispositions) {
      llvm::SmallVector<Value> operands;
      operands.reserve(disposition.qubits.size());
      for (unsigned qubit : disposition.qubits) {
        if (!current[qubit])
          return disposition.op.emitOpError(
              "qlx-to-pbc lost a source discard owner during rewriting");
        operands.push_back(current[qubit]);
        current[qubit] = Value();
      }
      DiscardOp::create(builder, disposition.op.getLoc(), operands,
                        disposition.reason);
    }

    llvm::SmallVector<Value> terminalOwners;
    for (Value owner : current)
      if (owner)
        terminalOwners.push_back(owner);
    if (!terminalOwners.empty())
      DiscardOp::create(builder, location, terminalOwners,
                        /*reason=*/StringAttr{});

    for (Measurement &measurement : measurements)
      measurement.op.erase();
    for (Disposition &disposition : dispositions)
      disposition.op.erase();
    return success();
  }

  Frame *activeFrame = nullptr;
};

} // namespace

static LogicalResult lowerToPBCInPlace(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  ctx->getOrLoadDialect<cflow::CflowDialect>();
  LogicalResult result = success();
  module.walk([&](ProgramOp program) {
    if (failed(result))
      return;
    ProgramLowering lowering(program);
    result = lowering.run();
  });
  return result;
}

LogicalResult qlx::lowerToPBC(ModuleOp module) {
  if (failed(mlir::verify(module)))
    return failure();
  OwningOpRef<ModuleOp> candidate(module.clone());
  if (failed(lowerToPBCInPlace(*candidate)))
    return failure();
  if (failed(mlir::verify(*candidate)))
    return failure();
  std::string error;
  if (failed(qlx::verifyPBCForm(*candidate, error))) {
    candidate->emitError(error);
    return failure();
  }
  module.getBodyRegion().takeBody(candidate->getBodyRegion());
  return success();
}

namespace {

struct QLXToPBCPass : public qlx::impl::QLXToPBCBase<QLXToPBCPass> {
  void runOnOperation() override {
    if (failed(qlx::lowerToPBC(getOperation())))
      signalPassFailure();
  }
};

} // namespace
