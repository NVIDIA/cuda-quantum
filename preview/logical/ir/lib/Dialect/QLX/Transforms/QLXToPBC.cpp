/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qlx/Dialect/QLX/Transforms/QLXToPBC.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"
#include "qlx/Dialect/QLX/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
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

} // namespace

LogicalResult qlx::lowerToPBC(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  ctx->getOrLoadDialect<arith::ArithDialect>();
  Type lqbit = LogicalQubitType::get(ctx);
  Type i1 = IntegerType::get(ctx, 1);

  // This first cut handles one program body at a time.
  LogicalResult result = success();
  module.walk([&](ProgramOp program) {
    if (failed(result))
      return;
    Block *block = &program.getBody().front();

    // Index qubits by preparation order; thread the index through every op so
    // result qubits inherit their operand's index.
    llvm::DenseMap<Value, unsigned> qidx;
    llvm::SmallVector<Value> initValue; // prepare result per qubit index
    llvm::SmallVector<ApplyOp> gates;
    struct Meas {
      MeasureOp op;
      unsigned qubit;
      Pauli basis;
    };
    llvm::SmallVector<Meas> measures;
    llvm::SmallVector<DiscardOp> oldDiscards;

    for (Operation &op : *block) {
      if (auto prep = dyn_cast<PrepareOp>(op)) {
        unsigned idx = initValue.size();
        qidx[prep.getResult()] = idx;
        initValue.push_back(prep.getResult());
        continue;
      }
      if (auto apply = dyn_cast<ApplyOp>(op)) {
        auto builtin = dyn_cast<BuiltinActionAttr>(apply.getActionAttr());
        if (!builtin) {
          result = apply.emitOpError("qlx-to-pbc expects builtin actions");
          return;
        }
        if (builtin.getValue() == BuiltinAction::pauli_rotation) {
          result = apply.emitOpError("qlx-to-pbc requires synthesized gates; "
                                     "run qlx-synthesize first");
          return;
        }
        if (builtin.getValue() == BuiltinAction::ccz) {
          result = apply.emitOpError("qlx-to-pbc does not yet lower ccz");
          return;
        }
        // Thread qubit indices operand->result positionally.
        for (auto [in, out] :
             llvm::zip(apply.getInputs(), apply.getResults())) {
          auto it = qidx.find(in);
          if (it != qidx.end())
            qidx[out] = it->second;
        }
        if (builtin.getValue() != BuiltinAction::idle)
          gates.push_back(apply);
        continue;
      }
      if (auto meas = dyn_cast<MeasureOp>(op)) {
        auto it = qidx.find(meas.getInput());
        if (it == qidx.end()) {
          result = meas.emitOpError("measured qubit has no tracked index");
          return;
        }
        measures.push_back({meas, it->second, meas.getBasis()});
        continue;
      }
      if (auto d = dyn_cast<DiscardOp>(op)) {
        oldDiscards.push_back(d);
        continue;
      }
      if (isa<ReturnOp, arith::ConstantOp>(op))
        continue; // dead angle constants from synthesis are harmless
      result = op.emitOpError("qlx-to-pbc cannot lower this op in a PBC body");
      return;
    }
    if (failed(result))
      return;

    unsigned n = initValue.size();
    Frame frame(n);

    // Seed one generator per measurement with its basis Pauli; these are
    // conjugated by the whole circuit into M_i = U^dagger B_i U.
    llvm::SmallVector<unsigned> measGen(measures.size());
    for (auto [i, m] : llvm::enumerate(measures)) {
      unsigned g = frame.add();
      bool x = false, z = false;
      basisBits(m.basis, x, z);
      frame.gens[g].x[m.qubit] = x;
      frame.gens[g].z[m.qubit] = z;
      measGen[i] = g;
    }

    // Reverse walk: commute Cliffords out; each T contributes a pi/4 column
    // seeded to Z on its qubit and conjugated by the gates preceding it.
    struct RotationGen {
      unsigned generator;
      ApplyOp source;
    };
    llvm::SmallVector<RotationGen> tGen; // in reverse-circuit order
    for (ApplyOp apply : llvm::reverse(gates)) {
      auto action = cast<BuiltinActionAttr>(apply.getActionAttr()).getValue();
      auto qOf = [&](unsigned k) { return qidx[apply.getInputs()[k]]; };
      switch (action) {
      // Walking in reverse, each gate acts by its inverse conjugation
      // (g^dagger P g), so measurements become C^dagger B C and T-columns
      // V^dagger Z V. Only S/S-dagger are not self-inverse, so they swap.
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
      case BuiltinAction::t:
      case BuiltinAction::tdg: {
        unsigned g = frame.add();
        frame.gens[g].z[qOf(0)] = true;
        frame.gens[g].sign = (action == BuiltinAction::tdg);
        tGen.push_back({g, apply});
        break;
      }
      default:
        result = apply.emitOpError("qlx-to-pbc: unsupported action");
        return;
      }
    }
    if (failed(result))
      return;

    // Rebuild the body: prepares stay; emit pi/4 rotations (forward order),
    // then Pauli-product measurements, then discard the qubits.
    OpBuilder b(block, block->begin());
    if (!initValue.empty())
      b.setInsertionPointAfterValue(initValue.back());
    Location loc = program.getLoc();

    llvm::SmallVector<Value> cur(initValue.begin(), initValue.end());

    for (RotationGen rotation : llvm::reverse(tGen)) {
      const Generator &col = frame.gens[rotation.generator];
      auto support = supportOf(col);
      if (support.empty())
        continue; // identity rotation: a global phase, drop it
      if (failed(requireRepresentableMask(rotation.source, support.size(),
                                          "Pauli-product rotation"))) {
        result = failure();
        return;
      }
      llvm::SmallVector<Value> operands;
      llvm::SmallVector<Type> results;
      for (unsigned q : support) {
        operands.push_back(cur[q]);
        results.push_back(lqbit);
      }
      auto angle =
          arith::ConstantOp::create(b, loc, b.getF64FloatAttr(M_PI / 4.0));
      operands.push_back(angle);
      auto op = ApplyOp::create(
          b, loc, TypeRange(results),
          BuiltinActionAttr::get(ctx, BuiltinAction::pauli_rotation), operands,
          pauliParams(b, support, col, /*withAngle=*/true));
      for (auto [k, q] : llvm::enumerate(support))
        cur[q] = op.getResult(k);
    }

    for (auto [i, m] : llvm::enumerate(measures)) {
      const Generator &mgen = frame.gens[measGen[i]];
      auto support = supportOf(mgen);
      if (failed(requireRepresentableMask(m.op, support.size(),
                                          "Pauli-product measurement"))) {
        result = failure();
        return;
      }
      llvm::SmallVector<Value> operands;
      llvm::SmallVector<Type> results;
      for (unsigned q : support) {
        operands.push_back(cur[q]);
        results.push_back(lqbit);
      }
      results.push_back(i1);
      auto op = InstrumentOp::create(
          b, loc, TypeRange(results),
          BuiltinInstrumentAttr::get(ctx, BuiltinInstrument::mpp), operands,
          pauliParams(b, support, mgen, /*withAngle=*/false));
      for (auto [k, q] : llvm::enumerate(support))
        cur[q] = op.getResult(k);
      m.op.getResult().replaceAllUsesWith(op.getResults().back());
    }

    // The mpp is nondestructive, so release the qubits explicitly.
    if (!cur.empty())
      DiscardOp::create(b, loc, cur, /*reason=*/StringAttr{});

    // Erase the old gate/measure/discard ops. They form a def-use chain
    // (prepare -> gates -> measure), so erase in reverse program order.
    llvm::SmallVector<Operation *> dead;
    for (DiscardOp d : oldDiscards)
      dead.push_back(d);
    for (Meas &m : measures)
      dead.push_back(m.op);
    for (ApplyOp g : gates)
      dead.push_back(g);
    llvm::sort(dead, [](Operation *a, Operation *b) {
      return b->isBeforeInBlock(a); // reverse program order
    });
    for (Operation *op : dead)
      op->erase();
  });
  return result;
}

namespace {

struct QLXToPBCPass : public qlx::impl::QLXToPBCBase<QLXToPBCPass> {
  void runOnOperation() override {
    if (failed(qlx::lowerToPBC(getOperation())))
      signalPassFailure();
  }
};

} // namespace
