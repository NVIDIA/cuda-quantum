/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#include "qlx/Dialect/QLX/Transforms/QLXVerifyPBC.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"
#include "qlx/Dialect/QLX/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include <limits>

namespace qlx {
#define GEN_PASS_DEF_QLXVERIFYPBC
#include "qlx/Dialect/QLX/Transforms/Passes.h.inc"
} // namespace qlx

using namespace mlir;
using namespace qlx;

namespace {

/// A measured Pauli product keyed by global qubit index: qubit -> (x, z).
struct MeasuredPauli {
  llvm::DenseMap<unsigned, std::pair<bool, bool>> bits;
};

/// Do two Pauli products commute? They anticommute iff the symplectic inner
/// product over their shared support is odd.
static bool commutes(const MeasuredPauli &a, const MeasuredPauli &b) {
  unsigned parity = 0;
  for (auto &[q, xz] : a.bits) {
    auto it = b.bits.find(q);
    if (it == b.bits.end())
      continue; // identity on q in b -> commutes there
    auto [xa, za] = xz;
    auto [xb, zb] = it->second;
    parity ^= (xa && zb) ^ (za && xb);
  }
  return parity == 0;
}

/// Map a pauli_rotation / mpp op's operands to global qubit indices and read
/// its x_mask / z_mask into a MeasuredPauli.
static MeasuredPauli pauliOf(Operation *op, ValueRange qubits,
                             DictionaryAttr params,
                             const llvm::DenseMap<Value, unsigned> &qidx) {
  MeasuredPauli p;
  int64_t xm = 0, zm = 0;
  if (params) {
    if (auto x = dyn_cast_or_null<IntegerAttr>(params.get("x_mask")))
      xm = x.getInt();
    if (auto z = dyn_cast_or_null<IntegerAttr>(params.get("z_mask")))
      zm = z.getInt();
  }
  unsigned bit = 0;
  for (Value q : qubits) {
    if (!isa<LogicalQubitType>(q.getType()))
      continue;
    auto it = qidx.find(q);
    if (it != qidx.end())
      p.bits[it->second] = {(xm >> bit) & 1, (zm >> bit) & 1};
    ++bit;
  }
  return p;
}

static std::string locStr(Operation *op) {
  std::string s;
  llvm::raw_string_ostream os(s);
  op->getLoc().print(os);
  return s;
}

static LogicalResult fail(Operation *op, const llvm::Twine &why,
                          std::string &error) {
  error = (why + " (at " + locStr(op) + ")").str();
  return failure();
}

static bool isI64(IntegerAttr value) {
  return value && value.getType().isSignlessInteger(64);
}

static LogicalResult verifyPauliProduct(Operation *op, ValueRange operands,
                                        DictionaryAttr params,
                                        llvm::StringRef kind,
                                        size_t expectedParameterCount,
                                        std::string &error) {
  auto xMask =
      params ? dyn_cast_or_null<IntegerAttr>(params.get("x_mask")) : nullptr;
  auto zMask =
      params ? dyn_cast_or_null<IntegerAttr>(params.get("z_mask")) : nullptr;
  auto sign =
      params ? dyn_cast_or_null<IntegerAttr>(params.get("sign")) : nullptr;
  if (!isI64(xMask) || !isI64(zMask) || !isI64(sign))
    return fail(op,
                llvm::Twine("PBC ") + kind +
                    " requires i64 x_mask, z_mask, and sign parameters",
                error);
  if (params.size() != expectedParameterCount)
    return fail(op,
                llvm::Twine("PBC ") + kind +
                    " requires its exact canonical parameter set",
                error);
  if (xMask.getInt() < 0 || zMask.getInt() < 0)
    return fail(
        op, llvm::Twine("PBC ") + kind + " requires nonnegative Pauli masks",
        error);
  if (sign.getInt() != 1 && sign.getInt() != -1)
    return fail(op, llvm::Twine("PBC ") + kind + " requires sign = +/-1",
                error);

  size_t arity = 0;
  for (Value operand : operands)
    arity += isa<LogicalQubitType>(operand.getType());
  constexpr size_t maxArity = std::numeric_limits<int64_t>::digits;
  if (arity > maxArity)
    return fail(op,
                llvm::Twine("PBC ") + kind + " has " + llvm::Twine(arity) +
                    " logical operands, exceeding the 63-bit nonnegative "
                    "i64 Pauli-mask limit",
                error);

  uint64_t used = static_cast<uint64_t>(xMask.getInt()) |
                  static_cast<uint64_t>(zMask.getInt());
  uint64_t allowed =
      arity == maxArity
          ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
          : ((uint64_t{1} << arity) - 1);
  if (used & ~allowed)
    return fail(op,
                llvm::Twine("PBC ") + kind +
                    " Pauli masks exceed the logical operand arity",
                error);
  if (used == 0)
    return fail(op,
                llvm::Twine("PBC ") + kind +
                    " does not permit an identity-only Pauli product",
                error);
  if (used != allowed)
    return fail(op,
                llvm::Twine("PBC ") + kind +
                    " must omit identity-only operand positions",
                error);
  return success();
}

static LogicalResult verifyRotationSignature(ApplyOp apply,
                                             std::string &error) {
  size_t qubitInputs = 0;
  for (Value input : apply.getInputs())
    qubitInputs += isa<LogicalQubitType>(input.getType());
  if (qubitInputs == 0 || apply.getInputs().size() != qubitInputs + 1 ||
      !apply.getInputs().back().getType().isF64() ||
      apply.getResults().size() != qubitInputs)
    return fail(apply,
                "PBC rotation requires exactly N logical-qubit inputs and "
                "results plus one trailing f64 angle input",
                error);
  for (Value result : apply.getResults())
    if (!isa<LogicalQubitType>(result.getType()))
      return fail(apply,
                  "PBC rotation requires exactly N logical-qubit inputs and "
                  "results plus one trailing f64 angle input",
                  error);
  return success();
}

static LogicalResult verifyMppSignature(InstrumentOp mpp, std::string &error) {
  if (mpp.getInputs().empty() ||
      mpp.getResults().size() != mpp.getInputs().size() + 1 ||
      !mpp.getResults().back().getType().isInteger(1))
    return fail(mpp,
                "PBC mpp requires exactly N logical-qubit inputs, N "
                "logical-qubit results, and one trailing i1 outcome",
                error);
  for (Value input : mpp.getInputs())
    if (!isa<LogicalQubitType>(input.getType()))
      return fail(mpp,
                  "PBC mpp requires exactly N logical-qubit inputs, N "
                  "logical-qubit results, and one trailing i1 outcome",
                  error);
  for (Value result : mpp.getResults().drop_back())
    if (!isa<LogicalQubitType>(result.getType()))
      return fail(mpp,
                  "PBC mpp requires exactly N logical-qubit inputs, N "
                  "logical-qubit results, and one trailing i1 outcome",
                  error);
  return success();
}

static LogicalResult
requireCurrentOwner(Operation *op, Value value,
                    const llvm::DenseMap<Value, unsigned> &qidx,
                    const llvm::DenseMap<unsigned, Value> &currentOwner,
                    llvm::StringRef what, std::string &error) {
  auto tracked = qidx.find(value);
  if (tracked == qidx.end())
    return fail(op, llvm::Twine(what) + " has no tracked qubit identity",
                error);
  auto current = currentOwner.find(tracked->second);
  if (current == currentOwner.end() || current->second != value)
    return fail(op,
                llvm::Twine(what) +
                    " must consume the current logical-qubit SSA owner for "
                    "identity " +
                    llvm::Twine(tracked->second),
                error);
  return success();
}

/// Verify and thread one canonical PBC rotation. `qidx` contains exactly the
/// qubit identities visible in the containing region, so an uncarried capture
/// in a folded repeat fails here rather than becoming implicit loop state.
static LogicalResult
verifyRotation(ApplyOp apply, llvm::DenseMap<Value, unsigned> &qidx,
               llvm::DenseMap<unsigned, Value> &currentOwner,
               std::string &error) {
  auto builtin = dyn_cast<BuiltinActionAttr>(apply.getActionAttr());
  if (!builtin || builtin.getValue() != BuiltinAction::pauli_rotation)
    return fail(apply,
                "PBC form permits no Clifford/gate actions; found a "
                "non-pauli_rotation apply",
                error);

  if (failed(verifyRotationSignature(apply, error)))
    return failure();

  auto params = apply.getParameters();
  if (failed(verifyPauliProduct(apply, apply.getInputs(),
                                params.value_or(DictionaryAttr()), "rotation",
                                5, error)))
    return failure();
  auto num = params
                 ? dyn_cast_or_null<IntegerAttr>(params->get("angle_pi_numer"))
                 : nullptr;
  auto den = params
                 ? dyn_cast_or_null<IntegerAttr>(params->get("angle_pi_denom"))
                 : nullptr;
  if (!isI64(num) || !isI64(den) || num.getInt() != 1 || den.getInt() != 4)
    return fail(apply,
                "PBC rotation is not canonical signed pi/4 "
                "(need angle_pi_numer = 1, angle_pi_denom = 4, "
                "sign = +/-1)",
                error);
  auto constant = apply.getInputs().back().getDefiningOp<arith::ConstantOp>();
  auto value =
      constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
  constexpr double quarterPi = 0.785398163397448309615660845819875721;
  if (!value || value.getValueAsDouble() != quarterPi)
    return fail(apply,
                "PBC rotation numeric angle must equal the exact pi/4 "
                "metadata magnitude",
                error);

  llvm::DenseSet<unsigned> consumed;
  for (Value input : apply.getInputs().drop_back()) {
    if (failed(requireCurrentOwner(apply, input, qidx, currentOwner,
                                   "PBC rotation input", error)))
      return failure();
    unsigned identity = qidx.lookup(input);
    if (!consumed.insert(identity).second)
      return fail(apply,
                  "PBC rotation logical operands must have distinct owners",
                  error);
  }
  for (auto [input, output] :
       llvm::zip(apply.getInputs().drop_back(), apply.getResults())) {
    auto it = qidx.find(input);
    if (it == qidx.end())
      return fail(apply,
                  "PBC rotation result has no tracked logical-qubit input",
                  error);
    unsigned identity = it->second;
    qidx[output] = identity;
    currentOwner[identity] = output;
  }
  return success();
}

static LogicalResult verifyRepeat(cflow::RepeatOp repeat,
                                  llvm::DenseMap<Value, unsigned> &outerQidx,
                                  llvm::DenseMap<unsigned, Value> &outerCurrent,
                                  std::string &error) {
  if (repeat.getEventIdAttr())
    return fail(repeat,
                "PBC repeat cannot carry a physical schedule event_id in P0",
                error);
  Block &body = repeat.getBody().front();
  if (repeat.getInits().size() != repeat.getResults().size())
    return fail(repeat,
                "PBC repeat requires one result for every explicit carry",
                error);

  llvm::DenseMap<Value, unsigned> bodyQidx;
  llvm::DenseMap<unsigned, Value> bodyCurrent;
  llvm::DenseSet<unsigned> carried;
  for (auto [init, argument, result] :
       llvm::zip(repeat.getInits(), body.getArguments(), repeat.getResults())) {
    if (!isa<LogicalQubitType>(init.getType()) ||
        !isa<LogicalQubitType>(argument.getType()) ||
        !isa<LogicalQubitType>(result.getType()))
      return fail(repeat,
                  "PBC repeat permits only logical-qubit carries and results",
                  error);
    auto found = outerQidx.find(init);
    if (found == outerQidx.end())
      return fail(repeat, "PBC repeat init has no tracked qubit identity",
                  error);
    unsigned identity = found->second;
    if (failed(requireCurrentOwner(repeat, init, outerQidx, outerCurrent,
                                   "PBC repeat init", error)))
      return failure();
    if (!carried.insert(identity).second)
      return fail(repeat, "PBC repeat carries must be distinct", error);
    bodyQidx[argument] = identity;
    bodyCurrent[identity] = argument;
    outerQidx[result] = identity;
  }

  for (Operation &operation : body) {
    if (auto apply = dyn_cast<ApplyOp>(operation)) {
      if (failed(verifyRotation(apply, bodyQidx, bodyCurrent, error)))
        return failure();
      continue;
    }
    if (auto nested = dyn_cast<cflow::RepeatOp>(operation)) {
      if (failed(verifyRepeat(nested, bodyQidx, bodyCurrent, error)))
        return failure();
      continue;
    }
    if (isa<arith::ConstantOp>(operation))
      continue;
    if (auto yield = dyn_cast<cflow::YieldOp>(operation)) {
      if (yield.getOperands().size() != body.getNumArguments())
        return fail(repeat, "PBC repeat must yield every carried qubit", error);
      for (auto [index, yielded] : llvm::enumerate(yield.getOperands())) {
        auto found = bodyQidx.find(yielded);
        if (found == bodyQidx.end() ||
            found->second != bodyQidx.lookup(body.getArgument(index)) ||
            bodyCurrent.lookup(found->second) != yielded)
          return fail(repeat,
                      "PBC repeat yields must preserve carried-qubit "
                      "positions",
                      error);
      }
      for (auto [index, result] : llvm::enumerate(repeat.getResults()))
        outerCurrent[bodyQidx.lookup(body.getArgument(index))] = result;
      continue;
    }
    return fail(&operation,
                "PBC repeat bodies permit only canonical pi/4 rotations, "
                "nested PBC repeats, constants, and cflow.yield",
                error);
  }
  return success();
}

} // namespace

LogicalResult qlx::verifyPBCForm(ModuleOp module, std::string &error) {
  if (failed(mlir::verify(module))) {
    error = "PBC form requires structurally valid MLIR";
    return failure();
  }
  LogicalResult result = success();
  module.walk([&](ProgramOp program) {
    if (failed(result))
      return;
    Block *block = &program.getBody().front();

    // Thread a qubit index through the body so measured Paulis can be compared
    // across ops on a common qubit axis.
    llvm::DenseMap<Value, unsigned> qidx;
    llvm::DenseMap<unsigned, Value> currentOwner;
    unsigned nextQ = 0;
    bool sawMeasurement = false;
    llvm::SmallVector<MeasuredPauli> measured;

    auto failHere = [&](Operation *op, const llvm::Twine &why) {
      result = fail(op, why, error);
    };

    for (Operation &op : *block) {
      if (failed(result))
        return;

      if (auto prep = dyn_cast<PrepareOp>(op)) {
        if (sawMeasurement) {
          failHere(&op,
                   "PBC form permits no logical-qubit preparation after the "
                   "terminal measurement phase begins");
          return;
        }
        unsigned identity = nextQ++;
        qidx[prep.getResult()] = identity;
        currentOwner[identity] = prep.getResult();
        continue;
      }
      if (auto apply = dyn_cast<ApplyOp>(op)) {
        // (3) rotations must precede measurements.
        if (sawMeasurement) {
          failHere(&op, "PBC form requires all rotations before measurements; "
                        "found a rotation after a measurement");
          return;
        }
        if (failed(verifyRotation(apply, qidx, currentOwner, error))) {
          result = failure();
          return;
        }
        continue;
      }
      if (auto repeat = dyn_cast<cflow::RepeatOp>(op)) {
        if (sawMeasurement) {
          failHere(&op, "PBC form requires all rotations before measurements; "
                        "found a repeat after a measurement");
          return;
        }
        if (failed(verifyRepeat(repeat, qidx, currentOwner, error))) {
          result = failure();
          return;
        }
        continue;
      }
      if (auto instr = dyn_cast<InstrumentOp>(op)) {
        auto builtin =
            dyn_cast<BuiltinInstrumentAttr>(instr.getInstrumentAttr());
        if (!builtin || builtin.getValue() != BuiltinInstrument::mpp) {
          failHere(&op, "PBC form permits only mpp instruments");
          return;
        }
        if (failed(verifyMppSignature(instr, error))) {
          result = failure();
          return;
        }
        llvm::DenseSet<unsigned> consumed;
        for (Value input : instr.getInputs()) {
          if (failed(requireCurrentOwner(&op, input, qidx, currentOwner,
                                         "PBC mpp input", error))) {
            result = failure();
            return;
          }
          unsigned identity = qidx.lookup(input);
          if (!consumed.insert(identity).second) {
            failHere(&op, "PBC mpp logical operands must have distinct owners");
            return;
          }
        }
        if (failed(verifyPauliProduct(
                &op, instr.getInputs(),
                instr.getParameters().value_or(DictionaryAttr()), "mpp", 3,
                error))) {
          result = failure();
          return;
        }
        sawMeasurement = true;
        measured.push_back(
            pauliOf(&op, instr.getInputs(),
                    instr.getParameters().value_or(DictionaryAttr()), qidx));
        for (auto [in, out] :
             llvm::zip(instr.getInputs(), instr.getResults())) {
          if (isa<LogicalQubitType>(in.getType())) {
            auto it = qidx.find(in);
            if (it != qidx.end()) {
              unsigned identity = it->second;
              qidx[out] = identity;
              currentOwner[identity] = out;
            }
          }
        }
        continue;
      }
      if (auto discard = dyn_cast<DiscardOp>(op)) {
        for (Value input : discard.getInputs()) {
          if (!isa<LogicalQubitType>(input.getType()))
            continue;
          if (failed(requireCurrentOwner(&op, input, qidx, currentOwner,
                                         "PBC discard input", error))) {
            result = failure();
            return;
          }
          currentOwner.erase(qidx.lookup(input));
        }
        continue;
      }
      if (auto ret = dyn_cast<ReturnOp>(op)) {
        for (Value operand : ret.getOperands())
          if (isa<LogicalQubitType>(operand.getType())) {
            failHere(&op,
                     "PBC form does not permit logical-qubit program returns; "
                     "terminal logical owners must be measured or discarded");
            return;
          }
        if (!currentOwner.empty()) {
          failHere(&op, "PBC form requires every logical-qubit owner to be "
                        "measured and discarded before program return");
          return;
        }
        continue;
      }
      if (isa<arith::ConstantOp>(op))
        continue;

      failHere(&op,
               "PBC form contains an op that is not "
               "prepare/pauli_rotation/repeat/mpp/discard/return/constant");
      return;
    }
    if (failed(result))
      return;

    // (4) measured Pauli products must pairwise commute.
    for (unsigned i = 0; i < measured.size(); ++i)
      for (unsigned j = i + 1; j < measured.size(); ++j)
        if (!commutes(measured[i], measured[j])) {
          error = "PBC measured Pauli products do not pairwise commute "
                  "(not simultaneously measurable)";
          result = failure();
          return;
        }
  });
  return result;
}

namespace {

struct QLXVerifyPBCPass : public qlx::impl::QLXVerifyPBCBase<QLXVerifyPBCPass> {
  void runOnOperation() override {
    std::string error;
    if (failed(qlx::verifyPBCForm(getOperation(), error))) {
      getOperation()->emitError(error);
      signalPassFailure();
    }
  }
};

} // namespace
