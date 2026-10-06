/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#include "qlx/Dialect/QLX/Transforms/QLXVerifyPBC.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"
#include "qlx/Dialect/QLX/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

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

} // namespace

LogicalResult qlx::verifyPBCForm(ModuleOp module, std::string &error) {
  LogicalResult result = success();
  module.walk([&](ProgramOp program) {
    if (failed(result))
      return;
    Block *block = &program.getBody().front();

    // Thread a qubit index through the body so measured Paulis can be compared
    // across ops on a common qubit axis.
    llvm::DenseMap<Value, unsigned> qidx;
    unsigned nextQ = 0;
    bool sawMeasurement = false;
    llvm::SmallVector<MeasuredPauli> measured;

    auto fail = [&](Operation *op, const llvm::Twine &why) {
      error = (why + " (at " + locStr(op) + ")").str();
      result = failure();
    };

    for (Operation &op : *block) {
      if (failed(result))
        return;

      if (auto prep = dyn_cast<PrepareOp>(op)) {
        qidx[prep.getResult()] = nextQ++;
        continue;
      }
      if (auto apply = dyn_cast<ApplyOp>(op)) {
        auto builtin = dyn_cast<BuiltinActionAttr>(apply.getActionAttr());
        if (!builtin || builtin.getValue() != BuiltinAction::pauli_rotation) {
          fail(&op, "PBC form permits no Clifford/gate actions; found a "
                    "non-pauli_rotation apply");
          return;
        }
        // (2) The numeric angle is the nonnegative pi/4 magnitude. Its sign
        // is carried only by the canonical Pauli-product sign field.
        auto params = apply.getParameters();
        auto num =
            params
                ? dyn_cast_or_null<IntegerAttr>(params->get("angle_pi_numer"))
                : nullptr;
        auto den =
            params
                ? dyn_cast_or_null<IntegerAttr>(params->get("angle_pi_denom"))
                : nullptr;
        auto sign = params ? dyn_cast_or_null<IntegerAttr>(params->get("sign"))
                           : nullptr;
        if (!num || !den || num.getInt() != 1 || den.getInt() != 4 || !sign ||
            (sign.getInt() != 1 && sign.getInt() != -1)) {
          fail(&op, "PBC rotation is not canonical signed pi/4 "
                    "(need angle_pi_numer = 1, angle_pi_denom = 4, "
                    "sign = +/-1)");
          return;
        }
        if (apply.getInputs().empty()) {
          fail(&op, "PBC rotation has no numeric angle operand");
          return;
        }
        auto constant =
            apply.getInputs().back().getDefiningOp<arith::ConstantOp>();
        auto value =
            constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
        constexpr double quarterPi = 0.785398163397448309615660845819875721;
        if (!value || value.getValueAsDouble() != quarterPi) {
          fail(&op, "PBC rotation numeric angle must equal the exact pi/4 "
                    "metadata magnitude");
          return;
        }
        // (3) rotations must precede measurements.
        if (sawMeasurement) {
          fail(&op, "PBC form requires all rotations before measurements; "
                    "found a rotation after a measurement");
          return;
        }
        // Thread indices operand -> result.
        for (auto [in, out] :
             llvm::zip(apply.getInputs(), apply.getResults())) {
          auto it = qidx.find(in);
          if (it != qidx.end())
            qidx[out] = it->second;
        }
        continue;
      }
      if (auto instr = dyn_cast<InstrumentOp>(op)) {
        auto builtin =
            dyn_cast<BuiltinInstrumentAttr>(instr.getInstrumentAttr());
        if (!builtin || builtin.getValue() != BuiltinInstrument::mpp) {
          fail(&op, "PBC form permits only mpp instruments");
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
            if (it != qidx.end())
              qidx[out] = it->second;
          }
        }
        continue;
      }
      if (isa<DiscardOp, ReturnOp, arith::ConstantOp>(op))
        continue;

      fail(&op, "PBC form contains an op that is not "
                "prepare/pauli_rotation/mpp/discard/return/constant");
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
