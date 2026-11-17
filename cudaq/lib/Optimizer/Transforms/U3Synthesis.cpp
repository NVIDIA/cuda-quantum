/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Matchers.h"
#include <array>
#include <map>

using namespace mlir;

/// Apply the uncontrolled gate `OpTy` to `target` and return the qubit value
/// that follows it: a new wire in value semantics, `target` otherwise.
template <typename OpTy>
static Value emitGate(OpBuilder &b, Location loc, Value target, bool adj) {
  if (isa<cudaq::quake::WireType>(target.getType()))
    return OpTy::create(b, loc, TypeRange{target.getType()}, adj, ValueRange{},
                        ValueRange{}, ValueRange{target}, DenseBoolArrayAttr{})
        .getWires()[0];
  OpTy::create(b, loc, adj, ValueRange{}, ValueRange{}, ValueRange{target});
  return target;
}

static FailureOr<Value> emitNamedGate(OpBuilder &b, Location loc, Value target,
                                      StringRef name) {
  std::string gate = name.lower();
  if (gate == "h")
    return emitGate<cudaq::quake::HOp>(b, loc, target, false);
  if (gate == "s" || gate == "sdg")
    return emitGate<cudaq::quake::SOp>(b, loc, target, gate == "sdg");
  if (gate == "t" || gate == "tdg")
    return emitGate<cudaq::quake::TOp>(b, loc, target, gate == "tdg");
  if (gate == "x")
    return emitGate<cudaq::quake::XOp>(b, loc, target, false);
  if (gate == "y")
    return emitGate<cudaq::quake::YOp>(b, loc, target, false);
  if (gate == "z")
    return emitGate<cudaq::quake::ZOp>(b, loc, target, false);
  return failure();
}

namespace {

class U3SynthesisPass
    : public PassWrapper<U3SynthesisPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(U3SynthesisPass)

  U3SynthesisPass(double epsilon,
                  std::shared_ptr<const cudaq::config::U3Synthesizer::Fn> fn)
      : epsilon(epsilon), fn(std::move(fn)) {}

  StringRef getArgument() const override { return "u3-synthesis"; }
  StringRef getDescription() const override {
    return "Replace constant-angle u3 gates with user-synthesized words.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cudaq::quake::QuakeDialect>();
  }

  void runOnOperation() override {
    // The words already returned, keyed on the angles.
    std::map<std::array<double, 3>, std::vector<std::string>> words;
    SmallVector<cudaq::quake::U3Op> ops;
    getOperation().walk([&](cudaq::quake::U3Op op) { ops.push_back(op); });
    for (auto op : ops)
      if (failed(synthesize(op, words)))
        return signalPassFailure();
  }

private:
  LogicalResult
  synthesize(cudaq::quake::U3Op op,
             std::map<std::array<double, 3>, std::vector<std::string>> &words) {
    if (!op.getControls().empty() || op.isAdj())
      return op.emitError("u3-synthesis: controlled or adjoint u3");
    std::array<double, 3> angles;
    for (auto [i, param] : llvm::enumerate(op.getParameters())) {
      FloatAttr attr;
      if (!matchPattern(param, m_Constant<FloatAttr>(&attr)))
        return op.emitError("u3-synthesis: u3 angle is not a constant");
      angles[i] = attr.getValueAsDouble();
    }

    auto [it, inserted] = words.try_emplace(angles);
    if (inserted) {
      auto result = (*fn)(angles[0], angles[1], angles[2], epsilon);
      if (result.error)
        return op.emitError("u3-synthesis: synthesizer failed: ")
               << *result.error;
      it->second = std::move(result.gates);
    }

    OpBuilder builder(op);
    Value qubit = op.getTarget();
    for (const auto &gate : it->second) {
      auto next = emitNamedGate(builder, op.getLoc(), qubit, gate);
      if (failed(next))
        return op.emitError("u3-synthesis: unknown gate '") << gate << "'";
      qubit = *next;
    }
    if (op.getNumResults())
      op.getWires()[0].replaceAllUsesWith(qubit);
    op.erase();
    return success();
  }

  double epsilon;
  std::shared_ptr<const cudaq::config::U3Synthesizer::Fn> fn;
};

} // namespace

std::unique_ptr<Pass> cudaq::opt::createU3Synthesis(
    double epsilon,
    std::shared_ptr<const cudaq::config::U3Synthesizer::Fn> fn) {
  return std::make_unique<U3SynthesisPass>(epsilon, std::move(fn));
}
