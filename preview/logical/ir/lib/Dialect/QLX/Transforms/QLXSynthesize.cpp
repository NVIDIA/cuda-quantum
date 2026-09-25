/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#include "qlx/Dialect/QLX/Transforms/QLXSynthesize.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"
#include "qlx/Dialect/QLX/Transforms/Passes.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace qlx {
#define GEN_PASS_DEF_QLXSYNTHESIZEROTATIONS
#define GEN_PASS_DEF_QLXVERIFYCLIFFORDT
#include "qlx/Dialect/QLX/Transforms/Passes.h.inc"
} // namespace qlx

using namespace mlir;
using namespace qlx;

namespace {

/// Basis changes in chronological application order. They map the selected
/// Pauli factor to Z before the parity rotation and undo the map afterwards.
/// S-dagger is normalized to S^3 so legalization emits positive generators.
static llvm::ArrayRef<BuiltinAction> prefixFor(char basis) {
  static const BuiltinAction kX[] = {BuiltinAction::h};
  static const BuiltinAction kY[] = {BuiltinAction::s, BuiltinAction::s,
                                     BuiltinAction::s, BuiltinAction::h};
  switch (basis) {
  case 'X':
    return kX;
  case 'Y':
    return kY;
  default:
    return {};
  }
}

static llvm::ArrayRef<BuiltinAction> suffixFor(char basis) {
  static const BuiltinAction kX[] = {BuiltinAction::h};
  static const BuiltinAction kY[] = {BuiltinAction::h, BuiltinAction::s};
  switch (basis) {
  case 'X':
    return kX;
  case 'Y':
    return kY;
  default:
    return {};
  }
}

/// Append one native synthesis gate as a chronological word over positive
/// H/S/T generators. W is global phase and X = H S^2 H.
static void appendGate(cudaq::synth::Gate gate,
                       llvm::SmallVectorImpl<BuiltinAction> &out) {
  switch (gate) {
  case cudaq::synth::Gate::H:
    out.push_back(BuiltinAction::h);
    break;
  case cudaq::synth::Gate::S:
    out.push_back(BuiltinAction::s);
    break;
  case cudaq::synth::Gate::T:
    out.push_back(BuiltinAction::t);
    break;
  case cudaq::synth::Gate::X:
    out.append({BuiltinAction::h, BuiltinAction::s, BuiltinAction::s,
                BuiltinAction::h});
    break;
  case cudaq::synth::Gate::W:
    break;
  }
}

/// Emit one built-in action and return the new linear qubit values.
static SmallVector<Value> emitApply(OpBuilder &builder, Location location,
                                    ValueRange inputs, BuiltinAction action) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(inputs.size());
  for (Value input : inputs)
    resultTypes.push_back(input.getType());
  auto actionAttr = BuiltinActionAttr::get(builder.getContext(), action);
  auto result =
      ApplyOp::create(builder, location, TypeRange(resultTypes), actionAttr,
                      inputs, /*parameters=*/DictionaryAttr{});
  return SmallVector<Value>(result.getResults());
}

static Value emitUnary(OpBuilder &builder, Location location, Value input,
                       BuiltinAction action) {
  return emitApply(builder, location, ValueRange{input}, action).front();
}

static Value emitSequence(OpBuilder &builder, Location location, Value input,
                          llvm::ArrayRef<BuiltinAction> actions) {
  Value current = input;
  for (BuiltinAction action : actions)
    current = emitUnary(builder, location, current, action);
  return current;
}

static void emitCx(OpBuilder &builder, Location location, Value &control,
                   Value &target) {
  SmallVector<Value, 2> inputs = {control, target};
  SmallVector<Value> results =
      emitApply(builder, location, inputs, BuiltinAction::cx);
  control = results[0];
  target = results[1];
}

static LogicalResult replaceResults(ApplyOp operation,
                                    ValueRange replacements) {
  if (operation.getNumResults() != replacements.size())
    return operation.emitOpError(
        "legalization produced the wrong number of linear results");
  for (auto [result, replacement] :
       llvm::zip_equal(operation.getResults(), replacements))
    result.replaceAllUsesWith(replacement);
  operation.erase();
  return success();
}

static LogicalResult replaceUnary(ApplyOp operation,
                                  llvm::ArrayRef<BuiltinAction> actions) {
  if (operation.getInputs().size() != 1 || operation.getNumResults() != 1)
    return operation.emitOpError(
        "expected a unary logical action during Clifford+T legalization");
  OpBuilder builder(operation);
  Value result = emitSequence(builder, operation.getLoc(),
                              operation.getInputs().front(), actions);
  return replaceResults(operation, ValueRange{result});
}

static SmallVector<Value, 3> emitCCZPhasePolynomial(OpBuilder &builder,
                                                    Location location,
                                                    ValueRange inputs) {
  static const BuiltinAction kTdg[] = {BuiltinAction::s, BuiltinAction::s,
                                       BuiltinAction::s, BuiltinAction::t};
  SmallVector<Value, 3> qubits(inputs);
  for (Value &qubit : qubits)
    qubit = emitUnary(builder, location, qubit, BuiltinAction::t);

  auto inverseTOnParity = [&](unsigned control, unsigned target) {
    emitCx(builder, location, qubits[control], qubits[target]);
    qubits[target] = emitSequence(builder, location, qubits[target], kTdg);
    emitCx(builder, location, qubits[control], qubits[target]);
  };
  inverseTOnParity(0, 1);
  inverseTOnParity(0, 2);
  inverseTOnParity(1, 2);

  emitCx(builder, location, qubits[0], qubits[2]);
  emitCx(builder, location, qubits[1], qubits[2]);
  qubits[2] = emitUnary(builder, location, qubits[2], BuiltinAction::t);
  emitCx(builder, location, qubits[1], qubits[2]);
  emitCx(builder, location, qubits[0], qubits[2]);
  return qubits;
}

/// Normalize the finite built-in logical actions to positive H/S/T/CX. CCZ is
/// realized by the seven-term phase polynomial
///
///   4abc = a+b+c-(a^b)-(a^c)-(b^c)+(a^b^c).
///
/// The deliberately simple compute/phase/uncompute network uses ten CX gates;
/// later circuit optimization may reduce it without changing this legalizer.
static LogicalResult legalizeFiniteActions(ModuleOp module) {
  SmallVector<ApplyOp> operations;
  module.walk([&](ApplyOp operation) { operations.push_back(operation); });

  static const BuiltinAction kSdg[] = {BuiltinAction::s, BuiltinAction::s,
                                       BuiltinAction::s};
  static const BuiltinAction kX[] = {BuiltinAction::h, BuiltinAction::s,
                                     BuiltinAction::s, BuiltinAction::h};
  static const BuiltinAction kY[] = {BuiltinAction::h, BuiltinAction::s,
                                     BuiltinAction::s, BuiltinAction::h,
                                     BuiltinAction::s, BuiltinAction::s};
  static const BuiltinAction kZ[] = {BuiltinAction::s, BuiltinAction::s};
  static const BuiltinAction kTdg[] = {BuiltinAction::s, BuiltinAction::s,
                                       BuiltinAction::s, BuiltinAction::t};

  for (ApplyOp operation : operations) {
    auto builtin = dyn_cast<BuiltinActionAttr>(operation.getActionAttr());
    if (!builtin)
      continue;
    switch (builtin.getValue()) {
    case BuiltinAction::h:
    case BuiltinAction::s:
    case BuiltinAction::t:
    case BuiltinAction::cx:
    case BuiltinAction::idle:
    case BuiltinAction::pauli_rotation:
      continue;
    case BuiltinAction::sdg:
      if (failed(replaceUnary(operation, kSdg)))
        return failure();
      continue;
    case BuiltinAction::x:
      if (failed(replaceUnary(operation, kX)))
        return failure();
      continue;
    case BuiltinAction::y:
      // X followed by Z is Y up to an irrelevant global phase.
      if (failed(replaceUnary(operation, kY)))
        return failure();
      continue;
    case BuiltinAction::z:
      if (failed(replaceUnary(operation, kZ)))
        return failure();
      continue;
    case BuiltinAction::tdg:
      if (failed(replaceUnary(operation, kTdg)))
        return failure();
      continue;
    case BuiltinAction::cz: {
      if (operation.getInputs().size() != 2 || operation.getNumResults() != 2)
        return operation.emitOpError(
            "expected a binary CZ during Clifford+T legalization");
      OpBuilder builder(operation);
      Value left = operation.getInputs()[0];
      Value right = emitUnary(builder, operation.getLoc(),
                              operation.getInputs()[1], BuiltinAction::h);
      emitCx(builder, operation.getLoc(), left, right);
      right = emitUnary(builder, operation.getLoc(), right, BuiltinAction::h);
      SmallVector<Value, 2> replacements = {left, right};
      if (failed(replaceResults(operation, replacements)))
        return failure();
      continue;
    }
    case BuiltinAction::ccz: {
      if (operation.getInputs().size() != 3 || operation.getNumResults() != 3)
        return operation.emitOpError(
            "expected a ternary CCZ during Clifford+T legalization");
      OpBuilder builder(operation);
      auto qubits = emitCCZPhasePolynomial(builder, operation.getLoc(),
                                           operation.getInputs());
      if (failed(replaceResults(operation, qubits)))
        return failure();
      continue;
    }
    case BuiltinAction::ccx: {
      if (operation.getInputs().size() != 3 || operation.getNumResults() != 3)
        return operation.emitOpError(
            "expected a ternary CCX during Clifford+T legalization");
      OpBuilder builder(operation);
      SmallVector<Value, 3> inputs(operation.getInputs());
      inputs[2] =
          emitUnary(builder, operation.getLoc(), inputs[2], BuiltinAction::h);
      auto qubits = emitCCZPhasePolynomial(builder, operation.getLoc(), inputs);
      qubits[2] =
          emitUnary(builder, operation.getLoc(), qubits[2], BuiltinAction::h);
      if (failed(replaceResults(operation, qubits)))
        return failure();
      continue;
    }
    }
  }
  return success();
}

/// Emit a positive-generator exact word for Rz(k*pi/4), with k reduced mod 8.
static void exactQuarterWord(int64_t quarterTurns,
                             llvm::SmallVectorImpl<BuiltinAction> &out) {
  int64_t reduced = ((quarterTurns % 8) + 8) % 8;
  for (int64_t i = 0; i < reduced / 2; ++i)
    out.push_back(BuiltinAction::s);
  if (reduced & 1)
    out.push_back(BuiltinAction::t);
}

/// Return the exact quarter-turn count when exact rational-pi source metadata
/// proves that the angle lies on the pi/4 Clifford+T lattice.
static std::optional<int64_t> exactQuarterTurns(ApplyOp operation) {
  auto parameters = operation.getParameters();
  if (!parameters)
    return std::nullopt;
  auto numerator =
      dyn_cast_or_null<IntegerAttr>(parameters->get("angle_pi_numer"));
  auto denominator =
      dyn_cast_or_null<IntegerAttr>(parameters->get("angle_pi_denom"));
  if (!numerator || !denominator || denominator.getInt() == 0)
    return std::nullopt;
  __int128 scaled = static_cast<__int128>(numerator.getInt()) * 4;
  __int128 denom = denominator.getInt();
  if (denom < 0) {
    scaled = -scaled;
    denom = -denom;
  }
  if (scaled % denom != 0)
    return std::nullopt;
  __int128 turns = scaled / denom;
  if (turns < INT64_MIN || turns > INT64_MAX)
    return std::nullopt;
  return static_cast<int64_t>(turns);
}

/// For float-authored angles, using the nearest exact lattice point is valid
/// only when that word itself meets the authored precision. This avoids the
/// old fixed-tolerance snap, which could violate tight precision requests.
static bool approximateQuarterWord(double angle, double precision,
                                   llvm::SmallVectorImpl<BuiltinAction> &out) {
  double quarter = angle / (M_PI / 4.0);
  int64_t nearest = std::llround(quarter);
  double nearestAngle = static_cast<double>(nearest) * (M_PI / 4.0);
  double error = 2.0 * std::abs(std::sin((angle - nearestAngle) / 4.0));
  if (error > precision)
    return false;
  exactQuarterWord(nearest, out);
  return true;
}

/// Read the constant f64 angle feeding a pauli_rotation.
static FailureOr<double> constantAngle(Value angle) {
  if (auto constant = angle.getDefiningOp<arith::ConstantOp>())
    if (auto value = dyn_cast<FloatAttr>(constant.getValue()))
      return value.getValueAsDouble();
  return failure();
}

/// The validated +/-1 operator sign of a rotation.
static FailureOr<int64_t> rotationSign(ApplyOp operation) {
  auto parameters = operation.getParameters();
  auto attr = parameters
                  ? dyn_cast_or_null<IntegerAttr>(parameters->get("sign"))
                  : IntegerAttr{};
  int64_t sign = attr ? attr.getInt() : 1;
  if (sign != 1 && sign != -1)
    return operation.emitOpError("pauli rotation sign must be +1 or -1");
  return sign;
}

/// The sanctioned readers of a rotation's angle. Storage keeps the operator
/// sign in the canonical `sign` field with a nonnegative-magnitude angle
/// (uniform with mpp); these fold it back in so no caller re-derives the fold
/// and the sign can never be silently dropped (the original defect).
static FailureOr<double> signedAngle(ApplyOp operation) {
  auto magnitude = constantAngle(operation.getInputs().back());
  if (failed(magnitude))
    return failure();
  auto sign = rotationSign(operation);
  if (failed(sign))
    return failure();
  return static_cast<double>(*sign) * *magnitude;
}

static std::optional<int64_t> signedExactQuarterTurns(ApplyOp operation) {
  auto turns = exactQuarterTurns(operation);
  if (!turns)
    return std::nullopt;
  auto parameters = operation.getParameters();
  auto attr = parameters
                  ? dyn_cast_or_null<IntegerAttr>(parameters->get("sign"))
                  : IntegerAttr{};
  return (attr ? attr.getInt() : 1) * *turns;
}

/// Enforce the rotation normal form the sign accessors assume: the operator
/// sign is in the `sign` field and the angle is a nonnegative magnitude.
static LogicalResult verifyCanonicalRotation(ApplyOp operation, double angle) {
  if (angle < 0.0)
    return operation.emitOpError(
        "non-canonical rotation: the angle must be a nonnegative magnitude; "
        "the operator sign belongs in the sign field");
  if (auto parameters = operation.getParameters())
    if (auto numer =
            dyn_cast_or_null<IntegerAttr>(parameters->get("angle_pi_numer")))
      if (numer.getInt() < 0)
        return operation.emitOpError(
            "non-canonical rotation: angle_pi_numer must be nonnegative");
  return success();
}

static LogicalResult synthesizePauliRotations(ModuleOp module,
                                              double defaultPrecision) {
  SmallVector<ApplyOp> rotations;
  module.walk([&](ApplyOp operation) {
    auto builtin = dyn_cast<BuiltinActionAttr>(operation.getActionAttr());
    if (builtin && builtin.getValue() == BuiltinAction::pauli_rotation)
      rotations.push_back(operation);
  });

  for (ApplyOp operation : rotations) {
    if (operation.getInputs().empty())
      return operation.emitOpError("pauli rotation has no angle operand");
    auto angle = constantAngle(operation.getInputs().back());
    if (failed(angle))
      return operation.emitOpError(
          "Clifford+T synthesis requires a static angle; specialize the "
          "program parameter before synthesis");

    SmallVector<Value> qubits;
    for (Value input : operation.getInputs().drop_back())
      if (isa<LogicalQubitType>(input.getType()))
        qubits.push_back(input);
    if (qubits.empty() || qubits.size() > 63 ||
        operation.getNumResults() != qubits.size())
      return operation.emitOpError(
          "Clifford+T synthesis requires 1..63 logical Pauli operands");

    auto parameters = operation.getParameters();
    auto xAttr = parameters
                     ? dyn_cast_or_null<IntegerAttr>(parameters->get("x_mask"))
                     : IntegerAttr{};
    auto zAttr = parameters
                     ? dyn_cast_or_null<IntegerAttr>(parameters->get("z_mask"))
                     : IntegerAttr{};
    if (!xAttr || !zAttr)
      return operation.emitOpError(
          "pauli rotation is missing its x_mask/z_mask parameters");

    double precision = defaultPrecision;
    if (parameters)
      if (auto authored =
              dyn_cast_or_null<FloatAttr>(parameters->get("precision")))
        precision = authored.getValueAsDouble();
    if (!std::isfinite(precision) || precision <= 0.0 || precision >= 1.0)
      return operation.emitOpError(
          "rotation precision must be finite and in the open interval (0, 1)");

    if (failed(verifyCanonicalRotation(operation, *angle)))
      return failure();
    auto signedTheta = signedAngle(operation);
    if (failed(signedTheta))
      return failure();

    uint64_t xMask = static_cast<uint64_t>(xAttr.getInt());
    uint64_t zMask = static_cast<uint64_t>(zAttr.getInt());
    uint64_t validMask = qubits.size() == 64
                             ? ~uint64_t{0}
                             : ((uint64_t{1} << qubits.size()) - 1);
    if (((xMask | zMask) & ~validMask) != 0)
      return operation.emitOpError(
          "pauli masks reference operands outside the rotation");

    SmallVector<char> bases;
    bases.reserve(qubits.size());
    for (unsigned index = 0; index < qubits.size(); ++index) {
      bool x = xMask & (uint64_t{1} << index);
      bool z = zMask & (uint64_t{1} << index);
      if (!x && !z)
        return operation.emitOpError(
            "identity factors are not legal rotation operands");
      bases.push_back(x && z ? 'Y' : (x ? 'X' : 'Z'));
    }

    SmallVector<BuiltinAction> core;
    std::optional<int64_t> exactTurns = signedExactQuarterTurns(operation);
    if (exactTurns) {
      exactQuarterWord(*exactTurns, core);
    } else if (!approximateQuarterWord(*signedTheta, precision, core)) {
      cudaq::synth::Real epsilon(precision);
      cudaq::synth::ScopedDefaultPrecision workingPrecision(
          cudaq::synth::details::required_precision(epsilon));
      cudaq::synth::GridsynthOptions options;
      // Compilation is a replayable transformation. CUDA-Q's GridSynth
      // intentionally uses entropy by default, so pin its factoring stream
      // at this compiler boundary to make identical inputs produce identical
      // Clifford+T words regardless of earlier synthesis calls.
      options.seed = 0;
      auto circuit =
          cudaq::synth::gridsynth(cudaq::synth::Real(*signedTheta),
                                  cudaq::synth::Real(precision), options);
      if (failed(circuit))
        return operation.emitOpError(
            "GridSynth failed to synthesize the rotation");

      // Circuit stores the algebraic product U = g1*g2*...*gn. QLX applies
      // gates chronologically, whose matrix is gn*...*g2*g1, so emit the
      // stored sequence in reverse. This is observable for off-lattice words.
      for (auto iterator = circuit->rbegin(); iterator != circuit->rend();
           ++iterator)
        appendGate(*iterator, core);
    }

    OpBuilder builder(operation);
    for (unsigned index = 0; index < qubits.size(); ++index)
      qubits[index] = emitSequence(builder, operation.getLoc(), qubits[index],
                                   prefixFor(bases[index]));

    unsigned accumulator = qubits.size() - 1;
    for (unsigned index = 0; index < accumulator; ++index)
      emitCx(builder, operation.getLoc(), qubits[index], qubits[accumulator]);
    qubits[accumulator] =
        emitSequence(builder, operation.getLoc(), qubits[accumulator], core);
    for (unsigned index = accumulator; index-- > 0;)
      emitCx(builder, operation.getLoc(), qubits[index], qubits[accumulator]);

    for (unsigned index = 0; index < qubits.size(); ++index)
      qubits[index] = emitSequence(builder, operation.getLoc(), qubits[index],
                                   suffixFor(bases[index]));

    if (failed(replaceResults(operation, qubits)))
      return failure();
  }
  return success();
}

} // namespace

LogicalResult qlx::verifyCliffordT(ModuleOp module) {
  bool failedVerification = false;
  module.walk([&](ApplyOp operation) {
    auto builtin = dyn_cast<BuiltinActionAttr>(operation.getActionAttr());
    if (!builtin) {
      operation.emitOpError(
          "is not a built-in action legal in the Clifford+T gate set");
      failedVerification = true;
      return;
    }
    switch (builtin.getValue()) {
    case BuiltinAction::h:
    case BuiltinAction::s:
    case BuiltinAction::t:
    case BuiltinAction::cx:
    case BuiltinAction::idle:
      return;
    default:
      operation.emitOpError(
          "remains outside the Clifford+T gate set after legalization");
      failedVerification = true;
      return;
    }
  });
  return failure(failedVerification);
}

LogicalResult qlx::synthesizeRotations(ModuleOp module, double precision) {
  if (!std::isfinite(precision) || precision <= 0.0 || precision >= 1.0)
    return module.emitError(
        "default synthesis precision must be finite and in (0, 1)");
  MLIRContext *context = module.getContext();
  context->getOrLoadDialect<arith::ArithDialect>();

  if (failed(legalizeFiniteActions(module)))
    return failure();
  if (failed(synthesizePauliRotations(module, precision)))
    return failure();
  return success();
}

namespace {

struct QLXSynthesizeRotationsPass
    : public qlx::impl::QLXSynthesizeRotationsBase<QLXSynthesizeRotationsPass> {
  using QLXSynthesizeRotationsBase::QLXSynthesizeRotationsBase;
  void runOnOperation() override {
    if (failed(qlx::synthesizeRotations(getOperation(), precision)))
      signalPassFailure();
  }
};

struct QLXVerifyCliffordTPass
    : public qlx::impl::QLXVerifyCliffordTBase<QLXVerifyCliffordTPass> {
  void runOnOperation() override {
    if (failed(qlx::verifyCliffordT(getOperation())))
      signalPassFailure();
  }
};

} // namespace
