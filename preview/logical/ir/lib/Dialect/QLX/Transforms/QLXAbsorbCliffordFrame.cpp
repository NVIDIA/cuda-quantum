/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#include "qlx/Dialect/QLX/Transforms/QLXAbsorbCliffordFrame.h"

#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"
#include "qlx/Dialect/QLX/Transforms/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace qlx {
#define GEN_PASS_DEF_QLXABSORBCLIFFORDFRAME
#define GEN_PASS_DEF_QLXVERIFYCLIFFORDFRAME
#include "qlx/Dialect/QLX/Transforms/Passes.h.inc"
} // namespace qlx

using namespace mlir;
using namespace qlx;

namespace {

/// `i^phase X^x Z^z` over one fixed global owner order.
struct Pauli {
  llvm::APInt x;
  llvm::APInt z;
  unsigned phase = 0;

  explicit Pauli(unsigned width, unsigned phase = 0)
      : x(width, 0), z(width, 0), phase(phase & 3U) {}

  bool operator==(const Pauli &other) const {
    return x == other.x && z == other.z && phase == other.phase;
  }
};

static Pauli multiply(const Pauli &left, const Pauli &right) {
  Pauli result(left.x.getBitWidth());
  result.x = left.x ^ right.x;
  result.z = left.z ^ right.z;
  result.phase =
      (left.phase + right.phase +
       2U * static_cast<unsigned>((left.z & right.x).popcount() & 1U)) &
      3U;
  return result;
}

static bool anticommutes(const Pauli &left, const Pauli &right) {
  return ((((left.x & right.z).popcount() + (left.z & right.x).popcount()) &
           1U) != 0U);
}

static std::optional<bool> hermitianSign(const Pauli &pauli) {
  unsigned canonical =
      static_cast<unsigned>((pauli.x & pauli.z).popcount()) & 3U;
  unsigned delta = (pauli.phase + 4U - canonical) & 3U;
  if (delta == 0U)
    return false;
  if (delta == 2U)
    return true;
  return std::nullopt;
}

static Pauli conjugateInverseQuarter(const Pauli &axis, Pauli value,
                                     int quarterTurns) {
  if (!anticommutes(axis, value))
    return value;
  int reduced = quarterTurns % 4;
  if (reduced == 0)
    return value;
  int direction = reduced > 0 ? 1 : -1;
  unsigned steps = static_cast<unsigned>(std::abs(reduced));
  for (unsigned step = 0; step < steps; ++step) {
    // Walking the authored circuit forward while retaining an inverse frame
    // requires D^dagger Q D. For D=exp(-i*pi*P/4), this is +i P Q;
    // the negative quarter turn uses -i P Q.
    value = multiply(axis, value);
    value.phase = (value.phase + (direction > 0 ? 1U : 3U)) & 3U;
  }
  return value;
}

class InverseFrame {
public:
  explicit InverseFrame(unsigned width) : width(width) {
    images.reserve(2U * width);
    for (unsigned q = 0; q < width; ++q) {
      Pauli x(width);
      x.x.setBit(q);
      images.push_back(std::move(x));
    }
    for (unsigned q = 0; q < width; ++q) {
      Pauli z(width);
      z.z.setBit(q);
      images.push_back(std::move(z));
    }
  }

  Pauli map(const Pauli &source) const {
    Pauli result(width, source.phase);
    for (unsigned q = 0; q < width; ++q)
      if (source.x[q])
        result = multiply(result, images[q]);
    for (unsigned q = 0; q < width; ++q)
      if (source.z[q])
        result = multiply(result, images[width + q]);
    return result;
  }

  void absorb(const Pauli &unsignedAxis, int quarterTurns) {
    int reduced = quarterTurns % 4;
    if (reduced == 0)
      return;
    Pauli mappedAxis = map(unsignedAxis);
    llvm::SmallVector<unsigned> affected;
    for (unsigned q = 0; q < width; ++q) {
      if (unsignedAxis.z[q])
        affected.push_back(q); // X_q anticommutes with Z/Y on q.
      if (unsignedAxis.x[q])
        affected.push_back(width + q); // Z_q anticommutes with X/Y.
    }
    for (unsigned index : affected)
      images[index] =
          conjugateInverseQuarter(mappedAxis, images[index], reduced);
  }

  bool operator==(const InverseFrame &other) const {
    return width == other.width && images == other.images;
  }

  unsigned size() const { return width; }

private:
  unsigned width;
  llvm::SmallVector<Pauli> images;
};

static llvm::SmallVector<Value> quantumValues(ValueRange values) {
  llvm::SmallVector<Value> result;
  for (Value value : values)
    if (isa<LogicalQubitType>(value.getType()))
      result.push_back(value);
  return result;
}

static LogicalResult readPauli(Operation *owner, ValueRange inputs,
                               DictionaryAttr parameters,
                               const llvm::DenseMap<Value, unsigned> &qidx,
                               unsigned width, Pauli &result,
                               bool includeSign = true) {
  if (!parameters)
    return owner->emitOpError("requires canonical Pauli parameters");
  auto xMask = parameters.getAs<IntegerAttr>("x_mask");
  auto zMask = parameters.getAs<IntegerAttr>("z_mask");
  auto sign = parameters.getAs<IntegerAttr>("sign");
  if (!xMask || !zMask || !sign)
    return owner->emitOpError("requires x_mask, z_mask, and sign parameters");
  if (sign.getValue().getBitWidth() > 64)
    return owner->emitOpError("Pauli sign is not representable as i64");
  int64_t signValue = sign.getValue().getSExtValue();
  if (signValue != 1 && signValue != -1)
    return owner->emitOpError("Pauli sign must be +1 or -1");

  auto quantum = quantumValues(inputs);
  llvm::APInt xm =
      xMask.getValue().zextOrTrunc(std::max<unsigned>(1U, quantum.size()));
  llvm::APInt zm =
      zMask.getValue().zextOrTrunc(std::max<unsigned>(1U, quantum.size()));
  if (xMask.getValue().getActiveBits() > quantum.size() ||
      zMask.getValue().getActiveBits() > quantum.size())
    return owner->emitOpError("Pauli mask exceeds its logical operand arity");

  result = Pauli(width);
  for (auto [position, value] : llvm::enumerate(quantum)) {
    auto found = qidx.find(value);
    if (found == qidx.end())
      return owner->emitOpError("Pauli operand has no tracked owner index");
    if (xm[position])
      result.x.setBit(found->second);
    if (zm[position])
      result.z.setBit(found->second);
  }
  result.phase = (static_cast<unsigned>((result.x & result.z).popcount()) +
                  (includeSign && signValue < 0 ? 2U : 0U)) &
                 3U;
  return success();
}

/// Return signed integer multiples of pi/2, or nullopt for a non-Clifford or
/// non-exact rotation. The product sign is folded into the returned angle.
static FailureOr<std::optional<int64_t>>
exactCliffordTurns(ApplyOp apply, bool verifyNumeric = true) {
  auto parameters = apply.getParameters();
  if (!parameters)
    return std::optional<int64_t>{};
  auto numerator = parameters->getAs<IntegerAttr>("angle_pi_numer");
  auto denominator = parameters->getAs<IntegerAttr>("angle_pi_denom");
  if (!numerator && !denominator)
    return std::optional<int64_t>{};
  if (!numerator || !denominator || numerator.getValue().getBitWidth() > 64 ||
      denominator.getValue().getBitWidth() > 64)
    return apply.emitOpError(
               "exact angle metadata must use representable integer fields"),
           failure();
  int64_t n = numerator.getValue().getSExtValue();
  int64_t d = denominator.getValue().getSExtValue();
  if (n < 0 || d <= 0)
    return apply.emitOpError(
               "exact angle metadata requires nonnegative numerator and "
               "positive denominator"),
           failure();
  if ((2 * n) % d != 0)
    return std::optional<int64_t>{};
  auto sign = parameters->getAs<IntegerAttr>("sign");
  if (!sign || sign.getValue().getBitWidth() > 64)
    return apply.emitOpError("exact Pauli rotation requires an i64 sign"),
           failure();
  int64_t signValue = sign.getValue().getSExtValue();
  if (signValue != 1 && signValue != -1)
    return apply.emitOpError("exact Pauli rotation sign must be +1 or -1"),
           failure();
  if (verifyNumeric) {
    auto inputs = apply.getInputs();
    auto constant = inputs.empty()
                        ? arith::ConstantOp{}
                        : inputs.back().getDefiningOp<arith::ConstantOp>();
    auto value =
        constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
    double expected = M_PI * static_cast<double>(n) / static_cast<double>(d);
    if (!value || value.getValueAsDouble() != expected)
      return apply.emitOpError(
                 "exact angle metadata disagrees with the numeric operand"),
             failure();
  }
  return std::optional<int64_t>{signValue * ((2 * n) / d)};
}

static DictionaryAttr transformedParameters(OpBuilder &builder,
                                            DictionaryAttr source,
                                            ArrayRef<unsigned> support,
                                            const Pauli &pauli) {
  unsigned maskWidth =
      std::max<unsigned>(64U, static_cast<unsigned>(support.size()) + 1U);
  llvm::APInt xMask(maskWidth, 0), zMask(maskWidth, 0);
  for (auto [position, q] : llvm::enumerate(support)) {
    if (pauli.x[q])
      xMask.setBit(position);
    if (pauli.z[q])
      zMask.setBit(position);
  }
  auto maskType = IntegerType::get(builder.getContext(), maskWidth);
  auto i64 = builder.getI64Type();
  llvm::SmallVector<NamedAttribute> fields;
  if (source)
    for (NamedAttribute field : source)
      if (field.getName() != "x_mask" && field.getName() != "z_mask" &&
          field.getName() != "sign")
        fields.push_back(field);
  fields.emplace_back(builder.getStringAttr("x_mask"),
                      IntegerAttr::get(maskType, xMask));
  fields.emplace_back(builder.getStringAttr("z_mask"),
                      IntegerAttr::get(maskType, zMask));
  bool negative = hermitianSign(pauli).value_or(false);
  fields.emplace_back(builder.getStringAttr("sign"),
                      IntegerAttr::get(i64, negative ? -1 : 1));
  return DictionaryAttr::get(builder.getContext(), fields);
}

static llvm::SmallVector<unsigned> supportOf(const Pauli &pauli) {
  llvm::SmallVector<unsigned> support;
  llvm::APInt mask = pauli.x | pauli.z;
  for (unsigned q = 0; q < mask.getBitWidth(); ++q)
    if (mask[q])
      support.push_back(q);
  return support;
}

static Pauli unsignedAxis(unsigned width, ArrayRef<unsigned> xs,
                          ArrayRef<unsigned> zs) {
  Pauli result(width);
  for (unsigned q : xs)
    result.x.setBit(q);
  for (unsigned q : zs)
    result.z.setBit(q);
  result.phase = static_cast<unsigned>((result.x & result.z).popcount()) & 3U;
  return result;
}

static LogicalResult absorbBuiltin(ApplyOp apply, BuiltinAction action,
                                   const llvm::DenseMap<Value, unsigned> &qidx,
                                   InverseFrame &frame) {
  auto inputs = quantumValues(apply.getInputs());
  auto q = [&](unsigned index) -> FailureOr<unsigned> {
    if (index >= inputs.size())
      return failure();
    auto found = qidx.find(inputs[index]);
    return found == qidx.end() ? FailureOr<unsigned>(failure())
                               : FailureOr<unsigned>(found->second);
  };
  auto one = q(0);
  if (failed(one))
    return apply.emitOpError("Clifford operand has no tracked owner index");
  unsigned width = frame.size();
  auto rotate = [&](Pauli axis, int turns) { frame.absorb(axis, turns); };
  auto rz = [&](unsigned target, int turns) {
    rotate(unsignedAxis(width, {}, {target}), turns);
  };
  auto rx = [&](unsigned target, int turns) {
    rotate(unsignedAxis(width, {target}, {}), turns);
  };
  auto ry = [&](unsigned target, int turns) {
    rotate(unsignedAxis(width, {target}, {target}), turns);
  };
  auto h = [&](unsigned target) {
    rz(target, 1);
    rx(target, 1);
    rz(target, 1);
  };
  auto cz = [&](unsigned left, unsigned right) {
    rz(left, 1);
    rz(right, 1);
    rotate(unsignedAxis(width, {}, {left, right}), -1);
  };

  switch (action) {
  case BuiltinAction::h:
    h(*one);
    return success();
  case BuiltinAction::s:
    rz(*one, 1);
    return success();
  case BuiltinAction::sdg:
    rz(*one, -1);
    return success();
  case BuiltinAction::x:
    rx(*one, 2);
    return success();
  case BuiltinAction::y:
    ry(*one, 2);
    return success();
  case BuiltinAction::z:
    rz(*one, 2);
    return success();
  case BuiltinAction::cz: {
    auto two = q(1);
    if (failed(two))
      return apply.emitOpError("CZ operand has no tracked owner index");
    cz(*one, *two);
    return success();
  }
  case BuiltinAction::cx: {
    auto two = q(1);
    if (failed(two))
      return apply.emitOpError("CX operand has no tracked owner index");
    h(*two);
    cz(*one, *two);
    h(*two);
    return success();
  }
  default:
    return apply.emitOpError(
        "unsupported built-in action in Clifford-frame normalization");
  }
}

struct IndexedBlock {
  llvm::DenseMap<Value, unsigned> qidx;
  llvm::SmallVector<Value> initial;
};

static FailureOr<IndexedBlock> indexBlock(Block &block, bool allowPrepare) {
  IndexedBlock indexed;
  for (BlockArgument argument : block.getArguments())
    if (isa<LogicalQubitType>(argument.getType())) {
      unsigned q = indexed.initial.size();
      indexed.qidx[argument] = q;
      indexed.initial.push_back(argument);
    }

  for (Operation &operation : block) {
    if (auto prepare = dyn_cast<PrepareOp>(operation)) {
      if (!allowPrepare)
        return prepare.emitOpError(
                   "folded Clifford-frame bodies cannot allocate local owners"),
               failure();
      unsigned q = indexed.initial.size();
      indexed.qidx[prepare.getResult()] = q;
      indexed.initial.push_back(prepare.getResult());
      continue;
    }
    if (auto apply = dyn_cast<ApplyOp>(operation)) {
      auto inputs = quantumValues(apply.getInputs());
      auto outputs = quantumValues(apply.getResults());
      if (inputs.size() != outputs.size())
        return apply.emitOpError("cannot index non-owner-preserving action"),
               failure();
      for (auto [input, output] : llvm::zip(inputs, outputs)) {
        auto found = indexed.qidx.find(input);
        if (found == indexed.qidx.end())
          return apply.emitOpError("input has no tracked owner index"),
                 failure();
        indexed.qidx[output] = found->second;
      }
      continue;
    }
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation)) {
      if (repeat.getInits().size() != repeat.getResults().size())
        return repeat.emitOpError("repeat carry/result arity differs"),
               failure();
      for (auto [input, output] :
           llvm::zip(repeat.getInits(), repeat.getResults())) {
        if (!isa<LogicalQubitType>(input.getType()))
          continue;
        auto found = indexed.qidx.find(input);
        if (found == indexed.qidx.end())
          return repeat.emitOpError("carry has no tracked owner index"),
                 failure();
        indexed.qidx[output] = found->second;
      }
    }
  }
  if (indexed.initial.empty())
    return emitError(block.getParentOp()->getLoc())
               << "Clifford-frame normalization requires logical owners",
           failure();
  return indexed;
}

static LogicalResult normalizeBlock(Block &block, InverseFrame &frame,
                                    bool requireClosedFrame);

static LogicalResult normalizeRepeat(qlx::cflow::RepeatOp repeat,
                                     const IndexedBlock &indexed,
                                     llvm::SmallVectorImpl<Value> &current,
                                     InverseFrame &frame) {
  auto quantumInits = quantumValues(repeat.getInits());
  if (quantumInits.size() != frame.size())
    return repeat.emitOpError(
        "Clifford-frame repeat must carry every live logical owner");
  for (auto [position, value] : llvm::enumerate(quantumInits)) {
    auto found = indexed.qidx.find(value);
    if (found == indexed.qidx.end() || found->second != position)
      return repeat.emitOpError(
          "Clifford-frame repeat carries must use canonical owner order");
  }

  InverseFrame bodyFrame = frame;
  if (failed(normalizeBlock(repeat.getBody().front(), bodyFrame,
                            /*requireClosedFrame=*/true)))
    return failure();
  if (!(bodyFrame == frame))
    return repeat.emitOpError(
        "folded repeat body does not close its Clifford frame");

  unsigned quantumPosition = 0;
  for (unsigned operand = 0; operand < repeat.getNumOperands(); ++operand) {
    Value value = repeat.getOperand(operand);
    if (!isa<LogicalQubitType>(value.getType()))
      continue;
    repeat->setOperand(operand, current[quantumPosition++]);
  }
  quantumPosition = 0;
  for (Value result : repeat.getResults())
    if (isa<LogicalQubitType>(result.getType()))
      current[quantumPosition++] = result;
  return success();
}

static LogicalResult normalizeBlock(Block &block, InverseFrame &frame,
                                    bool requireClosedFrame) {
  auto indexedOr = indexBlock(block, /*allowPrepare=*/!requireClosedFrame);
  if (failed(indexedOr))
    return failure();
  IndexedBlock indexed = std::move(*indexedOr);
  if (indexed.initial.size() != frame.size())
    return block.getParentOp()->emitOpError(
        "logical owner count does not match the incoming Clifford frame");
  llvm::SmallVector<Value> current(indexed.initial.begin(),
                                   indexed.initial.end());
  InverseFrame incoming = frame;
  llvm::SmallVector<Operation *> dead;
  llvm::SmallVector<unsigned> discarded;

  for (Operation &operation : llvm::make_early_inc_range(block)) {
    if (auto apply = dyn_cast<ApplyOp>(operation)) {
      auto builtin = dyn_cast<BuiltinActionAttr>(apply.getActionAttr());
      if (!builtin)
        return apply.emitOpError(
            "Clifford-frame normalization requires built-in actions");
      if (builtin.getValue() != BuiltinAction::pauli_rotation) {
        if (builtin.getValue() == BuiltinAction::t ||
            builtin.getValue() == BuiltinAction::tdg)
          return apply.emitOpError(
              "standard T/T-dagger must be represented as exact Pauli "
              "rotations before hybrid frame normalization");
        if (failed(
                absorbBuiltin(apply, builtin.getValue(), indexed.qidx, frame)))
          return failure();
        dead.push_back(apply);
        continue;
      }

      auto exact = exactCliffordTurns(apply);
      if (failed(exact))
        return failure();
      if (exact->has_value()) {
        Pauli axis(frame.size());
        if (failed(readPauli(apply, apply.getInputs(),
                             apply.getParameters().value_or(DictionaryAttr()),
                             indexed.qidx, frame.size(), axis,
                             /*includeSign=*/false)))
          return failure();
        frame.absorb(axis, static_cast<int>(**exact));
        dead.push_back(apply);
        continue;
      }

      Pauli source(frame.size());
      if (failed(readPauli(apply, apply.getInputs(),
                           apply.getParameters().value_or(DictionaryAttr()),
                           indexed.qidx, frame.size(), source)))
        return failure();
      Pauli transformed = frame.map(source);
      auto sign = hermitianSign(transformed);
      if (!sign)
        return apply.emitOpError(
            "Clifford frame transformed a Hermitian Pauli into a non-Hermitian "
            "operator");
      auto support = supportOf(transformed);
      if (support.empty())
        return apply.emitOpError(
            "non-Clifford identity rotation is only a global phase");
      OpBuilder builder(apply);
      llvm::SmallVector<Value> operands;
      llvm::SmallVector<Type> results;
      for (unsigned q : support) {
        operands.push_back(current[q]);
        results.push_back(LogicalQubitType::get(builder.getContext()));
      }
      operands.push_back(apply.getInputs().back());
      auto replacement = ApplyOp::create(
          builder, apply.getLoc(), TypeRange(results), apply.getActionAttr(),
          operands,
          transformedParameters(
              builder, apply.getParameters().value_or(DictionaryAttr()),
              support, transformed));
      for (auto [position, q] : llvm::enumerate(support))
        current[q] = replacement.getResult(position);
      dead.push_back(apply);
      continue;
    }

    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation)) {
      if (failed(normalizeRepeat(repeat, indexed, current, frame)))
        return failure();
      continue;
    }

    if (auto yield = dyn_cast<qlx::cflow::YieldOp>(operation)) {
      unsigned position = 0;
      for (unsigned operand = 0; operand < yield.getNumOperands(); ++operand) {
        Value value = yield.getOperand(operand);
        if (!isa<LogicalQubitType>(value.getType()))
          continue;
        if (position >= current.size())
          return yield.emitOpError("yield has too many logical owners");
        yield->setOperand(operand, current[position++]);
      }
      if (position != current.size())
        return yield.emitOpError("yield does not carry every logical owner");
      continue;
    }

    if (auto discard = dyn_cast<DiscardOp>(operation)) {
      unsigned position = 0;
      for (unsigned operand = 0; operand < discard.getNumOperands();
           ++operand) {
        Value value = discard.getOperand(operand);
        if (!isa<LogicalQubitType>(value.getType()))
          continue;
        auto found = indexed.qidx.find(value);
        if (found == indexed.qidx.end())
          return discard.emitOpError("discard has no tracked owner index");
        discard->setOperand(operand, current[found->second]);
        discarded.push_back(found->second);
        ++position;
      }
      continue;
    }

    if (isa<PrepareOp, ReturnOp, arith::ConstantOp>(operation))
      continue;
    return operation.emitOpError(
        "operation is outside hybrid Clifford-frame normalization scope");
  }

  if (requireClosedFrame && !(frame == incoming))
    return block.getParentOp()->emitOpError(
        "folded body leaves a non-identity Clifford-frame transition");

  if (!requireClosedFrame && !(frame == incoming)) {
    llvm::sort(discarded);
    discarded.erase(std::unique(discarded.begin(), discarded.end()),
                    discarded.end());
    if (discarded.size() != frame.size())
      return block.getParentOp()->emitOpError(
          "non-identity terminal Clifford frame escapes the closed program");
  }

  llvm::sort(dead, [](Operation *left, Operation *right) {
    return right->isBeforeInBlock(left);
  });
  for (Operation *operation : dead)
    operation->erase();
  return success();
}

static bool isCliffordBuiltin(BuiltinAction action) {
  switch (action) {
  case BuiltinAction::h:
  case BuiltinAction::s:
  case BuiltinAction::sdg:
  case BuiltinAction::x:
  case BuiltinAction::y:
  case BuiltinAction::z:
  case BuiltinAction::cx:
  case BuiltinAction::cz:
    return true;
  default:
    return false;
  }
}

static std::string locStr(Operation *operation) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  operation->getLoc().print(stream);
  return result;
}

} // namespace

LogicalResult qlx::absorbCliffordFrame(ModuleOp module) {
  module.getContext()->getOrLoadDialect<arith::ArithDialect>();
  module.getContext()->getOrLoadDialect<qlx::cflow::CflowDialect>();
  LogicalResult result = success();
  module.walk([&](ProgramOp program) {
    if (failed(result))
      return;
    auto indexed = indexBlock(program.getBody().front(), /*allowPrepare=*/true);
    if (failed(indexed)) {
      result = failure();
      return;
    }
    InverseFrame frame(indexed->initial.size());
    result = normalizeBlock(program.getBody().front(), frame,
                            /*requireClosedFrame=*/false);
  });
  return result;
}

LogicalResult qlx::verifyCliffordFrameForm(ModuleOp module,
                                           std::string &error) {
  LogicalResult result = success();
  module.walk([&](ProgramOp program) {
    if (failed(result))
      return;
    program.getBody().walk([&](Operation *operation) {
      if (failed(result))
        return;
      auto apply = dyn_cast<ApplyOp>(operation);
      if (!apply)
        return;
      auto builtin = dyn_cast<BuiltinActionAttr>(apply.getActionAttr());
      if (!builtin || isCliffordBuiltin(builtin.getValue()) ||
          builtin.getValue() != BuiltinAction::pauli_rotation) {
        error = "Clifford-frame form contains a non-rotation gate action (at " +
                locStr(operation) + ")";
        result = failure();
        return;
      }
      auto exact = exactCliffordTurns(apply);
      if (failed(exact)) {
        error = "Clifford-frame form contains malformed exact-angle metadata "
                "(at " +
                locStr(operation) + ")";
        result = failure();
        return;
      }
      if (exact->has_value()) {
        error = "Clifford-frame form retains an exact Clifford rotation (at " +
                locStr(operation) + ")";
        result = failure();
      }
    });
  });
  return result;
}

namespace {

struct QLXAbsorbCliffordFramePass
    : public qlx::impl::QLXAbsorbCliffordFrameBase<QLXAbsorbCliffordFramePass> {
  void runOnOperation() override {
    if (failed(qlx::absorbCliffordFrame(getOperation())))
      signalPassFailure();
  }
};

struct QLXVerifyCliffordFramePass
    : public qlx::impl::QLXVerifyCliffordFrameBase<QLXVerifyCliffordFramePass> {
  void runOnOperation() override {
    std::string error;
    if (failed(qlx::verifyCliffordFrameForm(getOperation(), error))) {
      getOperation()->emitError(error);
      signalPassFailure();
    }
  }
};

} // namespace
