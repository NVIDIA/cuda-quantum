/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include <unordered_set>

using namespace mlir;

#include "CanonicalPatterns.inc"

static LogicalResult verifyWireResultsAreLinear(Operation *op) {
  for (Value v : op->getOpResults())
    if (isa<quake::WireType>(v.getType())) {
      // Terminators can forward wire values, but they are not quantum
      // operations.
      if (v.hasOneUse() || v.use_empty())
        continue;
      // Allow a single cf.cond_br to use the value twice, once for each arm.
      std::unordered_set<Operation *> uniqs;
      for (auto *op : v.getUsers())
        uniqs.insert(op);
      if (uniqs.size() == 1 &&
          (*uniqs.begin())->hasTrait<OpTrait::IsTerminator>())
        continue;
      return op->emitOpError(
          "wires are a linear type and must have exactly one use");
    }
  return success();
}

/// When a quake operation is in value form, the number of wire arguments (wire
/// arity) must be the same as the number of wires returned as results (wire
/// coarity). This function verifies that this property is true.
LogicalResult quake::verifyWireArityAndCoarity(Operation *op) {
  std::size_t arity = 0;
  std::size_t coarity = 0;
  auto getCounts = [&](auto op) {
    for (auto arg : op.getTargets())
      if (isa<quake::WireType>(arg.getType()))
        ++arity;
    coarity = op.getWires().size();
  };
  if (auto gate = dyn_cast<OperatorInterface>(op)) {
    for (auto arg : gate.getControls())
      if (isa<quake::WireType>(arg.getType()))
        ++arity;
    getCounts(gate);
  } else if (auto meas = dyn_cast<MeasurementInterface>(op)) {
    getCounts(meas);
  }
  if (arity == coarity)
    return success();
  return op->emitOpError("arity does not equal coarity of wires");
}

bool quake::isSupportedMappingOperation(Operation *op) {
  return isa<OperatorInterface, MeasurementInterface, SinkOp, ReturnWireOp>(op);
}

ValueRange quake::getQuantumTypesFromRange(ValueRange range) {

  // Skip over classical types at the beginning
  int numClassical = 0;
  for (auto operand : range) {
    if (!isa<RefType, VeqType, WireType>(operand.getType()))
      numClassical++;
    else
      break;
  }

  ValueRange retVals = range.drop_front(numClassical);

  // Make sure all remaining operands are quantum
  for (auto operand : retVals)
    if (!isa<RefType, VeqType, WireType>(operand.getType()))
      return retVals.drop_front(retVals.size());

  return retVals;
}

ValueRange quake::getQuantumResults(Operation *op) {
  return getQuantumTypesFromRange(op->getResults());
}

ValueRange quake::getQuantumOperands(Operation *op) {
  return getQuantumTypesFromRange(op->getOperands());
}

LogicalResult quake::setQuantumOperands(Operation *op, ValueRange quantumVals) {
  ValueRange quantumOperands = getQuantumTypesFromRange(op->getOperands());

  if (quantumOperands.size() != quantumVals.size())
    return failure();

  // Count how many classical operands at beginning
  auto numClassical = op->getOperands().size() - quantumOperands.size();

  for (auto &&[i, quantumVal] : llvm::enumerate(quantumVals))
    op->setOperand(numClassical + i, quantumVal);

  return success();
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

Value quake::createConstantAlloca(PatternRewriter &builder, Location loc,
                                  OpResult result, ValueRange args) {
  auto newAlloca = [&]() {
    if (isa<quake::VeqType>(result.getType()) &&
        cast<quake::VeqType>(result.getType()).hasSpecifiedSize()) {
      return builder.create<quake::AllocaOp>(
          loc, cast<quake::VeqType>(result.getType()).getSize());
    }
    auto constOp = cast<arith::ConstantOp>(args[0].getDefiningOp());
    return builder.create<quake::AllocaOp>(
        loc, static_cast<std::size_t>(
                 cast<IntegerAttr>(constOp.getValue()).getInt()));
  }();
  return builder.create<quake::RelaxSizeOp>(
      loc, quake::VeqType::getUnsized(builder.getContext()), newAlloca);
}

LogicalResult quake::AllocaOp::verify() {
  // Result must be RefType or VeqType by construction.
  if (auto resTy = dyn_cast<VeqType>(getResult().getType())) {
    if (resTy.hasSpecifiedSize()) {
      if (getSize())
        return emitOpError("unexpected size operand");
    } else {
      if (auto size = getSize()) {
        if (auto cnt =
                dyn_cast_or_null<arith::ConstantOp>(size.getDefiningOp())) {
          std::int64_t argSize = cast<IntegerAttr>(cnt.getValue()).getInt();
          // TODO: This is a questionable check. We could have a very large
          // unsigned value that appears to be negative because of two's
          // complement. On the other hand, allocating 2^64 - 1 qubits isn't
          // going to go well.
          if (argSize < 0)
            return emitOpError("expected a non-negative integer size.");
        }
      } else {
        return emitOpError("size operand required");
      }
    }
  } else {
    // Size has no semantics for any type other than quake.veq.
    if (getSize())
      return emitOpError("cannot specify size with this quantum type");

    if (!quake::isConstantQuantumRefType(getResult().getType()))
      return emitOpError("struq type must have specified size");
  }

  // Check the uses. If any use is a InitializeStateOp, then it must be the only
  // use.
  Operation *self = getOperation();
  if (!self->getUsers().empty() && !self->hasOneUse())
    for (auto *op : self->getUsers())
      if (isa<quake::InitializeStateOp>(op))
        return emitOpError("init_state must be the only use");
  return success();
}

void quake::AllocaOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  // Use a canonicalization pattern as folding the constant into the veq type
  // changes the type. Uses may still expect a veq with unspecified size.
  // Folding is strictly reductive and doesn't allow the creation of ops.
  patterns.add<FuseConstantToAllocaPattern>(context);
}

quake::InitializeStateOp quake::AllocaOp::getInitializedState() {
  auto *self = getOperation();
  if (self->hasOneUse()) {
    auto x = self->getUsers().begin();
    return dyn_cast<quake::InitializeStateOp>(*x);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Apply
//===----------------------------------------------------------------------===//

void quake::ApplyOp::print(OpAsmPrinter &p) {
  if (getIsAdj())
    p << "<adj>";
  p << ' ';
  bool isDirect = getCallee().has_value();
  if (isDirect)
    p.printAttributeWithoutType(getCalleeAttr());
  else
    p << getIndirectCallee();
  p << ' ';
  if (!getControls().empty())
    p << '[' << getControls() << "] ";
  p << getArgs() << " : ";
  SmallVector<Type> operandTys{(*this)->getOperandTypes().begin(),
                               (*this)->getOperandTypes().end()};
  p.printFunctionalType(ArrayRef<Type>{operandTys}.drop_front(isDirect ? 0 : 1),
                        (*this)->getResultTypes());
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      {"operand_segment_sizes", "is_adj", getCalleeAttrNameStr()});
}

ParseResult quake::ApplyOp::parse(OpAsmParser &parser, OperationState &result) {
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseKeyword("adj") || parser.parseGreater())
      return failure();
    result.addAttribute("is_adj", parser.getBuilder().getUnitAttr());
  }
  SmallVector<OpAsmParser::UnresolvedOperand> calleeOperand;
  if (parser.parseOperandList(calleeOperand))
    return failure();
  bool isDirect = calleeOperand.empty();
  if (calleeOperand.size() > 1)
    return failure();
  if (isDirect) {
    NamedAttrList attrs;
    SymbolRefAttr funcAttr;
    if (parser.parseCustomAttributeWithFallback(
            funcAttr, parser.getBuilder().getType<NoneType>(),
            getCalleeAttrNameStr(), attrs))
      return failure();
    result.addAttribute(getCalleeAttrNameStr(), funcAttr);
  }

  SmallVector<OpAsmParser::UnresolvedOperand> controlOperands;
  if (succeeded(parser.parseOptionalLSquare()))
    if (parser.parseOperandList(controlOperands) || parser.parseRSquare())
      return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> miscOperands;
  if (parser.parseOperandList(miscOperands) || parser.parseColon())
    return failure();

  FunctionType applyTy;
  if (parser.parseType(applyTy) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(calleeOperand.size()),
                           static_cast<int32_t>(controlOperands.size()),
                           static_cast<int32_t>(miscOperands.size())}));
  result.addTypes(applyTy.getResults());
  if (isDirect) {
    if (parser.resolveOperands(
            llvm::concat<const OpAsmParser::UnresolvedOperand>(
                calleeOperand, controlOperands, miscOperands),
            applyTy.getInputs(), parser.getNameLoc(), result.operands))
      return failure();
  } else {
    auto loc = parser.getNameLoc();
    auto fnTy = parser.getBuilder().getFunctionType(
        applyTy.getInputs().drop_front(controlOperands.size()),
        applyTy.getResults());
    auto callableTy = cudaq::cc::CallableType::get(parser.getContext(), fnTy);
    if (parser.resolveOperands(calleeOperand, callableTy, loc,
                               result.operands) ||
        parser.resolveOperands(
            llvm::concat<const OpAsmParser::UnresolvedOperand>(controlOperands,
                                                               miscOperands),
            applyTy.getInputs(), loc, result.operands))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ApplyNoiseOp
//===----------------------------------------------------------------------===//

void quake::ApplyNoiseOp::print(OpAsmPrinter &p) {
  // noise_func or key
  p << ' ';
  if (auto fn = getNoiseFuncAttr())
    p << fn;
  else
    p << getKey();
  p << '(' << getParameters() << ") " << getQubits() << " : ";
  SmallVector<Type> operandTys{(*this)->getOperandTypes().begin(),
                               (*this)->getOperandTypes().end()};
  p.printFunctionalType(operandTys, (*this)->getResultTypes());
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"operand_segment_sizes", getNoiseFuncAttrName()});
}

ParseResult quake::ApplyNoiseOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> keyOperand;
  if (parser.parseOperandList(keyOperand))
    return failure();
  bool isDirect = keyOperand.empty();
  if (keyOperand.size() > 1)
    return failure();
  if (isDirect) {
    NamedAttrList attrs;
    SymbolRefAttr funcAttr;
    if (parser.parseCustomAttributeWithFallback(
            funcAttr, parser.getBuilder().getType<NoneType>(),
            getNoiseFuncAttrNameStr(), attrs))
      return failure();
    result.addAttribute(getNoiseFuncAttrNameStr(), funcAttr);
  }

  SmallVector<OpAsmParser::UnresolvedOperand> parameterOperands;
  if (succeeded(parser.parseOptionalLParen()))
    if (parser.parseOperandList(parameterOperands) || parser.parseRParen())
      return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> targetOperands;
  if (parser.parseOperandList(targetOperands) || parser.parseColon())
    return failure();

  FunctionType applyTy;
  if (parser.parseType(applyTy) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(keyOperand.size()),
                           static_cast<int32_t>(parameterOperands.size()),
                           static_cast<int32_t>(targetOperands.size())}));
  result.addTypes(applyTy.getResults());
  if (parser.resolveOperands(llvm::concat<const OpAsmParser::UnresolvedOperand>(
                                 keyOperand, parameterOperands, targetOperands),
                             applyTy.getInputs(), parser.getNameLoc(),
                             result.operands))
    return failure();
  return success();
}

LogicalResult quake::ApplyNoiseOp::verify() {
  // Must have either a noise_func or a key and not both.
  if (!getNoiseFuncAttr()) {
    if (!getKey())
      return emitOpError("must have a noise function or a key");
    if (getKey().getType() != IntegerType::get(getContext(), 64))
      return emitOpError("key must be i64");
  } else {
    if (getKey())
      return emitOpError("cannot have a noise function and a key");
  }

  // Parameters must be exactly one stdvec or 0 or more ptr<floating-point>.
  auto params = getParameters();
  if (params.size() == 1) {
    if (auto stdvecTy = dyn_cast<cudaq::cc::StdvecType>(params[0].getType())) {
      if (stdvecTy.getElementType() != Float64Type::get(getContext()))
        return emitOpError("must be std::vector<double>");
    } else if (auto ptrTy =
                   dyn_cast<cudaq::cc::PointerType>(params[0].getType())) {
      if (!isa<FloatType>(ptrTy.getElementType()))
        return emitOpError("must be floating-point");
    } else {
      return emitOpError("must be std::vector<double> or floating-point");
    }
  } else {
    for (auto p : params)
      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(p.getType()))
        if (!isa<FloatType>(ptrTy.getElementType()))
          return emitOpError("must be floating-point");
  }

  // Must have at least 1 qubit in qubits.
  if (getQubits().empty())
    return emitOpError("must have at least one qubit");
  return success();
}

//===----------------------------------------------------------------------===//
// BorrowWire
//===----------------------------------------------------------------------===//

LogicalResult quake::BorrowWireOp::verify() {
  std::int32_t id = getIdentity();
  if (id < 0)
    return emitOpError("id cannot be negative");
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  auto wires = module.lookupSymbol<quake::WireSetOp>(getSetName());
  if (!wires)
    return emitOpError("wire set could not be found");
  std::int32_t setCardinality = wires.getCardinality();
  if (id >= setCardinality)
    return emitOpError("id is out of bounds for wire set");
  return success();
}

//===----------------------------------------------------------------------===//
// Concat
//===----------------------------------------------------------------------===//

void quake::ConcatOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<ConcatSizePattern, ConcatNoOpPattern>(context);
}

//===----------------------------------------------------------------------===//
// ExpPauliRef
//===----------------------------------------------------------------------===//

static ParseResult
parseRawString(OpAsmParser &parser,
               std::optional<OpAsmParser::UnresolvedOperand> &value,
               StringAttr &rawString) {
  std::string stringVal;
  auto loc = UnknownLoc::get(parser.getContext());
  if (succeeded(parser.parseOptionalString(&stringVal))) {
    value = std::nullopt;
    rawString = StringAttr::get(parser.getContext(), stringVal);
    return success();
  }
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand))
    return emitError(loc, "must be an operand");
  value = operand;
  rawString = StringAttr{};
  return success();
}

template <typename OP>
void printRawString(OpAsmPrinter &printer, OP refOp, Value stringVal,
                    StringAttr rawString) {
  if (stringVal)
    printer.printOperand(stringVal);
  else if (rawString)
    printer << rawString;
}

void quake::ExpPauliOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add<BindExpPauliWord, AdjustAdjointExpPauliPattern>(context);
}

LogicalResult quake::ExpPauliOp::verify() {
  if (getPauliLiteralAttr()) {
    if (getPauli())
      return emitOpError("cannot have both a literal and a value Pauli word");
  } else {
    if (!getPauli())
      return emitOpError("must have either a literal or a value Pauli word");
  }
  if (!(getParameters().empty() || getParameters().size() == 1))
    return emitOpError("can only have 0 or 1 parameter");
  return verifyWireResultsAreLinear(getOperation());
}

//===----------------------------------------------------------------------===//
// ExtractRef
//===----------------------------------------------------------------------===//

static ParseResult
parseRawIndex(OpAsmParser &parser,
              std::optional<OpAsmParser::UnresolvedOperand> &index,
              IntegerAttr &rawIndex) {
  std::size_t constantIndex = quake::ExtractRefOp::kDynamicIndex;
  OptionalParseResult parsedInteger =
      parser.parseOptionalInteger(constantIndex);
  if (parsedInteger.has_value()) {
    if (failed(parsedInteger.value()))
      return failure();
    index = std::nullopt;
  } else {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand))
      return failure();
    index = operand;
  }
  auto i64Ty = IntegerType::get(parser.getContext(), 64);
  rawIndex = IntegerAttr::get(i64Ty, constantIndex);
  return success();
}

template <typename OP>
void printRawIndex(OpAsmPrinter &printer, OP refOp, Value index,
                   IntegerAttr rawIndex) {
  if (rawIndex.getValue() == OP::kDynamicIndex)
    printer.printOperand(index);
  else
    printer << rawIndex.getValue();
}

void quake::ExtractRefOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseConstantToExtractRefPattern, ForwardConcatExtractSingleton,
               ForwardConcatExtractPattern>(context);
}

LogicalResult quake::ExtractRefOp::verify() {
  if (getIndex()) {
    if (getRawIndex() != kDynamicIndex)
      return emitOpError(
          "must not have both a constant index and an index argument.");
  } else {
    if (getRawIndex() == kDynamicIndex) {
      return emitOpError("invalid constant index value");
    } else {
      auto veqSize = getVeq().getType().getSize();
      if (getVeq().getType().hasSpecifiedSize() && getRawIndex() >= veqSize)
        return emitOpError("invalid index [" + std::to_string(getRawIndex()) +
                           "] because >= size [" + std::to_string(veqSize) +
                           "]");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GetMemberOp
//===----------------------------------------------------------------------===//

LogicalResult quake::GetMemberOp::verify() {
  std::uint32_t index = getIndex();
  auto strTy = cast<quake::StruqType>(getStruq().getType());
  std::uint32_t size = strTy.getNumMembers();
  if (index >= size)
    return emitOpError("invalid index [" + std::to_string(index) +
                       "] because >= size [" + std::to_string(size) + "]");
  if (getType() != strTy.getMembers()[index])
    return emitOpError("result type does not match member " +
                       std::to_string(index) + " type");
  return success();
}

void quake::GetMemberOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<BypassMakeStruq>(context);
}

//===----------------------------------------------------------------------===//
// InitializeStateOp
//===----------------------------------------------------------------------===//

LogicalResult quake::InitializeStateOp::verify() {
  auto ptrTy = cast<cudaq::cc::PointerType>(getState().getType());
  Type ty = ptrTy.getElementType();
  if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(ty)) {
    if (!arrTy.isUnknownSize()) {
      std::size_t size = arrTy.getSize();
      if (!std::has_single_bit(size))
        return emitOpError(
            "initialize state vector must be power of 2, but is " +
            std::to_string(size) + " instead.");
    }
    if (!isa<FloatType, ComplexType>(arrTy.getElementType()))
      return emitOpError("invalid data pointer type");
  } else if (!isa<FloatType, ComplexType, quake::StateType>(ty)) {
    return emitOpError("invalid data pointer type");
  }
  return success();
}

void quake::InitializeStateOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ForwardAllocaTypePattern>(context);
}

//===----------------------------------------------------------------------===//
// MakeStruqOp
//===----------------------------------------------------------------------===//

LogicalResult quake::MakeStruqOp::verify() {
  if (getType().getNumMembers() != getNumOperands())
    return emitOpError("result type has different member count than operands");
  for (auto [ty, opnd] : llvm::zip(getType().getMembers(), getOperands())) {
    if (ty == opnd.getType())
      continue;
    auto veqTy = dyn_cast<quake::VeqType>(ty);
    auto veqOpndTy = dyn_cast<quake::VeqType>(opnd.getType());
    if (veqTy && !veqTy.hasSpecifiedSize() && veqOpndTy &&
        veqOpndTy.hasSpecifiedSize())
      continue;
    return emitOpError("member type not compatible with operand type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RelaxSizeOp
//===----------------------------------------------------------------------===//

LogicalResult quake::RelaxSizeOp::verify() {
  if (cast<quake::VeqType>(getType()).hasSpecifiedSize())
    emitOpError("return veq type must not specify a size");
  return success();
}

void quake::RelaxSizeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ForwardRelaxedSizePattern>(context);
}

//===----------------------------------------------------------------------===//
// SubVeqOp
//===----------------------------------------------------------------------===//

LogicalResult quake::SubVeqOp::verify() {
  if ((hasConstantLowerBound() && getRawLower() == kDynamicIndex) ||
      (!hasConstantLowerBound() && getRawLower() != kDynamicIndex))
    return emitOpError("invalid lower bound specified");
  if ((hasConstantUpperBound() && getRawUpper() == kDynamicIndex) ||
      (!hasConstantUpperBound() && getRawUpper() != kDynamicIndex))
    return emitOpError("invalid upper bound specified");
  if (hasConstantLowerBound() && hasConstantUpperBound()) {
    if (getRawLower() > getRawUpper())
      return emitOpError("invalid subrange specified");
    if (auto veqTy = dyn_cast<quake::VeqType>(getVeq().getType()))
      if (veqTy.hasSpecifiedSize())
        if (getRawLower() >= veqTy.getSize() ||
            getRawUpper() >= veqTy.getSize())
          return emitOpError(
              "subveq range does not fully intersect the input veq");
    if (auto veqTy = dyn_cast<quake::VeqType>(getResult().getType()))
      if (veqTy.hasSpecifiedSize())
        if (veqTy.getSize() != getRawUpper() - getRawLower() + 1)
          return emitOpError("incorrect size for result veq type");
  }
  return success();
}

void quake::SubVeqOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<FixUnspecifiedSubveqPattern, FuseConstantToSubveqPattern,
               RemoveSubVeqNoOpPattern>(context);
}

//===----------------------------------------------------------------------===//
// VeqSizeOp
//===----------------------------------------------------------------------===//

void quake::VeqSizeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<FoldInitStateSizePattern, ForwardConstantVeqSizePattern>(
      context);
}

//===----------------------------------------------------------------------===//
// WrapOp
//===----------------------------------------------------------------------===//

void quake::WrapOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<KillDeadWrapPattern>(context);
}

//===----------------------------------------------------------------------===//
// Measurements (MxOp, MyOp, MzOp)
//===----------------------------------------------------------------------===//

// Common verification for measurement operations.
template <typename MEAS>
LogicalResult verifyMeasurements(MEAS op, TypeRange targetsType,
                                 const Type bitsType) {
  if (failed(verifyWireResultsAreLinear(op)))
    return failure();
  bool mustBeStdvec =
      targetsType.size() > 1 ||
      (targetsType.size() == 1 && isa<quake::VeqType>(targetsType[0]));
  if (mustBeStdvec) {
    if (!isa<cudaq::cc::StdvecType>(op.getMeasOut().getType()))
      return op.emitOpError("must return `!cc.stdvec<!quake.measure>`, when "
                            "measuring a qreg, a series of qubits, or both");
  } else {
    if (!isa<quake::MeasureType>(op.getMeasOut().getType()))
      return op->emitOpError(
          "must return `!quake.measure` when measuring exactly one qubit");
  }
  if (op.getRegisterName())
    if (op.getRegisterName()->empty())
      return op->emitError("quake measurement name cannot be empty.");
  return success();
}

LogicalResult quake::MxOp::verify() {
  return verifyMeasurements(*this, getTargets().getType(),
                            getMeasOut().getType());
}

LogicalResult quake::MyOp::verify() {
  return verifyMeasurements(*this, getTargets().getType(),
                            getMeasOut().getType());
}

LogicalResult quake::MzOp::verify() {
  return verifyMeasurements(*this, getTargets().getType(),
                            getMeasOut().getType());
}

//===----------------------------------------------------------------------===//
// Discriminate
//===----------------------------------------------------------------------===//

LogicalResult quake::DiscriminateOp::verify() {
  if (isa<cudaq::cc::StdvecType>(getMeasurement().getType())) {
    auto stdvecTy = dyn_cast<cudaq::cc::StdvecType>(getResult().getType());
    if (!stdvecTy || !isa<IntegerType>(stdvecTy.getElementType()))
      return emitOpError("must return a !cc.stdvec<integral> type, when "
                         "discriminating a qreg, a series of qubits, or both");
  } else {
    auto measTy = isa<quake::MeasureType>(getMeasurement().getType());
    if (!measTy || !isa<IntegerType>(getResult().getType()))
      return emitOpError(
          "must return integral type when discriminating exactly one qubit");
  }
  return success();
}

LogicalResult quake::BundleCableOp::verify() {
  auto ty = cast<quake::CableType>(getResult().getType());
  if (getWires().size() != ty.getSize())
    return emitOpError("the bundle type size must equal the arity.");
  return success();
}

LogicalResult quake::TerminateCableOp::verify() {
  if (getResults().size() != getCable().getType().getSize())
    return emitOpError("the bundle type size must equal the coarity.");
  return success();
}

//===----------------------------------------------------------------------===//
// WireSetOp
//===----------------------------------------------------------------------===//

ParseResult quake::WireSetOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  StringAttr name;
  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();
  std::int32_t cardinality = 0;
  if (parser.parseLSquare() || parser.parseInteger(cardinality) ||
      parser.parseRSquare())
    return failure();
  result.addAttribute(getCardinalityAttrName(result.name),
                      parser.getBuilder().getI32IntegerAttr(cardinality));
  Attribute sparseEle;
  if (succeeded(parser.parseOptionalKeyword("adjacency")))
    if (parser.parseAttribute(sparseEle, getAdjacencyAttrName(result.name),
                              result.attributes))
      return failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  return success();
}

void quake::WireSetOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << '[' << getCardinality() << ']';
  if (auto adj = getAdjacency()) {
    p << " adjacency ";
    p.printAttribute(*adj);
  }
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getSymNameAttrName(), getCardinalityAttrName(), getAdjacencyAttrName()});
}

//===----------------------------------------------------------------------===//
// Operator interface
//===----------------------------------------------------------------------===//

// The following methods return to the operator's unitary matrix as a
// column-major array. For parameterizable operations, the matrix can only be
// built if the parameter can be computed at compilation time. These methods
// populate an empty array taken as a input. If the matrix was not successfully
// computed, the array will be left empty.

/// If the parameter is known at compilation-time, set the result value and
/// returns success. Otherwise, returns failure.
static LogicalResult getParameterAsDouble(Value parameter, double &result) {
  auto paramDefOp = parameter.getDefiningOp();
  if (!paramDefOp)
    return failure();
  if (auto constOp = dyn_cast<arith::ConstantOp>(paramDefOp)) {
    if (auto value = dyn_cast<FloatAttr>(constOp.getValue())) {
      result = value.getValueAsDouble();
      return success();
    }
  }
  return failure();
}

void quake::HOp::getOperatorMatrix(Matrix &matrix) {
  using namespace llvm::numbers;
  matrix.assign({inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2});
}

void quake::PhasedRxOp::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;

  // Get parameters
  double theta;
  double phi;
  if (failed(getParameterAsDouble(getParameter(), theta)) ||
      failed(getParameterAsDouble(getParameter(1), phi)))
    return;

  if (getIsAdj())
    theta *= -1;

  matrix.assign(
      {std::cos(theta / 2.), -1i * std::exp(1i * phi) * std::sin(theta / 2.),
       -1i * std::exp(-1i * phi) * std::sin(theta / 2.), std::cos(theta / 2.)});
}

void quake::R1Op::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;
  double theta;
  if (failed(getParameterAsDouble(getParameter(), theta)))
    return;
  if (getIsAdj())
    theta *= -1;
  matrix.assign({1, 0, 0, std::exp(theta * 1i)});
}

void quake::RxOp::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;
  double theta;
  if (failed(getParameterAsDouble(getParameter(), theta)))
    return;
  if (getIsAdj())
    theta *= -1;
  matrix.assign({std::cos(theta / 2.), -1i * std::sin(theta / 2.),
                 -1i * std::sin(theta / 2.), std::cos(theta / 2.)});
}

void quake::RxOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<MergeRotationPattern<quake::RxOp>>(context);
}

void quake::RyOp::getOperatorMatrix(Matrix &matrix) {
  // Get parameter
  double theta;
  if (failed(getParameterAsDouble(getParameter(), theta)))
    return;

  if (getIsAdj())
    theta *= -1;

  matrix.assign({std::cos(theta / 2.), std::sin(theta / 2.),
                 -std::sin(theta / 2.), std::cos(theta / 2.)});
}

void quake::RyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<MergeRotationPattern<quake::RyOp>>(context);
}

void quake::RzOp::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;

  // Get parameter
  double theta;
  if (failed(getParameterAsDouble(getParameter(), theta)))
    return;

  if (getIsAdj())
    theta *= -1;

  matrix.assign({std::exp(-1i * theta / 2.), 0, 0, std::exp(1i * theta / 2.)});
}

void quake::RzOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<MergeRotationPattern<quake::RzOp>>(context);
}

void quake::SOp::getOperatorMatrix(Matrix &matrix) {
  using namespace llvm::numbers;
  using namespace std::complex_literals;
  if (getIsAdj())
    matrix.assign({1, 0, 0, -1i});
  else
    matrix.assign({1, 0, 0, 1i});
}

void quake::SwapOp::getOperatorMatrix(Matrix &matrix) {
  matrix.assign({1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1});
}

void quake::TOp::getOperatorMatrix(Matrix &matrix) {
  using namespace llvm::numbers;
  if (getIsAdj())
    matrix.assign({1, 0, 0, {inv_sqrt2, -inv_sqrt2}});
  else
    matrix.assign({1, 0, 0, {inv_sqrt2, inv_sqrt2}});
}

void quake::U2Op::getOperatorMatrix(Matrix &matrix) {
  using namespace llvm::numbers;
  using namespace std::complex_literals;

  // Get parameters
  double phi;
  double lambda;
  if (failed(getParameterAsDouble(getParameter(), phi)) ||
      failed(getParameterAsDouble(getParameter(1), lambda)))
    return;

  if (getIsAdj()) {
    phi *= -1;
    lambda *= -1;
  }

  matrix.assign({inv_sqrt2, inv_sqrt2 * std::exp(phi * 1i),
                 -inv_sqrt2 * std::exp(lambda * 1i),
                 inv_sqrt2 * std::exp(1i * (phi + lambda))});
}

void quake::U3Op::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;

  // Get parameters
  double theta;
  double phi;
  double lambda;
  if (failed(getParameterAsDouble(getParameter(), theta)) ||
      failed(getParameterAsDouble(getParameter(1), phi)) ||
      failed(getParameterAsDouble(getParameter(2), lambda)))
    return;

  if (getIsAdj()) {
    theta *= -1;
    phi *= -1;
    lambda *= -1;
    std::swap(phi, lambda);
  }

  matrix.assign({std::cos(theta / 2.),
                 std::exp(phi * 1i) * std::sin(theta / 2.),
                 -std::exp(lambda * 1i) * std::sin(theta / 2.),
                 std::exp(1i * (phi + lambda)) * std::cos(theta / 2.)});
}

void quake::XOp::getOperatorMatrix(Matrix &matrix) {
  matrix.assign({0, 1, 1, 0});
}

void quake::YOp::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;
  matrix.assign({0, 1i, -1i, 0});
}

void quake::ZOp::getOperatorMatrix(Matrix &matrix) {
  matrix.assign({1, 0, 0, -1});
}

void quake::CustomUnitarySymbolOp::getOperatorMatrix(Matrix &matrix) {}

//===----------------------------------------------------------------------===//

/// Never inline a `quake.apply` of a variant form of a kernel. The apply
/// operation must be rewritten to a call before it is inlined when the apply
/// is a variant form.
bool cudaq::EnableInlinerInterface::isLegalToInline(Operation *call,
                                                    Operation *callable,
                                                    bool) const {
  if (auto applyOp = dyn_cast<quake::ApplyOp>(call))
    if (applyOp.applyToVariant())
      return false;
  if (auto destFunc = call->getParentOfType<func::FuncOp>())
    if (destFunc.getName().ends_with(".thunk"))
      if (auto srcFunc = dyn_cast<func::FuncOp>(callable))
        return !(srcFunc->hasAttr(cudaq::entryPointAttrName));
  return true;
}

using EffectsVectorImpl =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

/// For an operation with modeless effects, the operation always has effects
/// on the control and target quantum operands, whether those operands are in
/// reference or value form. A operation with modeless effects is not removed
/// when its result(s) is (are) unused.
[[maybe_unused]] inline static void
getModelessEffectsImpl(EffectsVectorImpl &effects, ValueRange controls,
                       ValueRange targets) {
  for (auto v : controls)
    effects.emplace_back(MemoryEffects::Read::get(), v,
                         SideEffects::DefaultResource::get());
  for (auto v : targets) {
    effects.emplace_back(MemoryEffects::Read::get(), v,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), v,
                         SideEffects::DefaultResource::get());
  }
}

/// For an operation with moded effects, the operation conditionally has
/// effects on the control and target quantum operands. If those operands are
/// in reference form, then the operation does have effects on those
/// references. Control operands have a read effect, while target operands
/// have both a read and write effect. If the operand is in value form, the
/// operation introduces no effects on that operand.
inline static void getModedEffectsImpl(EffectsVectorImpl &effects,
                                       ValueRange controls,
                                       ValueRange targets) {
  for (auto v : controls)
    if (isa<quake::RefType, quake::VeqType>(v.getType()))
      effects.emplace_back(MemoryEffects::Read::get(), v,
                           SideEffects::DefaultResource::get());
  for (auto v : targets)
    if (isa<quake::RefType, quake::VeqType>(v.getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), v,
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), v,
                           SideEffects::DefaultResource::get());
    }
}

/// Quake reset has modeless effects.
void quake::getResetEffectsImpl(EffectsVectorImpl &effects,
                                ValueRange targets) {
  getModedEffectsImpl(effects, {}, targets);
}

/// Quake measurement operations have moded effects.
void quake::getMeasurementEffectsImpl(EffectsVectorImpl &effects,
                                      ValueRange targets) {
  getModedEffectsImpl(effects, {}, targets);
}

/// Quake quantum operators have moded effects.
void quake::getOperatorEffectsImpl(EffectsVectorImpl &effects,
                                   ValueRange controls, ValueRange targets) {
  getModedEffectsImpl(effects, controls, targets);
}

// This is a workaround for ODS generating these member function declarations
// but not having a way to define them in the ODS.
// clang-format off
#define GATE_OPS(MACRO) MACRO(XOp) MACRO(YOp) MACRO(ZOp) MACRO(HOp) MACRO(SOp) \
  MACRO(TOp) MACRO(SwapOp) MACRO(U2Op) MACRO(U3Op) MACRO(R1Op) MACRO(RxOp)     \
  MACRO(RyOp) MACRO(RzOp) MACRO(PhasedRxOp) MACRO(CustomUnitarySymbolOp)
#define MEASURE_OPS(MACRO) MACRO(MxOp) MACRO(MyOp) MACRO(MzOp)
#define QUANTUM_OPS(MACRO) MACRO(ResetOp) MACRO(ExpPauliOp) GATE_OPS(MACRO)    \
  MEASURE_OPS(MACRO)
#define WIRE_OPS(MACRO) MACRO(FromControlOp) MACRO(ResetOp) MACRO(NullWireOp)  \
  MACRO(UnwrapOp)
// clang-format on
#define INSTANTIATE_CALLBACKS(Op)                                              \
  void quake::Op::getEffects(                                                  \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    getEffectsImpl(effects);                                                   \
  }

QUANTUM_OPS(INSTANTIATE_CALLBACKS)

#define INSTANTIATE_LINEAR_TYPE_VERIFY(Op)                                     \
  LogicalResult quake::Op::verify() {                                          \
    return verifyWireResultsAreLinear(getOperation());                         \
  }

#define VERIFY_OPS(MACRO) GATE_OPS(MACRO) WIRE_OPS(MACRO)

VERIFY_OPS(INSTANTIATE_LINEAR_TYPE_VERIFY)

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

using namespace cudaq;

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.cpp.inc"
