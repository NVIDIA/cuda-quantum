/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// CircuitOp
//===----------------------------------------------------------------------===//

// Parse a delimited list of arguments.  The argument list either has to
// consistently have ssa-id's followed by types, or just be a type list.
// It is _not_ ok to sometimes have ssa-id's and sometimes not.
static ParseResult parseArgs(OpAsmParser &parser,
                             OpAsmParser::Delimiter delimiter,
                             SmallVectorImpl<OpAsmParser::Argument> &args,
                             SmallVectorImpl<Type> &types, unsigned &counter) {
  return parser.parseCommaSeparatedList(delimiter, [&]() -> ParseResult {
    // Parse argument name if present.
    OpAsmParser::Argument arg;
    auto argPresent = parser.parseOptionalArgument(arg, /*allowType=*/true,
                                                   /*allowAttrs=*/true);
    if (argPresent.has_value()) {
      if (failed(argPresent.value()))
        return failure(); // Present but malformed.

      // Reject this if the preceding argument was missing a name.
      if (!args.empty() && args.back().ssaName.name.empty())
        return parser.emitError(arg.ssaName.location,
                                "expected type instead of SSA identifier");
    } else {
      arg.ssaName.location = parser.getCurrentLocation();
      // Otherwise we just have a type list without SSA names.  Reject this
      // if the preceding argument had a name.
      if (!args.empty() && !args.back().ssaName.name.empty())
        return parser.emitError(arg.ssaName.location,
                                "expected SSA identifier");

      NamedAttrList attrs;
      if (parser.parseType(arg.type))
        return failure();
      if (parser.parseOptionalAttrDict(attrs))
        return failure();
      if (parser.parseOptionalLocationSpecifier(arg.sourceLoc))
        return failure();

      arg.attrs = attrs.getDictionary(parser.getContext());
    }
    args.push_back(arg);
    types.push_back(arg.type);
    counter += 1;
    return success();
  });
}

static ParseResult parseInputs(OpAsmParser &parser,
                               SmallVectorImpl<OpAsmParser::Argument> &args,
                               unsigned &numParameters, unsigned &numTargets,
                               SmallVectorImpl<Type> &types) {
  typedef AsmParser::Delimiter Delimiter;
  if (parseArgs(parser, Delimiter::OptionalLessGreater, args, types,
                numParameters))
    return failure();
  if (parseArgs(parser, Delimiter::Paren, args, types, numTargets))
    return failure();
  return success();
}

static ParseResult parseResultTypes(OpAsmParser &parser,
                                    OpAsmParser::Delimiter delimiter,
                                    SmallVectorImpl<Type> &types,
                                    unsigned &counter) {
  return parser.parseCommaSeparatedList(delimiter, [&]() -> ParseResult {
    if (parser.parseType(types.emplace_back()))
      return failure();
    counter += 1;
    return success();
  });
}

static ParseResult parseResults(OpAsmParser &parser,
                                unsigned &numClassicalResults,
                                unsigned &numTargetResults,
                                SmallVectorImpl<Type> &signatureTypes) {
  typedef AsmParser::Delimiter Delimiter;
  if (parser.parseOptionalArrow())
    return success(); // No results. FIXME: is this ok?
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parseResultTypes(parser, Delimiter::OptionalLessGreater, signatureTypes,
                       numClassicalResults))
    return failure();
  if (parseResultTypes(parser, Delimiter::OptionalParen, signatureTypes,
                       numTargetResults))
    return failure();
  if (numClassicalResults || numTargetResults)
    return success(); //
  return parser.emitError(
      loc,
      "failed parsing the result types. The expected format is: `<` classical "
      "`>` `(` wires `)`.  NOTE: the delimiters are not optional");
}

ParseResult qtx::CircuitOp::parse(OpAsmParser &parser, OperationState &result) {
  unsigned numParameters = 0;
  unsigned numTargets = 0;
  unsigned numClassicalResults = 0;
  unsigned numTargetResults = 0;
  // TODO: Find a well-motivated choice for the number of inlined elements for
  // these SmallVector
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> signatureTypes;

  // Parse the visibility attribute.
  (void)impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  if (parseInputs(parser, args, numParameters, numTargets, signatureTypes))
    return failure();
  if (parseResults(parser, numClassicalResults, numTargetResults,
                   signatureTypes))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  result.addAttribute(qtx::CircuitOp::getArgSegmentSizesAttrName(),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(numParameters),
                           static_cast<int32_t>(numTargets)}));

  // Build the circuit's type and register as an attribute
  auto type =
      OperatorType::get(parser.getContext(), numParameters, numTargets,
                        numClassicalResults, numTargetResults, signatureTypes);
  result.addAttribute("operator_type", TypeAttr::get(type));

  // Parse the optional circuit body.
  Region *body = result.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, args, /*enableNameShadowing=*/false);

  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    if (body->empty())
      return parser.emitError(loc, "expected non-empty circuit body");
  }
  return success();
}

void qtx::CircuitOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  if (getSymVisibility())
    printer << getSymVisibility() << ' ';
  printer.printSymbolName(getName());

  // Print the inputs arguments.
  if (getNumParameters()) {
    printer << '<';
    llvm::interleaveComma(getParameters(), printer,
                          [&](auto arg) { printer.printRegionArgument(arg); });
    printer << '>';
  }

  printer << '(';
  llvm::interleaveComma(getTargets(), printer,
                        [&](auto arg) { printer.printRegionArgument(arg); });
  printer << ')';

  // Print the result types.
  auto classicResultTypes = getClassicalResultTypes();
  auto targetTypes = getTargetTypes();
  if (!classicResultTypes.empty() || !targetTypes.empty()) {
    printer << " -> ";
  }
  if (!classicResultTypes.empty()) {
    printer << '<';
    printer << classicResultTypes;
    printer << '>';
  }
  if (!targetTypes.empty()) {
    printer << '(';
    printer << targetTypes;
    printer << ')';
  }

  // Print out attributes, if present.
  llvm::StringRef ignoredAttrs[] = {
      getSymVisibilityAttrName(), getOperatorTypeAttrName(),
      qtx::CircuitOp::getArgSegmentSizesAttrName(),
      SymbolTable::getSymbolAttrName()};

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           ignoredAttrs);

  // Print the body if this is not an external circuit.
  Region &body = getBody();
  if (!body.empty()) {
    printer << ' ';
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

void qtx::CircuitOp::build(OpBuilder &builder, OperationState &result,
                           StringAttr name) {
  // Add an attribute for the name.
  result.addAttribute(builder.getStringAttr("sym_name"), name);

  // Create a region and a block for the body.
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
}

void qtx::CircuitOp::build(OpBuilder &builder, OperationState &result,
                           StringAttr visibility, StringAttr name,
                           ValueRange classicalArgs, ValueRange quantumArgs,
                           TypeRange classicalReturnsTypes) {
  auto operatorType = builder.getType<qtx::OperatorType>(
      TypeRange(classicalArgs), TypeRange(quantumArgs), classicalReturnsTypes,
      TypeRange(quantumArgs));
  build(builder, result, visibility, name, operatorType);
  result.addAttribute(
      qtx::CircuitOp::getArgSegmentSizesAttrName(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(classicalArgs.size()),
                                    static_cast<int32_t>(quantumArgs.size())}));
}

LogicalResult qtx::CircuitOp::verify() {
  const auto &type = getOperatorType();
  const auto targets = type.getTargets();
  const auto targetResults = type.getTargetResults();
  if (targets.size() != targetResults.size())
    return emitOpError("has ")
           << targets.size() << " target operand(s), but returns "
           << targetResults.size() << " target(s)";
  if (targets != targetResults)
    return emitOpError("has mismatching input(s) and result(s) target(s)");
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static ParseResult parseReturnLikeOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> classicalOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> quantumOperands;
  SmallVector<Type, 1> classicalTypes;
  SmallVector<Type, 1> quantumTypes;

  SMLoc classicalResultsLoc;
  if (succeeded(parser.parseOptionalLess())) {
    classicalResultsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(classicalOperands))
      return failure();
    if (parser.parseGreater())
      return failure();
  }

  SMLoc quantumResultsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(quantumOperands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (!classicalOperands.empty() || !quantumOperands.empty()) {
    if (parser.parseColon())
      return failure();
    if (succeeded(parser.parseOptionalLess())) {
      if (parser.parseTypeList(classicalTypes) || parser.parseGreater() ||
          parser.resolveOperands(classicalOperands, classicalTypes,
                                 classicalResultsLoc, result.operands))
        return failure();
    }
    if (!quantumOperands.empty())
      if (parser.parseTypeList(quantumTypes) ||
          parser.resolveOperands(quantumOperands, quantumTypes,
                                 quantumResultsLoc, result.operands))
        return failure();
  }

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(classicalOperands.size()),
                           static_cast<int32_t>(quantumOperands.size())}));
  return success();
}

template <typename Op>
static void printReturnLikeOp(OpAsmPrinter &printer, Op &op) {
  if (!op.getClassical().empty()) {
    printer << ' ' << '<';
    printer << op.getClassical();
    printer << '>';
  }
  if (!op.getTargets().empty()) {
    printer << ' ' << op.getTargets();
  }

  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{
                                    "operand_segment_sizes",
                                });

  if (!op.getClassical().empty() || !op.getTargets().empty()) {
    printer << ' ' << ':';
    if (!op.getClassical().empty()) {
      printer << ' ' << '<';
      printer << op.getClassical().getTypes();
      printer << '>';
    }
    if (!op.getTargets().empty()) {
      printer << ' ' << op.getTargets().getTypes();
    }
  }
}

ParseResult qtx::ReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseReturnLikeOp(parser, result);
}

void qtx::ReturnOp::print(OpAsmPrinter &printer) {
  printReturnLikeOp(printer, *this);
}

LogicalResult qtx::ReturnOp::verify() {
  auto circuit = cast<CircuitOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &classic = circuit.getOperatorType().getClassicalResults();
  const auto &targets = circuit.getOperatorType().getTargets();
  if (getClassical().size() != classic.size())
    return emitOpError("has ")
           << getClassical().size()
           << " classical operands, but enclosing circuit (@"
           << circuit.getName() << ") returns " << classic.size()
           << " classical results";
  if (getTargets().size() != targets.size())
    return emitOpError("has ")
           << getTargets().size()
           << " target operands, but enclosing circuit (@" << circuit.getName()
           << ") returns " << targets.size() << " target results";
  for (unsigned i = 0, e = classic.size(); i != e; ++i)
    if (getClassical()[i].getType() != classic[i])
      return emitError() << "type of return classical operand " << i << " ("
                         << getClassical()[i].getType()
                         << ") doesn't match circuit classical result type ("
                         << classic[i] << ")"
                         << " in circuit @" << circuit.getName();
  for (unsigned i = 0, e = targets.size(); i != e; ++i)
    if (getTargets()[i].getType() != targets[i])
      return emitError() << "type of return target operand " << i << " ("
                         << getTargets()[i].getType()
                         << ") doesn't match circuit target result type ("
                         << targets[i] << ")"
                         << " in circuit @" << circuit.getName();
  return success();
}

ParseResult qtx::UnrealizedReturnOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseReturnLikeOp(parser, result);
}

void qtx::UnrealizedReturnOp::print(OpAsmPrinter &printer) {
  printReturnLikeOp(printer, *this);
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

LogicalResult
qtx::ApplyOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto cktAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!cktAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  CircuitOp ckt =
      symbolTable.lookupNearestSymbolFrom<CircuitOp>(*this, cktAttr);
  if (!ckt)
    return emitOpError() << "'" << cktAttr.getValue()
                         << "' does not reference a valid circuit";

  // Verify the number of operands and results.
  auto cktType = ckt.getOperatorType();
  if (cktType.getNumParameters() != getNumParameters())
    return emitOpError("incorrect number of classical operands for callee");
  if (cktType.getNumTargets() != getNumTargets())
    return emitOpError("incorrect number of target operands for callee");
  if (cktType.getNumClassicalResults() != getNumClassicalResults())
    return emitOpError("incorrect number of classical results for callee");

  // Verify the operand and results types match the callee.
  for (unsigned i = 0, e = cktType.getNumParameters(); i != e; ++i)
    if (getParameter(i).getType() != cktType.getParameter(i))
      return emitOpError("parameter type mismatch: expected operand type ")
             << cktType.getParameter(i) << ", but provided "
             << getParameter(i).getType() << " for parameter number " << i;

  for (unsigned i = 0, e = cktType.getNumTargets(); i != e; ++i) {
    if (getTarget(i).getType() != cktType.getTarget(i))
      return emitOpError("target type mismatch: expected type ")
             << cktType.getTarget(i) << ", but provided "
             << getTarget(i).getType() << " for target number " << i;
    if (getNewTarget(i).getType() != cktType.getTarget(i))
      return emitOpError("target result type mismatch: expected type ")
             << cktType.getTarget(i) << ", but provided "
             << getTarget(i).getType() << " for target result number " << i;
  }

  for (unsigned i = 0, e = cktType.getNumClassicalResults(); i != e; ++i) {
    if (getClassicalResult(i).getType() != cktType.getClassicalResult(i))
      return emitOpError("classic result type mismatch: expected type ")
             << cktType.getClassicalResult(i) << ", but provided "
             << getClassicalResult(i).getType() << " for classic result number "
             << i;
  }
  return success();
}
