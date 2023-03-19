/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/QTX/QTXTypes.h"
#include "TypeDetail.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/QTX/QTXTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// OperatorType
//===----------------------------------------------------------------------===//

static ParseResult parseTypeList(AsmParser &parser,
                                 AsmParser::Delimiter delimiter,
                                 SmallVectorImpl<Type> &types) {
  return parser.parseCommaSeparatedList(delimiter, [&]() -> ParseResult {
    return parser.parseType(types.emplace_back());
  });
}

static void printOperatorType(AsmPrinter &printer, TypeRange parameters,
                              TypeRange targets, TypeRange classicResults,
                              TypeRange targetResults) {
  if (!parameters.empty()) {
    printer << '<';
    llvm::interleaveComma(parameters, printer,
                          [&](const Type &type) { printer.printType(type); });
    printer << '>';
  }
  printer << '(';
  llvm::interleaveComma(targets, printer,
                        [&](const Type &type) { printer.printType(type); });
  printer << ") -> ";
  if (!classicResults.empty()) {
    printer << '<';
    llvm::interleaveComma(classicResults, printer,
                          [&](const Type &type) { printer.printType(type); });
    printer << '>';
  }
  printer << '(';
  llvm::interleaveComma(targetResults, printer,
                        [&](const Type &type) { printer.printType(type); });
  printer << ')';
}

ParseResult qtx::parseOperatorType(AsmParser &parser,
                                   SmallVectorImpl<Type> &parameters,
                                   SmallVectorImpl<Type> &targets,
                                   SmallVectorImpl<Type> &classicResults,
                                   SmallVectorImpl<Type> &targetResults) {
  typedef AsmParser::Delimiter Delimiter;
  if (parseTypeList(parser, Delimiter::OptionalLessGreater, parameters) ||
      parseTypeList(parser, Delimiter::Paren, targets) || parser.parseArrow() ||
      parseTypeList(parser, Delimiter::OptionalLessGreater, classicResults) ||
      parseTypeList(parser, Delimiter::Paren, targetResults))
    return failure();
  return success();
}

ParseResult qtx::parseOperatorType(AsmParser &parser,
                                   SmallVectorImpl<Type> &parameters,
                                   SmallVectorImpl<Type> &targets,
                                   SmallVectorImpl<Type> &targetResults) {
  typedef AsmParser::Delimiter Delimiter;
  if (parseTypeList(parser, Delimiter::OptionalLessGreater, parameters) ||
      parseTypeList(parser, Delimiter::Paren, targets) || parser.parseArrow() ||
      parseTypeList(parser, Delimiter::Paren, targetResults))
    return failure();
  return success();
}

void qtx::printOperatorType(AsmPrinter &printer, Operation *,
                            TypeRange parameters, TypeRange targets,
                            TypeRange classicResults, TypeRange targetResults) {
  ::printOperatorType(printer, parameters, targets, classicResults,
                      targetResults);
}

void qtx::printOperatorType(AsmPrinter &printer, Operation *,
                            TypeRange parameters, TypeRange targets,
                            TypeRange targetResults) {
  ::printOperatorType(printer, parameters, targets, {}, targetResults);
}

Type qtx::OperatorType::parse(AsmParser &parser) {
  // TODO: Find a well-motivated choice for the number of inlined elements for
  // these SmallVector
  SmallVector<Type> parameters;
  SmallVector<Type> targets;
  SmallVector<Type> classicResults;
  SmallVector<Type> targetResults;

  if (parseOperatorType(parser, parameters, targets, classicResults,
                        targetResults))
    return nullptr;

  return OperatorType::get(parser.getContext(), parameters, targets,
                           classicResults, targetResults);
}

void qtx::OperatorType::print(AsmPrinter &printer) const {
  ::printOperatorType(printer, getParameters(), getTargets(),
                      getClassicalResults(), getTargetResults());
}

unsigned qtx::OperatorType::getNumParameters() const {
  return getImpl()->numParameters;
}

unsigned qtx::OperatorType::getNumTargets() const {
  return getImpl()->numTargets;
}

unsigned qtx::OperatorType::getNumClassicalResults() const {
  return getImpl()->numClassicalResults;
}

unsigned qtx::OperatorType::getNumTargetResults() const {
  return getImpl()->numTargetResults;
}

ArrayRef<Type> qtx::OperatorType::getParameters() const {
  return getImpl()->getParameters();
}

ArrayRef<Type> qtx::OperatorType::getTargets() const {
  return getImpl()->getTargets();
}

ArrayRef<Type> qtx::OperatorType::getClassicalResults() const {
  return getImpl()->getClassicalResults();
}

ArrayRef<Type> qtx::OperatorType::getTargetResults() const {
  return getImpl()->getTargetResults();
}

ArrayRef<Type> qtx::OperatorType::getResults() const {
  return getImpl()->getTargetResults();
}

qtx::OperatorType qtx::OperatorType::clone(unsigned numParameters,
                                           unsigned numTargets,
                                           unsigned numClassicalResults,
                                           unsigned numTargetResults,
                                           TypeRange types) const {
  return get(getContext(), numParameters, numTargets, numClassicalResults,
             numTargetResults, types);
}

//===----------------------------------------------------------------------===//

void qtx::QTXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cudaq/Optimizer/Dialect/QTX/QTXTypes.cpp.inc"
      >();
}
