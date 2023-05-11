/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/QTX/QTXTypes.h.inc"

//===----------------------------------------------------------------------===//

namespace qtx {

mlir::ParseResult
parseOperatorType(mlir::AsmParser &parser,
                  mlir::SmallVectorImpl<mlir::Type> &parameters,
                  mlir::SmallVectorImpl<mlir::Type> &targets,
                  mlir::SmallVectorImpl<mlir::Type> &classicResults,
                  mlir::SmallVectorImpl<mlir::Type> &targetResults);

mlir::ParseResult
parseOperatorType(mlir::AsmParser &parser,
                  mlir::SmallVectorImpl<mlir::Type> &parameters,
                  mlir::SmallVectorImpl<mlir::Type> &targets,
                  mlir::SmallVectorImpl<mlir::Type> &targetResults);

void printOperatorType(mlir::AsmPrinter &printer, mlir::Operation *op,
                       mlir::TypeRange parameters, mlir::TypeRange targets,
                       mlir::TypeRange classicResults,
                       mlir::TypeRange targetResults);

void printOperatorType(mlir::AsmPrinter &printer, mlir::Operation *op,
                       mlir::TypeRange parameters, mlir::TypeRange targets,
                       mlir::TypeRange targetResults);

} // namespace qtx
