/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "mlir/IR/OpImplementation.h"

namespace cudaq {

mlir::ParseResult parseParameters(
    mlir::OpAsmParser &parser, mlir::UnitAttr &isAdj,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &parameters);

void printParameters(mlir::OpAsmPrinter &printer, mlir::Operation *,
                     mlir::UnitAttr, mlir::OperandRange parameters);

mlir::ParseResult parseParameters(
    mlir::OpAsmParser &parser,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &parameters);

mlir::ParseResult parseParameters(
    mlir::OpAsmParser &parser,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &parameters,
    mlir::SmallVectorImpl<mlir::Type> &types);

void printParameters(mlir::OpAsmPrinter &printer, mlir::Operation *,
                     mlir::OperandRange parameters, mlir::TypeRange types = {});

} // namespace cudaq
