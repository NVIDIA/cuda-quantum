/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/Support/Registry.h"
#include "mlir/IR/Builders.h"

namespace nvqpp {

class QISBuilder {
public:
  virtual ~QISBuilder() = default;
  virtual mlir::Operation::result_range
  buildInstruction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ValueRange general_operands) = 0;
};
using QISBuilderRegistry = llvm::Registry<nvqpp::QISBuilder>;

} // namespace nvqpp
