/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

inline bool needsToBeRenamed(mlir::StringRef name) {
  return name.startswith(cudaq::opt::QIRQISPrefix) &&
         !name.endswith("__body") && !name.endswith("__adj") &&
         !name.endswith("__ctl");
}

inline bool callToInvokeWithXCtrlOneTarget(mlir::StringRef callee,
                                           mlir::ValueRange args) {
  if ((args.size() == 4) && (callee == cudaq::opt::NVQIRInvokeWithControlBits))
    if (auto addrOf = dyn_cast_or_null<mlir::LLVM::AddressOfOp>(
            args[1].getDefiningOp())) {
      return addrOf.getGlobalName().startswith(
          std::string(cudaq::opt::QIRQISPrefix) + "x__ctl");
    }
  return false;
}

inline bool isIntToPtrOp(mlir::Value operand) {
  return dyn_cast_or_null<mlir::LLVM::IntToPtrOp>(operand.getDefiningOp());
}

static constexpr char resultIndexName[] = "result.index";

inline mlir::Value createMeasureCall(mlir::PatternRewriter &builder,
                                     mlir::Location loc, mlir::OpResult result,
                                     mlir::ValueRange args) {
  auto op = cast<mlir::LLVM::CallOp>(result.getDefiningOp());
  auto ptrTy = cudaq::opt::getResultType(builder.getContext());
  if (auto intAttr =
          dyn_cast_or_null<mlir::IntegerAttr>(op->getAttr(resultIndexName))) {
    auto constOp = builder.create<mlir::LLVM::ConstantOp>(loc, intAttr);
    auto cast = builder.create<mlir::LLVM::IntToPtrOp>(loc, ptrTy, constOp);
    builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{}, cudaq::opt::QIRMeasureBody,
        mlir::ArrayRef<mlir::Value>{args[0], cast});
    return cast;
  }
  op.emitError("mz op must have an associated result index.");
  return {};
}

namespace {
#include "cudaq/Optimizer/CodeGen/Peephole.inc"
}
