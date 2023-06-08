/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace qtx {
class CircuitOp;
}

namespace cudaq::opt {

#define GEN_PASS_CLASSES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"

} // namespace cudaq::opt

#define GATE_OPS(MACRO)                                                        \
  MACRO(XOp), MACRO(YOp), MACRO(ZOp), MACRO(HOp), MACRO(SOp), MACRO(TOp),      \
      MACRO(SwapOp), MACRO(R1Op), MACRO(RxOp), MACRO(PhasedRxOp), MACRO(RyOp), \
      MACRO(RzOp), MACRO(U2Op), MACRO(U3Op)
#define MEASURE_OPS(MACRO) MACRO(MxOp), MACRO(MyOp), MACRO(MzOp)
#define QUANTUM_OPS(MACRO) GATE_OPS(MACRO), MEASURE_OPS(MACRO)
