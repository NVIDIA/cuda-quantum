/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef NVQPP_OPTIMIZER_CODEGEN_IQM
#define NVQPP_OPTIMIZER_CODEGEN_IQM

include "cudaq/Optimizer/Dialect/Quake/QuakeOps.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// This is the set of peephole transformations for the IQM target. The peepholes
// convert the standard gates to collections of a phased-RX op and CZ.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

def HasOneElement : Constraint<CPred<"$0.size() == 1">>;
def IsEmpty : Constraint<CPred<"$0.empty()">>;
def IsNull : Constraint<CPred<"$0 ? false : true">>;
def IsNotNull : Constraint<CPred<"$0 ? true : false">>;

// quake.rx(θ, ...)
// ───────────────────
// phased_r(θ, 0, ...)

def CreateRX : NativeCodeCall<
    "[&]() -> std::size_t {"
    "  auto zero = cudaq::iqm::createFloatConstant($_builder, $_loc,"
    "    $1.front().getType().cast<mlir::FloatType>(), 0.0);"
    "  $_builder.create<quake::PhasedRxOp>($_loc, $0,"
    "    mlir::ArrayRef<mlir::Value>{$1.front(), zero}, $2, $3, $4);"
    "  return 0; }()">;

def IQMRxPattern : Pat<
      (RxOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateRX $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.ry(θ, ...)
// ───────────────────────
// phased_r(θ, π / 2, ...)

def CreateRY : NativeCodeCall<
    "[&]() -> std::size_t {"
    "  auto halfPi = cudaq::iqm::createFloatConstant($_builder, $_loc, "
    "    $1.front().getType().cast<mlir::FloatType>(), M_PI_2);"
    "  $_builder.create<quake::PhasedRxOp>($_loc, $0,"
    "    mlir::ArrayRef<mlir::Value>{$1.front(), halfPi}, $2, $3, $4);"
    "  return 0; }()">;

def IQMRyPattern : Pat<
      (RyOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateRY $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.rz(θ, ...)
// ─────────────────────────
// phased_r(π / 2, 0, ...)
// phased_r(-θ, π / 2, ...)
// phased_r(-π / 2, 0, ...)

def CreateRZ : NativeCodeCall<
    "cudaq::iqm::"
    "createRZ($_builder, $_loc, $0, $1.front(), $2, $3, $4)">;

def IQMRzPattern : Pat<
      (RzOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateRZ $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.h(...)
// ───────────────────────────
// phased_r(π / 2, π / 2, ...)
// phased_r(π, 0, ...)

def CreateH : NativeCodeCall<
    "cudaq::iqm::"
    "createH($_builder, $_loc, $0, $1, $2, $3, $4)">;

def IQMHPattern : Pat<
      (HOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateH $adj, $parm, $ctrl, $targ, $neg),
      [(IsEmpty $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.x(...)
// ───────────────────
// phased_r(π, 0, ...)

def CreateX : NativeCodeCall<
    "cudaq::iqm::"
    "createX($_builder, $_loc, $0, $1, $2, $3, $4)">;

def IQMXPattern : Pat<
      (XOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateX $adj, $parm, $ctrl, $targ, $neg),
      [(IsEmpty $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.y(...)
// ───────────────────────
// phased_r(π, π / 2, ...)

def CreateY : NativeCodeCall<
    "cudaq::iqm::"
    "createY($_builder, $_loc, $0, $1, $2, $3, $4)">;

def IQMYPattern : Pat<
      (YOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateY $adj, $parm, $ctrl, $targ, $neg),
      [(IsEmpty $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.z(...)
// ────────────────────────
// phased_r(π / 2, 0, ...)
// phased_r(-π, π / 2, ...)
// phased_r(-π / 2, 0, ...)

def CreateZ : NativeCodeCall<
    "cudaq::iqm::"
    "createZ($_builder, $_loc, $0, $1, $2, $3, $4)">;

def IQMZPattern : Pat<
      (ZOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateZ $adj, $parm, $ctrl, $targ, $neg),
      [(IsEmpty $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.s(...)
// ────────────────────────
// phased_r(π / 2, 0, ...)
// phased_r(-π / 2, π / 2, ...)
// phased_r(-π / 2, 0, ...)

def CreateS : NativeCodeCall<
    "cudaq::iqm::"
    "createS($_builder, $_loc, $0, $1, $2, $3, $4)">;

def IQMSPattern : Pat<
      (SOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateS $adj, $parm, $ctrl, $targ, $neg),
      [(IsEmpty $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.t(...)
// ────────────────────────
// phased_r(π / 2, 0, ...)
// phased_r(-π / 4, π / 2, ...)
// phased_r(-π / 2, 0, ...)

def CreateT : NativeCodeCall<
    "cudaq::iqm::"
    "createT($_builder, $_loc, $0, $1, $2, $3, $4)">;

def IQMTPattern : Pat<
      (TOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateT $adj, $parm, $ctrl, $targ, $neg),
      [(IsEmpty $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.r1(θ, ...)
// ────────────────────────
// phased_r(π / 2, 0, ...)
// phased_r(-θ, π / 2, ...)
// phased_r(-π / 2, 0, ...)

def CreateR1 : NativeCodeCall<
    "cudaq::iqm::"
    "createR1($_builder, $_loc, $0, $1.front(), $2, $3, $4)">;

def IQMR1Pattern : Pat<
      (R1Op $adj, $parm, $ctrl, $targ, $neg),
      (CreateR1 $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $parm), (IsEmpty $ctrl), (IsNull $adj), (IsNull $neg)]>;

// quake.x<ctrl> i, j
// ──────────────────
// quake.h j
// quake.z<ctrl> i, j
// quake.h j

def CreateXctrl : NativeCodeCall<
    "[&]() -> std::size_t {"
    "  $_builder.create<quake::HOp>($_loc, mlir::ValueRange{}, $3);"
    "  $_builder.create<quake::ZOp>($_loc, $2, $3);"
    "  $_builder.create<quake::HOp>($_loc, mlir::ValueRange{}, $3);"
    "  return 0; }()">;

def IQMXctrlPattern : Pat<
      (XOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateXctrl $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $ctrl), (IsEmpty $parm), (IsNull $adj), (IsNull $neg)]>;

#endif
