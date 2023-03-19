/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef NVQPP_OPTIMIZER_CODEGEN_DECOMPOSITION
#define NVQPP_OPTIMIZER_CODEGEN_DECOMPOSITION

include "cudaq/Optimizer/Dialect/Quake/QuakeOps.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// These are the peephole transformations for general decompositions.
//===----------------------------------------------------------------------===//

def HasOneElement : Constraint<CPred<"$0.size() == 1">>;
def IsEmpty : Constraint<CPred<"$0.empty()">>;
def IsNull : Constraint<CPred<"$0 ? false : true">>;
def IsNotNull : Constraint<CPred<"$0 ? true : false">>;

// quake.rx(θ, κ, ϟ)
// ─────────────────
// quake.s(ϟ)
// quake.x(κ, ϟ)
// quake.ry(-θ/2, ϟ)
// quake.x(κ, ϟ)
// quake.ry(θ/2, ϟ)
// quake.rz(-π/2, ϟ)

def CreateRx : NativeCodeCall<
      "cudaq::decomposition::"
      "createRx($_builder, $_loc, $0, $1.front(), $2.front(), $3, $4)">;

def QtnmRxPattern : Pat<
      (RxOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateRx $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $parm), (HasOneElement $ctrl), (IsNull $adj),
       (IsNull $neg)]>;

// quake.ry(θ, κ, ϟ)
// ─────────────────
// quake.ry(θ/2, ϟ)
// quake.x(κ, ϟ)
// quake.ry(-θ/2, ϟ)
// quake.x(κ, ϟ)

def CreateRy : NativeCodeCall<
      "cudaq::decomposition::"
      "createRy($_builder, $_loc, $0, $1.front(), $2.front(), $3, $4)">;

def QtnmRyPattern : Pat<
      (RyOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateRy $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $parm), (HasOneElement $ctrl), (IsNull $adj),
       (IsNull $neg)]>;

// quake.rz(θ, κ, ϟ)
// ─────────────────
// quake.rz(θ/2, ϟ)
// quake.x(κ, ϟ)
// quake.rz(-θ/2, ϟ)
// quake.x(κ, ϟ)

def CreateRz : NativeCodeCall<
      "cudaq::decomposition::"
      "createRz($_builder, $_loc, $0, $1.front(), $2.front(), $3, $4)">;

def QtnmRzPattern : Pat<
      (RzOp $adj, $parm, $ctrl, $targ, $neg),
      (CreateRz $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $parm), (HasOneElement $ctrl), (IsNull $adj),
       (IsNull $neg)]>;

// quake.r1(θ, κ, ϟ)
// ─────────────────
// quake.rz(θ, κ, ϟ)
// quake.rz(θ/2, κ)

def CreateR1 : NativeCodeCall<
      "cudaq::decomposition::"
      "createR1($_builder, $_loc, $0, $1.front(), $2.front(), $3, $4)">;

def QtnmR1Pattern : Pat<
      (R1Op $adj, $parm, $ctrl, $targ, $neg),
      (CreateR1 $adj, $parm, $ctrl, $targ, $neg),
      [(HasOneElement $parm), (HasOneElement $ctrl), (IsNull $adj),
       (IsNull $neg)]>;

#endif
