/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef NVQPP_OPTIMIZER_DIALECT_QUAKE_CANONICAL
#define NVQPP_OPTIMIZER_DIALECT_QUAKE_CANONICAL

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeOps.td"

def KnownSizePred : Constraint<
      CPred<"$0.getType().isa<quake::VeqType>() && "
            "$0.getType().cast<quake::VeqType>().hasSpecifiedSize()">>;

def UnknownSizePred : Constraint<
      CPred<"$0.getType().isa<quake::VeqType>() && "
            "!$0.getType().cast<quake::VeqType>().hasSpecifiedSize()">>;

def createConstantOp : NativeCodeCall<
      "$_builder.create<mlir::arith::ConstantOp>($_loc, $0.getType(),"
      "  $_builder.getIntegerAttr($0.getType(),"
      "   $1.getType().cast<quake::VeqType>().getSize()))">;

// %4 = quake.veq_size(%3 : !quake.veq<10>) : 164
// ────────────────────────────────────────────────
// %4 = constant 10 : i64
def ForwardConstantVeqSizePattern : Pat<
      (quake_VeqSizeOp:$res $veq), (createConstantOp $res, $veq),
      [(KnownSizePred $veq)]>;

def SizeIsPresentPred : Constraint<CPred<
      "$0.size() == 1 &&"
      " dyn_cast_or_null<mlir::arith::ConstantOp>($0[0].getDefiningOp())">>;

def createAllocaOp : NativeCodeCall<
      "quake::createConstantAlloca($_builder, $_loc, $0, $1)">;

// %2 = constant 10 : i32
// %3 = quake.alloca (%2 : i32) : !quake.ref<?>
// ───────────────────────────────────────────
// %3 = quake.alloca : !quake.ref<10>
def FuseConstantToAllocaPattern : Pat<
      (quake_AllocaOp:$alloca $optSize), (createAllocaOp $alloca, $optSize),
      [(SizeIsPresentPred $optSize)]>;

def createSizedSubVecOp : NativeCodeCall<
      "quake::createSizedSubVecOp($_builder, $_loc, $0, $1, $2, $3)">;

def ArgIsConstantPred : Constraint<CPred<
      "dyn_cast_or_null<mlir::arith::ConstantOp>($0.getDefiningOp())">>;

def IsUnknownVec : Constraint<CPred<
      "dyn_cast_or_null<mlir::arith::ConstantOp>($0.getDefiningOp())">>;

// %1 = constant 4 : i64
// %2 = constant 10 : i64
// %3 = quake.subvec (%0 : !quake.ref<12>, %1 : i64, %2 : i64) : !quake.ref<?>
// ─────────────────────────────────────────────────────────────────────────────
// %1 = constant 4 : i64
// %2 = constant 10 : i64
// %new3 = quake.subvec (%0 : !quake.ref<12>, %1 : i64, %2 : i64) :
//     !quake.ref<7>
// %3 = quake.relax_size %new3 : (!quake.ref<7>) -> !quake.ref<?>
def FuseConstantToSubvecPattern : Pat<
      (quake_SubVecOp:$subvec $v, $lo, $hi),
      (createSizedSubVecOp $subvec, $v, $lo, $hi),
      [(UnknownSizePred $subvec), (ArgIsConstantPred $lo),
       (ArgIsConstantPred $hi)]>;

#endif
