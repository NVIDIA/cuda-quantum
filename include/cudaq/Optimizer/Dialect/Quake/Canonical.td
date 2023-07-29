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

// %4 = quake.veq_size %3 : (!quake.veq<10>) -> 164
// ────────────────────────────────────────────────
// %4 = constant 10 : i64
def ForwardConstantVeqSizePattern : Pat<
      (quake_VeqSizeOp:$res $veq), (createConstantOp $res, $veq),
      [(KnownSizePred $veq)]>;

def SizeIsPresentPred : Constraint<CPred<
      "$0.size() == 1 &&"
      "isa<mlir::arith::ConstantIntOp, mlir::arith::ConstantIndexOp>("
      "  $0[0].getDefiningOp())">>;

def createAllocaOp : NativeCodeCall<
      "quake::createConstantAlloca($_builder, $_loc, $0, $1)">;

// %2 = constant 10 : i32
// %3 = quake.alloca !quake.veq<?>[%2 : i32]
// ───────────────────────────────────────────
// %3 = quake.alloca !quake.veq<10>
def FuseConstantToAllocaPattern : Pat<
      (quake_AllocaOp:$alloca $optSize), (createAllocaOp $alloca, $optSize),
      [(SizeIsPresentPred $optSize)]>;

def createExtractRefOp : NativeCodeCall<
      "$_builder.create<quake::ExtractRefOp>($_loc, $0,"
      " cast<mlir::arith::ConstantOp>($1[0].getDefiningOp()).getValue()."
      " cast<mlir::IntegerAttr>().getInt())">;

// %2 = constant 10 : i32
// %3 = quake.extract_ref %1[%2] : (!quake.veq<?>, i32) -> !quake.ref
// ───────────────────────────────────────────
// %3 = quake.extract_ref %1[10] : (!quake.veq<?>) -> !quake.ref
def FuseConstantToExtractRefPattern : Pat<
      (quake_ExtractRefOp $veq, $index, $rawIndex),
      (createExtractRefOp $veq, $index),
      [(SizeIsPresentPred $index)]>;

def createSizedSubVeqOp : NativeCodeCall<
      "quake::createSizedSubVeqOp($_builder, $_loc, $0, $1, $2, $3)">;

def ArgIsConstantPred : Constraint<CPred<
      "dyn_cast_or_null<mlir::arith::ConstantOp>($0.getDefiningOp())">>;

def IsUnknownVec : Constraint<CPred<
      "dyn_cast_or_null<mlir::arith::ConstantOp>($0.getDefiningOp())">>;

// %1 = constant 4 : i64
// %2 = constant 10 : i64
// %3 = quake.subveq (%0 : !quake.ref<12>, %1 : i64, %2 : i64) : !quake.ref<?>
// ─────────────────────────────────────────────────────────────────────────────
// %1 = constant 4 : i64
// %2 = constant 10 : i64
// %new3 = quake.subveq (%0 : !quake.ref<12>, %1 : i64, %2 : i64) :
//     !quake.ref<7>
// %3 = quake.relax_size %new3 : (!quake.ref<7>) -> !quake.ref<?>
def FuseConstantToSubveqPattern : Pat<
      (quake_SubVeqOp:$subveq $v, $lo, $hi),
      (createSizedSubVeqOp $subveq, $v, $lo, $hi),
      [(UnknownSizePred $subveq), (ArgIsConstantPred $lo),
       (ArgIsConstantPred $hi)]>;

#endif
