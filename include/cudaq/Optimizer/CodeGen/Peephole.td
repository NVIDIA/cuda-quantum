/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef NVQPP_OPTIMIZER_CODEGEN_PEEPHOLE
#define NVQPP_OPTIMIZER_CODEGEN_PEEPHOLE

include "cudaq/Optimizer/Dialect/Quake/QuakeOps.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//

def InvokeOnXWithOneControl : Constraint<CPred<
    "callToInvokeWithXCtrlOneTarget($0.getValue(), $1)">>;

def CreateCallCnot : NativeCodeCall<
    "[&]() -> std::size_t {"
    "  $_builder.create<mlir::LLVM::CallOp>($_loc,"
    "    mlir::TypeRange{}, cudaq::opt::QIRCnot, $0.drop_front(2));"
    "  return 0; }()">;

// %1 = address_of @__quantum__qis__x__ctl
// %2 = call @invokewithControlBits %1, %ctrl, %targ
// ─────────────────────────────────────────────────
// %2 = call __quantum__qis__cnot %ctrl, %targ
def XCtrlOneTargetToCNot : Pat<
    (LLVM_CallOp $callee, $args, $_, $_), (CreateCallCnot $args),
    [(InvokeOnXWithOneControl $callee, $args)]>;

//===----------------------------------------------------------------------===//

def NeedsRenaming : Constraint<CPred<"needsToBeRenamed($0.getValue())">>;

def CreateAddressOf : NativeCodeCall<
    "$_builder.create<mlir::LLVM::AddressOfOp>($_loc, $0.getType(),"
    "  $1.getValue().str() + \"__body\")">;

// %4 = address_of @__quantum__cis__*
// ────────────────────────────────────────
// %4 = address_of @__quantum__cis__*__body
def AddrOfCisToBase : Pat<
    (LLVM_AddressOfOp:$addr $global), (CreateAddressOf $addr, $global),
    [(NeedsRenaming $global)]>;

//===----------------------------------------------------------------------===//

// Apply special rule for `mz`. See below.
def FuncNotMeasure : Constraint<CPred<
    "!$_self.getValue().startswith(cudaq::opt::QIRMeasure)">>;

def CreateCallOp : NativeCodeCall<
    "[&]() -> std::size_t {"
    "  $_builder.create<mlir::LLVM::CallOp>($_loc, mlir::TypeRange{},"
    "    mlir::FlatSymbolRefAttr::get($_builder.getContext(),"
    "    $0.getValue().str() + \"__body\"), $1, $2, $3);"
    "  return 0; }()">;

// %4 = call @__quantum__cis__*
// ──────────────────────────────────
// %4 = call @__quantum__cis__*__body
def CalleeConv : Pat<
      (LLVM_CallOp $callee, $args, $fm, $bw), 
      (CreateCallOp $callee, $args, $fm, $bw),
      [(NeedsRenaming $callee), (FuncNotMeasure:$callee)]>;

//===----------------------------------------------------------------------===//

def IsArrayGetElementPtrId : Constraint<CPred<
    "$0.getValue().str() == cudaq::opt::QIRArrayGetElementPtr1d">>;

def EraseArrayGEPOp : NativeCodeCall<
    "$_builder.create<mlir::LLVM::UndefOp>($_loc,"
    "  cudaq::opt::getQubitType($_builder.getContext()))">;

def EraseDeadArrayGEP : Pat<
    (LLVM_CallOp:$call $callee, $_, $_, $_), (EraseArrayGEPOp),
    [(IsArrayGetElementPtrId $callee), (HasNoUseOf:$call)]>;

//===----------------------------------------------------------------------===//

def IsaAllocateCall : Constraint<CPred<
    "$0.getValue().str() == cudaq::opt::QIRArrayQubitAllocateArray">>;

def EraseArrayAllocateOp : NativeCodeCall<
    "$_builder.create<mlir::LLVM::UndefOp>($_loc,"
    "  cudaq::opt::getArrayType($_builder.getContext()))">;

// Replace the call with a dead op to DCE.
//
// %0 = call @allocate ... : ... -> T*
// ───────────────────────────────────
// %0 = undef : T*
def EraseArrayAlloc : Pat<
    (LLVM_CallOp $callee, $_, $_, $_), (EraseArrayAllocateOp),
    [(IsaAllocateCall $callee)]>;

//===----------------------------------------------------------------------===//

def IsaReleaseCall : Constraint<CPred<
    "$0.getValue().str() == cudaq::opt::QIRArrayQubitReleaseArray || "
    "$0.getValue().str() == cudaq::opt::QIRArrayQubitReleaseQubit">>;

def EraseArrayReleaseOp : NativeCodeCall<"static_cast<std::size_t>(0)">;

// Remove the release calls. This removes both array allocations as well as
// qubit singletons.
//
// call @release %5 : (!Qubit) -> ()
// ─────────────────────────────────
def EraseArrayRelease : Pat<
    (LLVM_CallOp $callee, $_, $_, $_), (EraseArrayReleaseOp),
    [(IsaReleaseCall $callee)]>;

//===----------------------------------------------------------------------===//

def IsaMeasureCall : Constraint<CPred<
    "$_self.getValue() == cudaq::opt::QIRMeasure">>;
    
def IsaIntToPtrOperand : Constraint<CPred<"isIntToPtrOp($0[0])">>;

def CreateMeasureCall : NativeCodeCall<
    "createMeasureCall($_builder, $_loc, $0, $1)">;

// %result = call @__quantum__qis__mz(%qbit) : (!Qubit) -> i1
// ──────────────────────────────────────────────────────────────
// call @__quantum__qis__mz_body(%qbit, %result) : (Q*, R*) -> ()
def MeasureCallConv : Pat<
    (LLVM_CallOp:$call $callee, $args, $_, $_),
    (CreateMeasureCall $call, $args),
    [(IsaMeasureCall:$callee), (IsaIntToPtrOperand $args)]>;

//===----------------------------------------------------------------------===//

def IsaMeasureToRegisterCall : Constraint<CPred<
    "$_self.getValue() == cudaq::opt::QIRMeasureToRegister">>;

// %result = call @__quantum__qis__mz__to__register(%qbit, i8) : (!Qubit) -> i1
// ────────────────────────────────────────────────────────────────────────────
// call @__quantum__qis__mz_body(%qbit, %result) : (Q*, R*) -> ()
def MeasureToRegisterCallConv : Pat<
    (LLVM_CallOp:$call $callee, $args, $_, $_),
    (CreateMeasureCall $call, $args),
    [(IsaMeasureToRegisterCall:$callee), (IsaIntToPtrOperand $args)]>;

#endif
