/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_LINEAR_OWNERSHIP_INTERFACES
#define CUDAQ_OPTIMIZER_DIALECT_LINEAR_OWNERSHIP_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Linear types
//===----------------------------------------------------------------------===//

def LinearTypeInterface : TypeInterface<"LinearTypeInterface"> {
  let cppNamespace = "::cudaq";
  let description = [{
    Marks a type whose values are linear. Ordinary MLIR SSA allows any number
    of uses, but a linear value must have exactly one owner on every
    execution path.

    A use is a *consume* unless its owner carries the `LinearBorrowsOperands`
    trait. A linear value must be consumed exactly once on every path. A
    `mayDrop()` type (value-semantic state with no disposal operation) may
    instead go unconsumed.

    This is a correctness contract checked by a verifier. It is related to,
    but not the same as, `SSITypeTrait`. That trait stops MLIR
    canonicalization from merging block arguments across control-flow paths.
    Some of those merges break linearity, such as folding a loop-carried value
    into the value from outside the loop. Others do not. The types that carry
    `SSITypeTrait` today (`!quake.wire`, `!quake.cable`) are linear by
    definition.
  }];
  let methods = [
    InterfaceMethod<
      "Whether values of this type instance are single-owner linear values. "
      "A parameterized type may be linear for some parameters only, such as "
      "a handle whose ownership parameter is \"shared\".",
      "bool", "isLinear", (ins), [{}], [{ return true; }]>,
    InterfaceMethod<
      "Whether a linear value of this type may end its life without a "
      "consuming use.",
      "bool", "mayDrop", (ins), [{}], [{ return false; }]>
  ];
}

//===----------------------------------------------------------------------===//
// Operation roles in linear-ownership checking
//===----------------------------------------------------------------------===//

class LinearOwnershipTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::cudaq";
}

// The op's body is one independent linear-ownership scope. Every linear
// value defined in it, outside nested scopes, is checked on its own.
def LinearOwnershipScope : LinearOwnershipTrait<"LinearOwnershipScope">;

// Every use by this op only observes its linear operands and never takes
// ownership. An observation must not follow the consuming use.
def LinearBorrowsOperands : LinearOwnershipTrait<"LinearBorrowsOperands">;

// Exactly one region of this op runs. A captured linear value consumed in
// one region must be consumed in every non-empty region, and consuming it in
// two regions is one dynamic consume.
def LinearExclusiveRegions : LinearOwnershipTrait<"LinearExclusiveRegions">;

// The regions of this op may run many times. A linear value defined outside
// the op must not be consumed inside it. It must enter as an explicit carry.
def LinearRepeatedRegions : LinearOwnershipTrait<"LinearRepeatedRegions">;

#endif // CUDAQ_OPTIMIZER_DIALECT_LINEAR_OWNERSHIP_INTERFACES
