/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef QLX_INTERFACES_LINEAR_OWNERSHIP_INTERFACES_TD
#define QLX_INTERFACES_LINEAR_OWNERSHIP_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Linear types
//===----------------------------------------------------------------------===//

def QLX_LinearTypeInterface : TypeInterface<"LinearTypeInterface"> {
  let cppNamespace = "::qlx";
  let description = [{
    Marks a type whose values are linear. Ordinary MLIR SSA allows any number
    of uses, but a linear value must have exactly one owner on every
    execution path. `qlx::verifyLinearOwnership` enforces this rule for every
    value whose type implements this interface and reports `isLinear()`.

    A use is a *consume* unless its owner carries the `LinearBorrowsOperands`
    trait. A linear value must be consumed exactly once on every path. A
    `mayDrop()` type (value-semantic frame state, which has no disposal
    operation) may instead go unconsumed.
  }];
  let methods = [
    InterfaceMethod<
      "Whether values of this type instance are single-owner linear values. "
      "A parameterized type may be linear for some parameters only, such as "
      "an event handle whose ownership parameter is \"shared\".",
      "bool", "isLinear", (ins), [{}], [{ return true; }]>,
    InterfaceMethod<
      "Whether a linear value of this type may end its life without a "
      "consuming use.",
      "bool", "mayDrop", (ins), [{}], [{ return false; }]>
  ];
}

//===----------------------------------------------------------------------===//
// Operation roles in the linear-ownership analysis
//===----------------------------------------------------------------------===//

class QLX_LinearOwnershipTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::qlx";
}

// The op's body is one independent linear-ownership scope. Every linear
// value defined in it, outside nested scopes, is checked on its own.
def QLX_LinearOwnershipScope : QLX_LinearOwnershipTrait<"LinearOwnershipScope">;

// Every use by this op only observes its linear operands and never takes
// ownership. An observation must not follow the consuming use.
def QLX_LinearBorrowsOperands
    : QLX_LinearOwnershipTrait<"LinearBorrowsOperands">;

// Exactly one region of this op runs. A captured linear value consumed in
// one region must be consumed in every non-empty region, and consuming it in
// two regions is one dynamic consume.
def QLX_LinearExclusiveRegions
    : QLX_LinearOwnershipTrait<"LinearExclusiveRegions">;

// The regions of this op may run many times. A linear value defined outside
// the op must not be consumed inside it; it must enter as an explicit carry.
def QLX_LinearRepeatedRegions
    : QLX_LinearOwnershipTrait<"LinearRepeatedRegions">;

#endif // QLX_INTERFACES_LINEAR_OWNERSHIP_INTERFACES_TD
