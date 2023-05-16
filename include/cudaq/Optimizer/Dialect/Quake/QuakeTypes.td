/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QUAKE_TYPES
#define CUDAQ_OPTIMIZER_DIALECT_QUAKE_TYPES

include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// BaseType
//===----------------------------------------------------------------------===//

class QuakeType<string name, string typeMnemonic, list<Trait> traits = [],
        string baseCppClass = "mlir::Type">
    : TypeDef<QuakeDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Quantum value type
//===----------------------------------------------------------------------===//

def WireType : QuakeType<"Wire", "wire"> {
  let description = [{
    A `wire` is a primitive quantum "SSA-value" that differs from a traditional
    SSA-value in that it can and will be modified when used in any target
    position. That means quantum "SSA-values" are not true SSA values. `wire`
    values can only be used once and have somewhat similar properties to
    volatile memory. An operation that uses a `wire` in a target position is
    required to return a new `wire` value reflecting the updated/new state.

    The following is a simple example of using `wire` values.

    ```mlir
      %q0 = ... : !quake.wire
      %q1 = ... : !quake.wire
      %q2 = quake.gate1 [%q0] %q1 : (!quake.wire, !quake.wire) -> !quake.wire
      %q3 = quake.gate2 %q2 : (!quake.wire) -> !quake.wire
    ```

    See also the description of the `ref` type.
  }];

  let genStorageClass = 0;
}

def ControlType : QuakeType<"Control", "control"> {
  let description = [{
    A qubit of type control is a value of type wire used in a control position.
    Using a qubit in a control position means the qubit is an SSA-value. It is
    defined once and can be used multiple times (in multiple control positions)
    without changing value.
  }];
  let genStorageClass = 0;
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

def RefType : QuakeType<"Ref", "ref"> {
  let summary = "reference to a wire";
  let description = [{
    A `ref` represents a reference to a wire.  One can view the values of
    this type as the horizontal lines in the following a quantum circuit
    diagram. A value of type `ref` is an SSA-value.

    ```
    q0 : ─────●────────── <--+
              │              |
            ┌─┴┐  ┌──┐       |
    q1 : ───┤  ├──┤  ├─── <--+-- `ref`s
            └──┘  └──┘
    ```

    The following example is the quake representation of the above circuit.
    (gate1 and gate2 are quantum gates.)

    ```mlir
      %q0 = quake.alloca : !quake.ref
      %q1 = quake.alloca : !quake.ref
      quake.gate1 [%q0] (%q1)
      quake.gate2 (%q1)
    ```

    The semantics is that `gate1` has quantum memory side-effects through the
    reference `%q1` to the referent wire, which is modified (read and write).
    (The reference does not change, but the wire it refers to is considered
    volatile.) Similarly `gate2` has the same side-effects through the same
    reference, `%q1`. Because these operations modify the same wire, they
    cannot be reordered relative to one another.
    
    Furthermore, `gate1` implies only a quantum memory read side-effect on the
    referent wire through the reference `%q0`. Because `%q0` appears in a
    control position in `gate1`, the wire value is not updated.

    Not all operations will have side-effects on the referent wire of the
    reference value of `ref` type.
  }];
  let genStorageClass = 0;
}

//===----------------------------------------------------------------------===//
// VeqType
//===----------------------------------------------------------------------===//

def VeqType : QuakeType<"Veq", "veq"> {
  let summary = "a aggregate of wire references";
  let description = [{
    A value of type `veq` is a (linear) collection of values of type `ref`.
    These aggregates are a convenience for referring to an entire group of
    references to wires. A `veq` value is a proper SSA-value. `ref` values
    in a `veq` are not volatile.
  }];

  let parameters = (ins "std::size_t":$size);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    bool hasSpecifiedSize() const { return getSize(); }
    static VeqType getUnsized(mlir::MLIRContext *ctx) {
      return VeqType::get(ctx, 0);
    }
  }];
}

def AnyQTypeLike : TypeConstraint<Or<[WireType.predicate, VeqType.predicate,
        ControlType.predicate, RefType.predicate]>, "quake quantum types">;
def AnyQType : Type<AnyQTypeLike.predicate, "quantum type">;
def AnyRefTypeLike : TypeConstraint<Or<[VeqType.predicate,
        RefType.predicate]>, "quake quantum reference types">;
def AnyRefType : Type<AnyRefTypeLike.predicate, "quantum reference type">;
def AnyQValueTypeLike : TypeConstraint<Or<[WireType.predicate,
        ControlType.predicate]>, "quake quantum value types">;
def AnyQValueType : Type<AnyQValueTypeLike.predicate, "quantum value type">;

def IsStdvecTypePred : CPred<"$_self.isa<::cudaq::cc::StdvecType>()">;

class StdvecOf<list<Type> allowedTypes> : Type<
    And<[IsStdvecTypePred, Concat<"[](::mlir::Type elementType) { return ",
      SubstLeaves<"$_self", "elementType", AnyTypeOf<allowedTypes>.predicate>,
        "; }($_self.cast<::cudaq::cc::StdvecType>().getElementType())">]>,
    "stdvec of " # AnyTypeOf<allowedTypes>.summary # " values",
    "::cudaq::cc::StdvecType">;

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_TYPES
