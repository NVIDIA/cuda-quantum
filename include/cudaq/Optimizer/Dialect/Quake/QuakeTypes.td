/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QUAKE_TYPES
#define CUDAQ_OPTIMIZER_DIALECT_QUAKE_TYPES

include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// BaseType
//===----------------------------------------------------------------------===//

class QuakeType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QuakeDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// QRefType
//===----------------------------------------------------------------------===//

def QRefType : QuakeType<"QRef", "qref"> {
  let summary = "reference to a qubit";
  let description = [{
    A `qref` represents reference to a qubit.  One can view the values of this
    type as the horizontal lines in a quantum circuit diagram:
    ```
    q0 : ─────●────────── <--+
              │              |
            ┌─┴┐  ┌──┐       |
    q1 : ───┤  ├──┤  ├─── <--+-- `qref`s
            └──┘  └──┘
    ```

    Note that these are reference values.  Each use of the value as a target
    implies side-effects on the value. (Similar to `volatile` semantics.)

    Take the follwoing example, which implements the above circuit:
    ```mlir
    %q0 = quake.alloc : !quake.qref
    %q1 = quake.alloc : !quake.qref
    quake.op1 [%q0] (%q1)
    quake.op2 (%q1)
    ```
    The semantics assumes that `op1` will have side-effects on the
    reference-value `%q1` and thus must dominate `op2` to preserve a valid
    execution of the program.  Note that op1 does not imply a side-effect on
    `%q0` because it uses it as control.
  }];
  let genStorageClass = 0;
}

//===----------------------------------------------------------------------===//
// QVecType
//===----------------------------------------------------------------------===//

def QVecType : QuakeType<"QVec", "qvec"> {
  let summary = "a aggregate of qubit references";
  let description = [{
    A value of type `qvec` is a (linear) collection of values of type `qref`.
    These aggregates are a convenience for refering to an entire group of
    references to qubits.
  }];

  let parameters = (ins "std::size_t":$size);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    bool hasSpecifiedSize() const { return getSize(); }
    static QVecType getUnsized(mlir::MLIRContext *ctx) {
      return QVecType::get(ctx, 0);
    }
  }];
}

def AnyQTypeLike : TypeConstraint<Or<[QRefType.predicate, QVecType.predicate]>,
    "quake.qref or quake.qvec">;
def AnyQType : Type<AnyQTypeLike.predicate, "quantum type">;

def IsStdvecTypePred : CPred<"$_self.isa<::cudaq::cc::StdvecType>()">;

class StdvecOf<list<Type> allowedTypes> : Type<
    And<[IsStdvecTypePred, Concat<"[](::mlir::Type elementType) { return ",
      SubstLeaves<"$_self", "elementType", AnyTypeOf<allowedTypes>.predicate>,
        "; }($_self.cast<::cudaq::cc::StdvecType>().getElementType())">]>,
    "stdvec of " # AnyTypeOf<allowedTypes>.summary # " values",
    "::cudaq::cc::StdvecType">;

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_TYPES
