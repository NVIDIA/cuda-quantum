/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_TYPES
#define CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "cudaq/Optimizer/Dialect/QTX/QTXDialect.td"

//===----------------------------------------------------------------------===//
// BaseType
//===----------------------------------------------------------------------===//

class QTXType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QTXDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// WireType
//===----------------------------------------------------------------------===//

def WireType : QTXType<"Wire", "wire"> {
  let summary = "wire";
  let description = [{
    A `wire` represents an intermediate "state" of a single qubit in time.  One
    can view the values of this type as line segments in a quantum circuit
    diagram:
    ```
                            wire                         w
         <----------------------------------------->┌──┐<->
    q0 : ───────────────────●──────●─────────────●──┤  ├───
                            │      │             │  └──┘
                 wire       │      │  w      w   │   wire
         <------------------│--->┌─┴┐<->┌──┐<->┌─┴┐<------>
    q1 : ────────────●──────●────┤  ├───┤  ├───┤  ├────────
                     │      │    └──┘   └──┘   └──┘
          w      w   │  w   │         wire               w
         <->┌──┐<->┌─┴┐<->┌─┴┐<-------------------->┌──┐<->
    q2 : ───┤  ├───┤  ├───┤  ├──────────────────────┤  ├───
            └──┘   └──┘   └──┘                      └──┘
    ```

    Note that these are reference values.  Semantically, these values have
    little relation to quantum states, they exist only so that we can exploit
    the SSA def-use chains to represent quantum circuits as DAG within the IR.
  }];
}

//===----------------------------------------------------------------------===//
// WireArrayType
//===----------------------------------------------------------------------===//

def WireArrayType : QTXType<"WireArray", "wire_array"> {
  let summary = "a fixed-sized aggregate of wires";
  let description = [{
    While value semantics and the SSA form are great to represent dataflow of
    scalar variables, representing aggregate structure such as arrays or
    vectors has been traditionally difficult.  In QTX, these difficulties are
    exacerbated because of quantum mechanics properties (e.g., No-Cloning
    Theorem).

    A common pattern in quantum circuits is to apply operators to specific
    qubits in a register.  In QTX, a register of qubits is represented as an
    array of wires.  We need to be extra careful when using specific wires from
    an array.  In particular, we need to make sure that wires are not cloned.
    For example:

    First, we allocate a qubit register with 4 qubtis:
    ```mlir
    %array = qtx.alloc : !qtx.wire_array<4>
    ```

    Now, what should one do to apply an operator to the qubit at index 0?  QTX
    solves the problem using the concept of wire borrowing: one can borrow
    individual wires from an array and yield then back to the array after use.

    To support the wire borrowing concept and better prevent mistakes, we
    implement a wire array type which keep tracks of how many qubits are
    currently borrowed (i.e., dead).  For example:  once we borrow a wire from
    an array, a new array value will be created with a different type, which
    explicitly informs the existence of a “dead” wire within the array:
    ```mlir
    !qtx.wire_array<n, dead = 1>
    ```

    When we yield the wire back, we get a new array with the original type.

    Example:
    ```mlir
    // Wire array with four wires
    !qtx.wire_array<4>

    // By default, the number of dead qubits is assumed to be zero.  Hence
    // the following type is the same as the previous one.
    !qtx.wire_array<4, dead = 0>
    ```

    Note: If `dead = 0`, the printer will omit the parameter.
  }];

  let parameters = (ins "size_t":$size,
                        DefaultValuedParameter<"size_t", "0">:$dead);
  let assemblyFormat = "`<` $size (`,` `dead` `=` $dead^)? `>`";
}

//===----------------------------------------------------------------------===//
// Quantum Types
//===----------------------------------------------------------------------===//

def AnyQType : AnyTypeOf<[WireType, WireArrayType]>;

//===----------------------------------------------------------------------===//
// OperatorType
//===----------------------------------------------------------------------===//

def OperatorType : QTXType<"Operator", "operator"> {
  let summary = "Map from a list of inputs to a list of results";
  let description = [{
    Conceptually, a `quantum operator` is an effect that _might_ modify the
    state of a subset of wires.  Most often, this effect is unitary evolution
    ---in this case, we say that the operator is a unitary.

    This type is fairly similar to the builtin `FunctionType`.  One can think of
    it as an operator signature:  It consists of a list of formal argument types
    and a list of formal result types.

    The main difference with regards to `FunctionType` is its structure.  We
    divide the input arguments and results into classical and quantum.  The
    input arguments are:
      * An optional list of classical input types---the operator's parameters.
      * A list of quantum inputs types---the operator's targets.

    The results are:
      * An optional list of classical results.
      * A list of quantum outputs, which must match the list of quantum inputs.

    Syntax:
    ```
    classical-type-list ::=  `|` `|` | `|` type-list-no-parens `|`
    operator-inputs ::= classical-type-list? type-list-parens
    operator-results ::= classical-type-list? type-list-parens
    operator-type ::= operator-inputs `->` operator-results
    ```

    Example:
    ```
    // Operator that takes one classical parameter and acts on one wire
    |f64|(!qtx.wire) -> (!qtx.wire)

    // Operator that acts on two wires
    (!qtx.wire, !qtx.wire) -> (!qtx.wire, !qtx.wire)
    ```
  }];
  let parameters = (ins "::mlir::ArrayRef<::mlir::Type>":$parameters,
                        "::mlir::ArrayRef<::mlir::Type>":$targets,
                        "::mlir::ArrayRef<::mlir::Type>":$classical_results,
                        "::mlir::ArrayRef<::mlir::Type>":$target_results);
  let skipDefaultBuilders = 1;
  let genStorageClass = 0;
  let builders = [
    TypeBuilder<(ins
        CArg<"unsigned">:$n_parameters,
        CArg<"unsigned">:$n_targets,
        CArg<"unsigned">:$n_classical_results,
        CArg<"unsigned">:$n_target_results,
        CArg<"::mlir::TypeRange">:$types), [{
      return $_get($_ctxt, n_parameters, n_targets, n_classical_results,
                   n_target_results, types);
    }]>,
    TypeBuilder<(ins
        CArg<"::mlir::TypeRange">:$parameters,
        CArg<"::mlir::TypeRange">:$targets,
        CArg<"::mlir::TypeRange">:$classical_results,
        CArg<"::mlir::TypeRange">:$target_results), [{
      // TODO: Find a well-motivated choice for the number of inlined elements
      // for this SmallVector
      mlir::SmallVector<mlir::Type> types;
      types.reserve(parameters.size() + targets.size() +
                    classical_results.size() + target_results.size());

      types.append(parameters.begin(), parameters.end());
      types.append(targets.begin(), targets.end());
      types.append(classical_results.begin(), classical_results.end());
      types.append(target_results.begin(), target_results.end());
      return $_get($_ctxt, parameters.size(), targets.size(),
                   classical_results.size(), target_results.size(), types);
    }]>
  ];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Structural Information
    //===------------------------------------------------------------------===//

    unsigned getNumParameters() const;
    unsigned getNumTargets() const;
    unsigned getNumClassicalResults() const;
    unsigned getNumTargetResults() const;

    //===------------------------------------------------------------------===//
    // Element Access
    //===------------------------------------------------------------------===//

    using Type = ::mlir::Type;

    Type getParameter(unsigned i) const { return getParameters()[i]; }
    Type getTarget(unsigned i) const { return getTargets()[i]; }
    Type getClassicalResult(unsigned i) const {
      return getClassicalResults()[i];
    }
    Type getTargetResult(unsigned i) const { return getTargetResults()[i]; }

    //===------------------------------------------------------------------===//

    mlir::ArrayRef<mlir::Type> getResults() const;

    using TypeRange = ::mlir::TypeRange;

    OperatorType clone(unsigned numParameters, unsigned numTargets,
                       unsigned numClassicalResults, unsigned numTargetResults,
                       TypeRange types) const;
  }];
  let hasCustomAssemblyFormat = 1;
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_TYPES

