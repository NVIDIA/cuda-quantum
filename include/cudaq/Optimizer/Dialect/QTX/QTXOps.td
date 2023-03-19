/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_OPS
#define CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_OPS

include "cudaq/Optimizer/Dialect/QTX/QTXDialect.td"
include "cudaq/Optimizer/Dialect/QTX/QTXInterfaces.td"
include "cudaq/Optimizer/Dialect/QTX/QTXTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class QTXOp<string mnemonic, list<Trait> traits = []> :
    Op<QTXDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

def qtx_AllocaOp : QTXOp<"alloca"> {
  let summary = "allocate qubit(s) and return a wire (or wire array)";
  let description = [{
    The operation allocates either a single qubit or a register of qubits.  The
    return value is either a wire, `!qtx.wire`, or an array of wires,
    `qtx.wire_array<size>`, that represents the intermediate state of a qubit or
    qubits, resp., in time.

    In QTX, all qubits must be explicitly deallocated within the _same_ scope of
    their allocation. (TODO: Implement a way of verifying this rule.)

    Examples:

    ```mlir
    // Allocate a single qubis and return a wire
    %wire = qtx.alloc : !qtx.wire

    // Allocate a qubit register and return an array of wires
    %array = qtx.alloc : !qtx.wire_array<5>
    ```
  }];

  let arguments = (ins );
  let results = (outs AnyQType:$wire_or_array);
  let builders = [
    OpBuilder<(ins), [{
      return build($_builder, $_state, $_builder.getType<WireType>());
    }]>,
    OpBuilder<(ins "size_t":$size), [{
      return build($_builder, $_state,
                   $_builder.getType<WireArrayType>(size, /*dead=*/0));
    }]>
  ];
  let assemblyFormat = "attr-dict `:` type($wire_or_array)";
  let hasVerifier = 1;
}

def qtx_DeallocOp : QTXOp<"dealloc"> {
  let summary = "deallocates a qubit from a its wire";
  let description = [{
    The `qtx.dealloc` operation deallocates a qubit or a register of qubits.  It
    takes as input a wire (or array of wires) that represents a \"state\".

    NOTE:  An arrays must not have dead wires to be deallocated.

    Example:

    ```mlir
    // Deallocates a single qubit
    qtx.dealloc %wire : !qtx.wire

    // Deallocates a qubit register
    qtx.dealloc %array : !qtx.wire_array<5>

    // Deallocates a qubit register
    qtx.dealloc %wire, %array : !qtx.wire, !qtx.wire_array<5>
    ```
  }];
  let arguments = (ins Variadic<AnyQType>:$targets);
  let results = (outs);
  let assemblyFormat = "$targets attr-dict `:` type($targets)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Wire array processing
//===----------------------------------------------------------------------===//

def qtx_ArrayCreateOp : QTXOp<"array_create"> {
  let summary = "create an array from a list of wires";
  let description = [{
    This operation creates a wire array from a list of wires provided as input.
    The input list must contain at least one wire.  Once the array is created,
    the original wires must not be used in further operations.

    Example:

    ```mlir
    %one_wire_array = array_create %q0 : !qtx.wire_array<1>

    %array = array_create %q0, %q1, %q2 : !qtx.wire_array<3>
    ```
  }];

  let arguments = (ins Variadic<WireType>:$wires);
  let results = (outs WireArrayType:$array);
  let assemblyFormat = "$wires attr-dict `:` qualified(type($array))";
  let hasVerifier = 1;
}

def qtx_ArraySplitOp : QTXOp<"array_split", [InferTypeOpInterface]> {
  let summary = "split an array into a list of wires";
  let description = [{
    This operation splits a wire array into a list of wires.  The input array
    must not contain any dead wire(s).  Once the array is splited, the original
    array must not be used in further operations.

    Example:

    ```mlir
    %q0, %q1, %q2 = array_split %q0 : !qtx.wire_array<3>
    ```
  }];

  let arguments = (ins WireArrayType:$array);
  let results = (outs Variadic<WireType>:$wires);
  let assemblyFormat = "$array attr-dict `:` qualified(type($array))";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InferTypeOpInterface
    //===------------------------------------------------------------------===//
    static mlir::LogicalResult
    inferReturnTypes(mlir::MLIRContext *context,
                     std::optional<::mlir::Location> location,
                     mlir::ValueRange operands, mlir::DictionaryAttr attributes,
                     mlir::RegionRange regions,
                     llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      Adaptor op(operands, attributes);
      auto type = dyn_cast<qtx::WireArrayType>(op.getArray().getType());
      assert(type && "array_split must receive a wire_array as operand");
      inferredReturnTypes.resize(type.getSize(), qtx::WireType::get(context));
      return mlir::success();
    }
  }];
  let hasVerifier = 1;
}

def qtx_ArrayBorrowOp : QTXOp<"array_borrow"> {
  let summary = "borrow wire(s) from an array";
  let description = [{
    This operation borrows one or more wires from an array.  It takes as input a
    variadic list of indices and an array of wires.

    Since we are modeling quantum circuits, we need to be extra careful when
    dealing with such an array.  In particular, we need to make sure that wires
    are not cloned.  Hence, the semantics of this operation is to borrow wires
    from the array, leaving a "hole" ("dead" wire value) behind.

    The operation results are:
      * A variadic list of wires, each corresponding to one specified index
      * A new value for the array, which will have a different type from the
        one taken as input.  This new type explicitly tell us how many wires
        were taken from the array.

    Example:  Suppose we have an array with 5 wires. `qtx.wire_array<5>`.  We
    can visualize the borrowing second wire in the following way:

    ```
      0    1    2    3    4
    .---..---..---..---..---.
    | a ||   || c || d || e | : qtx.wire_array<5, dead = 1>
    '---''-|-''---''---''---'   (an array with one dead position)
           |
           V
           new_b : qtx.wire
    ```

    Borrowing the same wire from the array must not be possible.  An analysis
    pass together with run time checks must guarantee this situation never
    occurs.

    Code examples:

    * Borrowing one wire:
    ```mlir
    // New array with 5 wires
    %w_a = alloc(5) : !qtx.wire_array<5>

    // Borrow the 0th wire
    %i = arith.constant 0 : index
    %w0, %w_a_1d, = qtx.array_borrow %i from %w_a
        : (!qtx.wire_array<5>) -> (!qtx.wire, !qtx.wire_array<5, dead = 1>)
    ```

    * We could also borrow two at the same time:
    ```mlir
    // New array with 5 wires
    %w_a = alloc(5) : !qtx.wire_array<5>

    // Borrow 0th and 1th wires
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %w0, %w1, %w_a_2d = qtx.array_borrow %i, %j from %w_a
        : (!qtx.wire_array<5>) -> (!qtx.wire, !qtx.wire, !qtx.wire_array<5, dead = 2>)
    ```
  }];

  let arguments = (ins Variadic<AnySignlessIntegerOrIndex>:$indices,
                       WireArrayType:$array);
  let results = (outs Variadic<WireType>:$wires, WireArrayType:$new_array);
  // The following builder simplify creating borrowing operations: we can
  // infer the resulting types from the operands.
  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$indices, "mlir::Value":$array), [{
      mlir::SmallVector<mlir::Type> types(indices.size(),
                                    $_builder.getType<WireType>());
      auto arrayType = dyn_cast<qtx::WireArrayType>(array.getType());
      unsigned size = arrayType.getSize();
      unsigned dead = arrayType.getDead() + indices.size();
      types.push_back($_builder.getType<WireArrayType>(size, dead));
      return build($_builder, $_state, types, indices, array);
    }]>
  ];
  let assemblyFormat = [{
    $indices `from` $array attr-dict `:`
    type($indices) `from` qualified(type($array)) `->` type(results)
  }];
  let hasVerifier = 1;
}

def qtx_ArrayYieldOp : QTXOp<"array_yield"> {
  let summary = "give borrowed wire(s) back to the array";
  let description = [{
    This operation yield wire(s) back to the original array.  It takes as input
    a variadic list of wires and the array.

    A borrwed wire must be yield back to its original array.  An analysis
    pass together with run time checks must guarantee this.

    Example:

    ```mlir
    // Suppose `%array` had a wire, `%q0`, borrwed.  We yield it back using:
    %new_array = array_yield %q0 to %array :
      (!qtx.wire, !qtx.wire_array<3, dead = 1>) -> (!qtx.wire_array<3>)
    ```
  }];

  let arguments = (ins Variadic<WireType>:$wires, WireArrayType:$array);
  let results = (outs WireArrayType:$new_array);
  // The following builder simplify creating yield operations: we can
  // infer the resulting types from the operands.
  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$wires, "mlir::Value":$array), [{
      auto arrayType = dyn_cast<qtx::WireArrayType>(array.getType());
      unsigned size = arrayType.getSize();
      unsigned dead = arrayType.getDead() - wires.size();
      mlir::Type resultType = $_builder.getType<WireArrayType>(size, dead);
      return build($_builder, $_state, resultType, wires, array);
    }]>
  ];
  let assemblyFormat = [{
    $wires `to` $array attr-dict
    `:` qualified(type($array)) `->` qualified(type($new_array))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Measurements and resets
//===----------------------------------------------------------------------===//

class Measurement<string mnemonic>
    : QTXOp<mnemonic, [
        AllMatch<["getTargets().getType()", "getNewTargets().getType()"],
                  "target input types match new target types (results)">
      ]> {
  let arguments = (ins
    Variadic<AnyQType>:$targets,
    OptionalAttr<StrAttr>:$registerName
  );
  let results = (outs
    AnyTypeOf<[I1, VectorOf<[I1]>]>:$bits,
    Variadic<AnyQType>:$new_targets);
  // The following builder simplify creating measurement operations: we can
  // infer the resulting types from the operands.  Note that in the case of a
  // wire array of size one, this builder will return a classical bit of type
  // `i1` instead of a vector of size one.
  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$targets,
                   CArg<"mlir::StringAttr", "nullptr">:$registerName), [{
      mlir::SmallVector<mlir::Type> types(targets.size() + 1);
      unsigned size = 0u;
      for (auto [i, type] : llvm::enumerate(targets.getType())) {
        if (auto arrayType = type.dyn_cast_or_null<qtx::WireArrayType>())
          size += arrayType.getSize();
        else
          size += 1;
        types[i + 1] = type;
      }
      if (size == 1)
        types[0] = $_builder.getI1Type();
      else
        types[0] = mlir::VectorType::get(size, $_builder.getI1Type());
      return build($_builder, $_state, types, targets, registerName);
    }]>
  ];
  let assemblyFormat = [{
    $targets attr-dict `:`
    type($targets) `->` `<` type($bits) `>` type($new_targets)
  }];
  let hasVerifier = 1;
}

def qtx_ResetOp : QTXOp<"reset", [InferTypeOpInterface,
     AllMatch<["getTargets().getType()", "getNewTargets().getType()"],
               "target input types match new target types (results)">
  ]> {
  let summary = "Reset the qubit to the |0> state";
  let description = [{
    The `qtx.reset` operation resets a qubit (or register of qubits) to the
    |0> (or |0..0>) state.

    Example:

    ```mlir
    %new_wire = qtx2.reset %wire : !qtx.wire
    ```
  }];

  let arguments = (ins Variadic<AnyQType>:$targets);
  let results = (outs Variadic<AnyQType>:$new_targets);
  let assemblyFormat = "$targets attr-dict `:` type($targets)";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InferTypeOpInterface
    //===------------------------------------------------------------------===//
    static mlir::LogicalResult
    inferReturnTypes(mlir::MLIRContext *context,
                     std::optional<::mlir::Location> location,
                     mlir::ValueRange operands, mlir::DictionaryAttr attributes,
                     mlir::RegionRange regions,
                     llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      auto types = operands.getTypes();
      inferredReturnTypes.append(types.begin(), types.end());
      return mlir::success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Base quantum instructions
//===----------------------------------------------------------------------===//

// The following classes factor out some class declarations common to quantum
// instruction operations.  For example, it enable us to easily modify the
// arguments, results and assembly format for all instructions in one place.

class Operator<string mnemonic, list<Trait> traits = []>
    : QTXOp<mnemonic,
        !listconcat(traits, [AttrSizedOperandSegments, InferTypeOpInterface,
                             OperatorInterface])
  > {
  let arguments = (ins
    UnitAttr:$is_adj,
    Variadic<AnyFloat>:$parameters,
    Variadic<WireType>:$controls,
    Variadic<WireType>:$targets,
    OptionalAttr<DenseBoolArrayAttr>:$negated_qubit_controls
  );
  let results = (outs Variadic<WireType>:$new_targets);
  // The following builders simplify creating quantum operations: since we know
  // that the numbers of result targets is equal to the number of input targets,
  // and that the results must be of `WireType`, we don't need the user to
  // provide the range of result types.
  let builders = [
    OpBuilder<(ins "bool":$is_adj, "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, parameters, controls,
                   targets, {});
    }]>,
    OpBuilder<(ins "bool":$is_adj, "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, {}, controls, targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$parameters, "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, parameters, controls,
                   targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, controls, targets);
    }]>,
    OpBuilder<(ins "bool":$is_adj, "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, {}, targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, targets);
    }]>
  ];
  let assemblyFormat = [{
    `` custom<Parameters>($is_adj, $parameters)
    (` ``[` $controls^ ( `neg` $negated_qubit_controls^ )? `]`)?
    $targets attr-dict `:`
    (`<` type($parameters)^ `>`)? (` ``[` type($controls)^ `]`)? type($targets)
  }];
  code OperatorExtraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InferTypeOpInterface
    //===------------------------------------------------------------------===//
    static mlir::LogicalResult
    inferReturnTypes(mlir::MLIRContext *context,
                     std::optional<::mlir::Location> location,
                     mlir::ValueRange operands, mlir::DictionaryAttr attributes,
                     mlir::RegionRange regions,
                     llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      Adaptor op(operands, attributes);
      auto types = op.getTargets().getTypes();
      inferredReturnTypes.append(types.begin(), types.end());
      return mlir::success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Common instructions
//===----------------------------------------------------------------------===//

include "cudaq/Optimizer/Dialect/Common/Ops.td"

//===----------------------------------------------------------------------===//
// Structure Operations
//===----------------------------------------------------------------------===//

include "cudaq/Optimizer/Dialect/QTX/QTXStructure.td"

#endif // CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_OPS
