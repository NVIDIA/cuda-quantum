/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QUAKE_OPS
#define CUDAQ_OPTIMIZER_DIALECT_QUAKE_OPS

//===----------------------------------------------------------------------===//
// High-level CUDA Quantum support
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "cudaq/Optimizer/Dialect/CC/CCTypes.td"
include "cudaq/Optimizer/Dialect/Common/Traits.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.td"

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class QuakeOp<string mnemonic, list<Trait> traits = []> :
    Op<QuakeDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def quake_AllocaOp : QuakeOp<"alloca", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Allocates a collection of qubits.";
  let description = [{
    The operation allocates either a single qubit or a register of qubits
    depending whether a size operand is provided.  The return value will be
    either a qubit, `!quake.qref`, or an vector of qubits, `!quake.qvec<...>`,
    that represents references to qubit(s).

    In Quake, all qubits must be explicitly deallocated within the _same_ scope
    of their allocation.  There is a pass that automatically adds deallocations.

    Examples:

    ```mlir
    // Allocate a single qubit
    %qubit = quake.alloca : !quake.qref

    // Allocate a qubit register with a size known at compilation time
    %qvec = quake.qalloca : !quake.qvec<4>

    // Allocate a qubit register with a size known at runtime time
    %qvec = quake.qalloca(%size : i32) : !quake.qvec<?>
    ```

    Note:  The parameter of the returned type is populated when the
    canonicalizer for this operation is run, and the value of the size is known.
    ```mlir
    %four = arith.constant 4 : i32
    %0 = quake.alloca(%four : i32) : !quake.qvec<4>
    ```
  }];

  let arguments = (ins Optional<AnySignlessInteger>:$size);
  let results = (outs AnyQType:$qref_or_vec);

  let builders = [
    OpBuilder<(ins ), [{
      return build($_builder, $_state, $_builder.getType<QRefType>(), {});
    }]>,
    OpBuilder<(ins "size_t":$size), [{
      return build($_builder, $_state, $_builder.getType<QVecType>(size), {});
    }]>
  ];

  let assemblyFormat = [{
    (`(` $size^ `:` type($size)`)`)? attr-dict `:` qualified(type($qref_or_vec))
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def quake_ConcatOp : QuakeOp<"concat", [Pure]> {
  let summary = "Construct a qvec from a list of other qref/qvec values.";
  let description = [{
    The `quake.concat` operation allows one to group a list of values of type
    QRef and/or QVec into a new QVec aggregate.
    ```
    %veq = quake.concat %r1, %v1, %r2 : (!quake.qref, !quake.qvec<?>, !quake.qref) -> !quake.qvec<?>
    ```
  }];

  let arguments = (ins Variadic<AnyQType>:$qbits);
  let results = (outs QVecType);

  let assemblyFormat = [{
    $qbits attr-dict `:` functional-type(operands, results)
  }];
}

// A ComputeActionOp is marked as having memory effects to prevent the MLIR
// optimizer from eliding it as dead code. It will be transformed into a series
// of CallOps.
def quake_ComputeActionOp : QuakeOp<"compute_action",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Captures the compute/action/uncompute high-level idiom.";
  let description = [{
    CUDA Quantum supports the high-level compute/action/uncompute idiom by
    providing a custom template function (class) that takes pure kernels (a
    callable like a λ) as arguments. This operation captures uses of the idiom
    and can be systematically expanded into a quantum circuit via successive
    transformations.
  }];

  // FIXME: The compute and action can be other sorts of Callable instances and
  // need not be inline λ expressions.
  let arguments = (ins
    UnitAttr:$is_dagger,
    cc_LambdaType:$compute,
    cc_LambdaType:$action
  );
  
  let assemblyFormat = [{
    (`<` `dag` $is_dagger^ `>`)? $compute `,` $action attr-dict
    `:` qualified(type(operands))
  }];
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

def quake_DeallocOp : QuakeOp<"dealloc"> {
  let summary = "Deallocates a collection of qubits.";
  let description = [{
    This operation is used to deallocate a collection of qubits, which is 
    represented in quake as a `quake.qvec<?>` parameterized type. 
    Quake DeallocOps are added using the -quake-add-deallocs pass. 
    ```
    %four = arith.constant 4 : i32
    %0 = quake.alloca(%four : i32) : !quake.qvec<?>
    quake.dealloc(%0)
    ```
  }];
  let arguments = (ins
    Arg<AnyQType, "qubit reference (or vector) to deallocate",
    [MemFree]>:$qreg_or_vec);
  let assemblyFormat = [{
    `(` $qreg_or_vec `:` qualified(type($qreg_or_vec)) `)` attr-dict 
  }];
}

//===----------------------------------------------------------------------===//
// QVec, QRef manipulation
//===----------------------------------------------------------------------===//

def quake_QExtractOp : QuakeOp<"qextract", [Pure]> {
  let summary = "Get a reference to a single qubit.";
  let description = [{
    The `quake.qextract` operation extracts qubit references from a vector.
    ```
    %zero = arith.constant 0 : i32
    %qr = quake.qextract %qv[%zero] : !quake.qvec<?>[i32] -> !quake.qref
    ```
    The produced SSA value is then used for all operations on the 0'th qubit
    of the `%qv` vector of qubits.
  }];
  let arguments = (ins
    QVecType:$qvec,
    AnySignlessIntegerOrIndex:$index
  );
  let builders = [
    OpBuilder<(ins "mlir::Value":$qvec, "mlir::Value":$index), [{
      return build($_builder, $_state, $_builder.getType<QRefType>(), qvec,
                   index);
    }]>,
  ];
  let results = (outs QRefType:$qref);
  let assemblyFormat = [{
    $qvec `[` $index `]` attr-dict
    `:`  qualified(type($qvec)) `[` type($index) `]` `->` qualified(type($qref))
  }];
  let hasFolder = 1;
}

def quake_RelaxSizeOp : QuakeOp<"relax_size", [Pure]> {
  let summary = "Relax the constant size on a !qvec to be unknown.";
  let description = [{
    At times, the IR needs to forget the length of a `!qvec<n>` and demote it
    to a `!qvec<?>` in order to preserve a strongly-typed IR.
  }];

  let arguments = (ins QVecType:$inputVec);
  let results = (outs QVecType);

  let assemblyFormat = [{
    $inputVec `:` functional-type(operands, results) attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def quake_SubVecOp : QuakeOp<"subvec", [Pure]> {
  let summary = "Extract a subvector from a qvec reference value.";
  let description = [{
    The `quake.subvec` operation gets a subvector reference from a qvec.
    ```
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %qr = quake.subvec %qv, %zero, %four : (!quake.qvec<?>, i32, i32) -> !quake.qvec<5>
    ```
    The produced SSA value is then used for all operations on the zeroth through
    the fourth (inclusive, total of 5) qubit of the `%qv` vector of qubits. The
    returned vector is not a constructor and is non-owning. The subvector of
    qubits are the same qubits as in the original `%qv`. (This is identical to
    the semantics of the `qextract` op.)
  }];

  let arguments = (ins
    QVecType:$qvec, 
    AnySignlessIntegerOrIndex:$low,
    AnySignlessIntegerOrIndex:$high
  );
  let results = (outs QVecType:$qsub);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// QVecSizeOp
//===----------------------------------------------------------------------===//

def quake_QVecSizeOp : QuakeOp<"qvec_size", [Pure]> {
  let summary = "Get size of a qvec.";
  let description = [{
    Get the size of a qvec. 
    ```
    %0 = quake.alloca(%four : i32) : !quake.qvec<4>
    %s = quake.qvec_size (%0 : !quake.qvec<?>) : i64
    ```
  }];

  let arguments = (ins QVecType:$qvec);
  let results = (outs AnySignlessIntegerOrIndex:$size);

  let assemblyFormat = [{
    $qvec attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Measurements and resets
//===----------------------------------------------------------------------===//

class Measurement<string mnemonic> : QuakeOp<mnemonic> {
  let arguments = (ins
    Arg<Variadic<AnyQType>,
      "qubit reference(s) (or vector(s)) to measure",
      [MemRead, MemWrite]>:$targets,
    OptionalAttr<StrAttr>:$registerName
  );
  let results = (outs AnyTypeOf<[I1, StdvecOf<[I1]>]>:$bits);
  let assemblyFormat = [{
    `(` $targets `:` type($targets) `)` attr-dict `:` type($bits)
  }];
  let hasVerifier = 1;
}

def quake_ResetOp : QuakeOp<"reset"> {
  let summary = "Reset the qubit (qreg) to the |0> (|0..0>) state";
  let description = [{
    The `quake.reset` operation resets a qubit (qreg) to the |0> (|0..0>) state.

    Example:
    ```mlir
    quake.reset(%0 : !quake.qref)
    ```
  }];

  let arguments = (ins
    Arg<AnyQType,
      "qubit reference (or vector) to reset", [MemRead, MemWrite]>:$targets
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $targets `:` type($targets) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Application
//===----------------------------------------------------------------------===//

def quake_ApplyOp : QuakeOp<"apply",
    [AttrSizedOperandSegments, CallOpInterface]> {
  let summary = "Abstract application of a function in Quake.";
  let description = [{
    User-defined kernels define both predicated and unpredicated functions.
    The predicated form is implicitly defined. To simplify lowering, the
    unpredicated function may be defined while an ApplyOp may use the
    implied predicated function. A subsequent pass will then instantiate both
    the unpredicated and predicated variants.
  }];

  let arguments = (ins
    SymbolRefAttr:$callee,
    UnitAttr:$is_adj,
    Variadic<AnyQType>:$controls,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    (`<` `adj` $is_adj^ `>`)? $callee (`[` $controls^ `:` type($controls) `]`)?
      $args attr-dict `:` functional-type($args,results)
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getCalleeAttrNameStr() { return "callee"; }

    mlir::FunctionType getFunctionType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      if (getControls().empty())
        return {operand_begin(), operand_end()};
      return {getArgs().begin(), getArgs().end()};
    }

    bool applyToVariant() {
      return getIsAdj() || !getControls().empty();
    }

    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>(getCalleeAttrName());
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
    : QuakeOp<mnemonic,
        !listconcat([AttrSizedOperandSegments, OperatorInterface], traits)> {

  let arguments = (ins
    UnitAttr:$is_adj,
    Variadic<AnyFloat>:$parameters,
    Arg<Variadic<AnyQType>,
      "qubit reference(s) (or vector(s)) used as control(s)",
      [MemRead]>:$controls,
    Arg<Variadic<QRefType>,
      "qubit reference(s) to target", [MemRead, MemWrite]>:$targets,
    OptionalAttr<DenseBoolArrayAttr>:$negated_qubit_controls
  );
  let builders = [
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, parameters, controls,
                   targets, {});
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, parameters, controls,
                   targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, mlir::ValueRange{}, controls, targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, mlir::ValueRange{}, targets);
    }]>
  ];
  let assemblyFormat = [{
    (`<` `adj` $is_adj^ `>`)?
    (`[` $controls^ (`neg` $negated_qubit_controls^ )? `:` type($controls) `]`)?
    custom<Parameters>($parameters, type($parameters))
    `(` $targets `)` attr-dict
  }];
  code OperatorExtraClassDeclaration = "";
}

//===----------------------------------------------------------------------===//
// Common instructions
//===----------------------------------------------------------------------===//

include "cudaq/Optimizer/Dialect/Common/Ops.td"

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_OPS
