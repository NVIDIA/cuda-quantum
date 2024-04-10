/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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
  let summary = "Allocates a reference or collection of references to wires.";
  let description = [{
    The `alloca` operation allocates either a single quantum reference or a
    vector of quantum references. The size of the vector may be provided either
    statically as part of the type or dynamically as an integer-like argument,
    `size`. The return value will be a quantum reference, type `!quake.ref`,
    or a vector of such, type `!quake.veq<N>`.

    All references are assumed to be initialized to the value `|0>` initially.
    See also the `null_wire` op.

    The `QuakeAddDeallocs` and `UnwindLowering` passes will insert deallocation
    ops for the scopes in which allocations appear automatically. This is
    helpful for generating code for targets such as QIR, which require
    allocation/deallocation pairs.

    Examples:
    ```mlir
      // Allocate a single qubit
      %qubit = quake.alloca !quake.ref

      // Allocate a qubit register with a size known at compilation time
      %veq = quake.alloca !quake.veq<4> {name = "quantum"}

      // Allocate a qubit register with a size known at runtime time
      %veq = quake.alloca(%size : i32) !quake.veq<?>
    ```

    Canonicalization for this op will fold constant vector sizes directly into
    the type.

    See DeallocOp.
  }];

  let arguments = (ins
    Optional<AnySignlessInteger>:$size
  );
  let results = (outs
    AnyRefType:$ref_or_vec
  );

  let builders = [
    OpBuilder<(ins ), [{
      return build($_builder, $_state, $_builder.getType<RefType>(), {});
    }]>,
    OpBuilder<(ins "size_t":$size), [{
      return build($_builder, $_state, $_builder.getType<VeqType>(size), {});
    }]>,
    OpBuilder<(ins "mlir::Type":$ty), [{
      return build($_builder, $_state, ty, {});
    }]>
  ];

  let assemblyFormat = [{
    qualified(type($ref_or_vec)) (`[` $size^ `:` type($size) `]`)? attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasInitializedState() {
      auto *self = getOperation();
      return self->hasOneUse() &&
        mlir::isa<quake::InitializeStateOp>(*self->getUsers().begin());
    }

    quake::InitializeStateOp getInitializedState();
  }];
}

def quake_ConcatOp : QuakeOp<"concat", [Pure]> {
  let summary = "Construct a veq from a list of other ref/veq values.";
  let description = [{
    The `concat` operation allows one to concatenate a list of SSA-values of
    either type Ref or Veq into a new Veq vector.

    Example:
    ```mlir
      %veq = quake.concat %r1, %v1, %r2 : (!quake.ref, !quake.veq<?>,
                                           !quake.ref) -> !quake.veq<?>
    ```
  }];

  let arguments = (ins Variadic<AnyRefType>:$qbits);
  let results = (outs VeqType);

  let assemblyFormat = [{
    $qbits attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

// A ComputeActionOp will be transformed into a series of CallOps.
def quake_ComputeActionOp : QuakeOp<"compute_action"> {
  let summary = "Captures the compute/action/uncompute high-level idiom.";
  let description = [{
    CUDA Quantum supports the high-level compute, action, uncompute idiom by
    providing a custom template function (class) that takes pure kernels (a
    callable like a λ) as arguments. This operation captures uses of the idiom
    and can be systematically expanded into a quantum circuit via successive
    transformations.

    The `is_dagger` attribute can be used to "reverse" this idiom to one of
    uncompute, action, compute.

    The uncompute step is generated automatically by generating the adjoint of
    the compute kernel.
  }];

  let arguments = (ins
    UnitAttr:$is_dagger,
    cc_CallableType:$compute,
    cc_CallableType:$action
  );

  let assemblyFormat = [{
    (`<` `dag` $is_dagger^ `>`)? $compute `,` $action `:`
      qualified(type(operands)) attr-dict
  }];
}

def quake_ToControlOp : QuakeOp<"to_ctrl", [Pure]> {
  let summary = "Convert a wire value to a control value.";
  let description = [{
    This operation makes the conversion of a wire value to a control value
    explicit in the quake IR. These values have different semantics in the IR.
    This op ensures these semantics via the type system.

    A value of type control is (nearly) an SSA-value. Once defined, via the
    `to_ctrl` operation, it can be used as an argument to other operations.
    These uses are qualified. They must be in control argument positions and
    these operations must dominate a `from_ctrl` operation that returns the
    control qubit back to a wire. The operand value and result value of a
    `to_ctrl` may NOT be used as arguments to the same operation.
  }];

  let arguments = (ins WireType:$qubit);
  let results = (outs ControlType);

  let assemblyFormat = [{
    $qubit `:` functional-type(operands, results) attr-dict
  }];
}

def quake_FromControlOp : QuakeOp<"from_ctrl", [Pure]> {
  let summary = "Convert a control value to a wire value.";
  let description = [{
    This operation makes the conversion of a control value to a wire value
    explicit in the quake IR. These values have different semantics in the IR.
    This op ensures these semantics via the type system.
  }];

  let arguments = (ins ControlType:$ctrlbit);
  let results = (outs WireType);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $ctrlbit `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

def quake_DeallocOp : QuakeOp<"dealloc"> {
  let summary = "Deallocates a collection of qubits.";
  let description = [{
    The `dealloc` operation deallocates a quantum reference. The deallocation
    can be a single quantum reference, `!quake.ref`, or a vector of quantum
    references, `!quake.veq<N>`.

    Deallocations are automatically inserted by the `AddDeallocs` and
    `UnwindLowering` passes.

    Example:
    ```mlir
      %1 = quake.alloca !quake.veq<4>
      ...
      quake.dealloc %1 : !quake.veq<4>
    ```

    See AllocaOp.
  }];

  let arguments = (ins
    Arg<AnyRefType, "qubit reference (or vector) to deallocate",
    [MemFree]>:$reference
  );

  let assemblyFormat = [{
    $reference `:` qualified(type($reference)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Veq, Ref manipulation
//===----------------------------------------------------------------------===//

def quake_ExtractRefOp : QuakeOp<"extract_ref", [Pure]> {
  let summary = "Extract a quantum reference from a quantum vector.";
  let description = [{
    The `extract_ref` operation extracts a quantum reference from a vector of
    quantum references.

    The following example extracts the quantum reference at position 0 from a
    vector of references. The vector, in this case, has unknown size.

    Example:
    ```mlir
      %zero = arith.constant 0 : i32
      %qr = quake.extract_ref %qv[%zero] : (!quake.veq<?>, i32) -> !quake.ref
    ```
  }];

  let arguments = (ins
    VeqType:$veq,
    Optional<AnySignlessIntegerOrIndex>:$index,
    I64Attr:$rawIndex
  );
  let results = (outs RefType:$ref);

  let builders = [
    OpBuilder<(ins "mlir::Value":$veq, "mlir::Value":$index,
                   "mlir::IntegerAttr":$rawIndex), [{
      return build($_builder, $_state, $_builder.getType<RefType>(), veq,
                   index, rawIndex);
    }]>,
    OpBuilder<(ins "mlir::Value":$veq, "mlir::Value":$index), [{
      return build($_builder, $_state, $_builder.getType<RefType>(), veq,
                   index, ExtractRefOp::kDynamicIndex);
    }]>,
    OpBuilder<(ins "mlir::Value":$veq, "std::size_t":$rawIndex), [{
      auto i64Ty = $_builder.getI64Type();
      return build($_builder, $_state, $_builder.getType<RefType>(), veq,
                   mlir::Value{}, mlir::IntegerAttr::get(i64Ty, rawIndex));
    }]>
  ];

  let assemblyFormat = [{
    $veq `[` custom<RawIndex>($index, $rawIndex) `]` `:`
      functional-type(operands, results) attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static constexpr std::size_t kDynamicIndex =
      std::numeric_limits<std::size_t>::max();

    bool hasConstantIndex() { return !getIndex(); }
    std::size_t getConstantIndex() { return getRawIndex(); }
  }];
}

def quake_InitializeStateOp : QuakeOp<"init_state", [MemoryEffects<[MemAlloc,
    MemWrite]>]> {
  let summary = "Initialize the quantum state to a specific complex vector.";
  let description = [{
    Given a !cc.ptr pointing to a complex data array of size 2**N, where N is
    the number of qubits in the targets operand, initialize the state of those
    target qubits to the provided state vector. This operation returns a new
    quake.veq instance. There should be no other uses of the input veq value,
    \em{targets}, that was allocated. This supports a RAII (resource allocation
    is initialization) semantics on the qubits in the vector.
  }];

  let arguments = (ins
    VeqType:$targets,
    cc_PointerType:$state
  );
  let results = (outs VeqType);

  let assemblyFormat = [{
    $targets `,` $state `:` functional-type(operands, results) attr-dict
  }];

  let hasVerifier = 1;
}

def quake_RelaxSizeOp : QuakeOp<"relax_size", [Pure]> {
  let summary = "Relax the constant size on a !veq to be unknown.";
  let description = [{
    At times, the IR needs to forget the length of an SSA-value of type
    `!quake.veq<N>` and demote it to type `!quake.veq<?>` where the size is
    said to be unknown. This demotion is required to preserve a valid,
    strongly-typed IR.

    Example:
    ```mlir
      %uqv = quake.relax_size %qv : (!quake.veq<4>) -> !quake.veq<?>
    ```
  }];

  let arguments = (ins VeqType:$inputVec);
  let results = (outs VeqType);

  let assemblyFormat = [{
    $inputVec `:` functional-type(operands, results) attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def quake_SubVeqOp : QuakeOp<"subveq", [Pure]> {
  let summary = "Extract a subvector from a veq reference value.";
  let description = [{
    The `subveq` operation returns a subvector of references, type
    `!quake.veq<N>` from a vector of references, type `!quake.veq<M>`, where
    `M >= N`.

    In the following example, the operation produces an SSA-value with 5
    references. These references may be indexed from 0 to 4 via `%qr` and are
    the same references as those from 2 to 6 indexed via `%qv`. Specifically,
    the returned vector, `%qr`, is not a constructor and does not own the
    references; it simply makes a copy as a new (shorter) vector. Therefore,
    subvectors need never be deallocated.

    Example:
    ```mlir
      %0 = arith.constant 2 : i32
      %1 = arith.constant 6 : i32
      %qr = quake.subveq %qv, %0, %1 : (!quake.veq<?>, i32, i32) ->
                                        !quake.veq<5>
    ```
  }];

  let arguments = (ins
    VeqType:$veq,
    AnySignlessIntegerOrIndex:$low,
    AnySignlessIntegerOrIndex:$high
  );
  let results = (outs VeqType:$qsub);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// VeqSizeOp
//===----------------------------------------------------------------------===//

def quake_VeqSizeOp : QuakeOp<"veq_size", [Pure]> {
  let summary = "Return the size of a veq.";
  let description = [{
    Returns the size of a value of type `!quake.veq<n>`. If the vector has a
    static size, the static size is returned (effectively as a constant). If
    the size of the vector is dynamic, the size value will be an SSA-value.

    Examples:
    ```mlir
      %0 = quake.alloca !quake.veq<4>
      // %1 will be 4 with canonicalization.
      %1 = quake.veq_size %0 : (!quake.veq<4>) -> i64
      
      %2 = ... : !quake.veq<?>
      // %3 may not be computed until runtime.
      %3 = quake.veq_size %2 : (!quake.veq<?>) -> i64
    ```
  }];

  let arguments = (ins VeqType:$veq);
  let results = (outs AnySignlessIntegerOrIndex:$size);

  let assemblyFormat = [{
    $veq `:` functional-type(operands, results) attr-dict
  }];

  let hasCanonicalizer = 1;
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
    OptionalAttr<SymbolRefAttr>:$callee,
    Variadic<cc_CallableType>:$indirect_callee, // must be 0 or 1 element
    UnitAttr:$is_adj,
    Variadic<AnyQType>:$controls,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>);

  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$retTy,
                   "mlir::SymbolRefAttr":$callee,
                   "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$args), [{
      return build($_builder, $_state, retTy, callee, mlir::ValueRange{},
                   is_adj, controls, args);
    }]>,
    OpBuilder<(ins "mlir::TypeRange":$retTy,
                   "mlir::SymbolRefAttr":$callee,
                   "bool":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$args), [{
      return build($_builder, $_state, retTy, callee, mlir::ValueRange{},
                   is_adj, controls, args);
    }]>,
    OpBuilder<(ins "mlir::TypeRange":$retTy,
                   "mlir::Value":$callable,
                   "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$args), [{
      return build($_builder, $_state, retTy, mlir::SymbolRefAttr{},
                   mlir::ValueRange{callable}, is_adj, controls, args);
    }]>,
    OpBuilder<(ins "mlir::TypeRange":$retTy,
                   "mlir::Value":$callable,
                   "bool":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$args), [{
      return build($_builder, $_state, retTy, mlir::SymbolRefAttr{},
                   mlir::ValueRange{callable}, is_adj, controls, args);
    }]>
  ];

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
// Memory and register conversion instructions: These operations are useful for
// intermediate conversions between memory-SSA and value-SSA semantics and vice
// versa of the IR. They mainly exist during the conversion process.
//===----------------------------------------------------------------------===//

def quake_UnwrapOp : QuakeOp<"unwrap"> {
  let summary = "Unwrap a reference to a wire and return the wire value.";
  let description = [{
    A quantum reference is an SSA-value that is associated with a volatile
    quantum wire. The unwrap operation allows conversion from the reference
    value semantics (memory SSA) to the volatile quantum wire value semantics
    when/as desired. The binding of a reference value corresponds to a
    particular data flow of volatile quantum wire values.

    Unwrap and wrap operations should (typically) form pairs as in the following
    example.

    ```mlir
      %0 = ... : !quake.ref
      %1 = quake.unwrap %0 : (!quake.ref) -> !quake.wire
      %2 = quake.rx (%dbl) %1 : (f64, !quake.wire) -> !quake.wire
      quake.wrap %2 to %0 : !quake.wire, !quake.ref
    ```
  }];

  let arguments = (ins Arg<RefType,"",[MemRead]>:$ref_value);
  let results = (outs WireType);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $ref_value `:` functional-type(operands, results) attr-dict
  }];
}

def quake_WrapOp : QuakeOp<"wrap"> {
  let summary = "Wrap a wire value with its reference.";
  let description = [{
    Each data flow of a volatile wire in a quantum circuit may be associated
    and identified with an invariant and unique quantum reference value of type
    `!ref`. The wrap operation provides a means of returning a quantum value,
    a wire, back to the reference value domain.
  }];

  let arguments = (ins
    WireType:$wire_value,
    Arg<RefType,"",[MemWrite]>:$ref_value
  );
  let assemblyFormat = [{
    $wire_value `to` $ref_value `:` qualified(type(operands)) attr-dict
  }];

  let hasCanonicalizer = 1;
}

def quake_NullWireOp : QuakeOp<"null_wire"> {
  let summary = "Initial state of a wire.";
  let description = [{
    |0> - the initial state of a wire when first constructed. A wire is assumed
    to be defined in the |0> state as its initial state.
  }];

  let results = (outs WireType);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Reset
//===----------------------------------------------------------------------===//

def quake_ResetOp : QuakeOp<"reset", [QuantumGate,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Reset the wire to the |0> (|0..0>) state.";
  let description = [{
    The `quake.reset` operation resets a wire to the |0> (|0..0>) state. It
    may take an argument that is either a reference to a wire, type `!ref`,
    or a wire, type `!wire`.

    Example:
    ```mlir
      quake.reset %0 : (!quake.ref) -> ()
      %2 = quake.reset %1 : (!quake.wire) -> !quake.wire
    ```
  }];

  let arguments = (ins
    AnyQTargetType:$targets
  );
  let results = (outs
    Variadic<WireType>:$wires
  );
  let hasVerifier = 1;

  let assemblyFormat = [{
     $targets `:` functional-type(operands, results) attr-dict
  }];

  let extraClassDeclaration = [{
    void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::
        EffectInstance<mlir::MemoryEffects::Effect>> &effects) {
      quake::getResetEffectsImpl(effects, getTargets());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Sink
//===----------------------------------------------------------------------===//

def quake_SinkOp : QuakeOp<"sink"> {
  let summary = "Sink for a qubit that is used but not measured.";
  let description = [{
    The `quake.sink` operation is used to mark a particular wire in the value
    semantics as used at the end of a circuit. Typically a wire is measured, but
    in some cases the wire needs to be marked as if it were measured even though
    a measurement is not desired.

    This op is similar to the dealloc op in the reference semantics.

    Example:
    ```mlir
      quake.sink %0 : !quake.wire
    ```
  }];

  let arguments = (ins 
    Arg<WireType,"wire to sink",[MemRead, MemWrite]>:$target
  );
  let assemblyFormat = [{
     $target `:` qualified(type(operands)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Measurements
//===----------------------------------------------------------------------===//

class Measurement<string mnemonic> : QuakeOp<mnemonic, [MeasurementInterface,
    QuantumMeasure, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    Variadic<AnyQTargetType>:$targets,
    OptionalAttr<StrAttr>:$registerName
  );
  let results = (outs
    AnyTypeOf<[MeasureType, StdvecOf<[MeasureType]>]>:$measOut,
    Variadic<WireType>:$wires
  );

  let assemblyFormat = [{
    $targets (`name` $registerName^)? `:` functional-type(operands, results)
      attr-dict
  }];

  code OpBaseDeclaration = [{
    void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
      mlir::MemoryEffects::Effect>> &effects) {
      quake::getMeasurementEffectsImpl(effects, getTargets());
    }
  }];

  let hasVerifier = 1;
}

def MxOp : Measurement<"mx"> {
  let summary = "Measurement along the x-axis";
  let description = [{
    The `mx` operation measures the state of qubits into classical bits
    represented by a `i1` (or a vector of `i1`), along the x-axis.

    The state of the qubits is collapsed into one of the computational basis
    states, i.e., either |0> or |1>. A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation. Another option is to deallocate the qubit using `dealloc`.
  }];
  let extraClassDeclaration = OpBaseDeclaration;
}

def MyOp : Measurement<"my"> {
  let summary = "Measurement along the y-axis";
  let description = [{
    The `my` operation measures the state of qubits into classical bits
    represented by a `i1` (or a vector of `i1`), along the y-axis.

    The state of the qubit is collapsed into one of the computational basis
    states, i.e., either |0> or |1>. A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation. Another option is to deallocate the qubit using `dealloc`.
  }];
  let extraClassDeclaration = OpBaseDeclaration;
}

def MzOp : Measurement<"mz"> {
  let summary = "Measurement along the z-axis";
  let description = [{
    The `mz` operation measures the state of qubits into a classical bits
    represented by a `i1` (or a vector of `i1`), along the z-axis---the
    so-called computational basis.

    The state of the qubit is collapsed into one of the computational basis
    states, i.e., either |0> or |1>. A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation. Another option is to deallocate the qubit using `dealloc`.
  }];
  let extraClassDeclaration = OpBaseDeclaration;
}

def DiscriminateOp : QuakeOp<"discriminate", [Pure]> {
  let summary = "Converts a measurement to a classical integral value.";
  let description = [{
    Quake's measurement operators return a value of type `!quake.measure`. The
    discriminate operation converts a value of type measure to a classical
    integral value. This value is typically an `i1` type, but might be `i2` for
    qutrits, or even an `i8` for general qudits.

    While a measurement of a wire changes/corrupts the state of the wire, the
    model maintains that a `!quake.measure` value is non-volatile. Therefore,
    multiple applications of discriminate on the same `!quake.measure` value
    will yield the same result value for a given result type.
  }];

  let arguments = (ins
    AnyTypeOf<[MeasureType, StdvecOf<[MeasureType]>]>:$measurement
  );
  let results = (outs
    AnyTypeOf<[AnySignlessInteger, StdvecOf<[AnySignlessInteger]>]>
  );

  let assemblyFormat = [{
    $measurement `:` functional-type(operands, results) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Quantum gates
//===----------------------------------------------------------------------===//

class QuakeOperator<string mnemonic, list<Trait> traits = []>
    : QuakeOp<mnemonic,
        !listconcat([QuantumGate, AttrSizedOperandSegments, OperatorInterface,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>], traits)> {

  let arguments = (ins
    UnitAttr:$is_adj,
    Variadic<AnyFloat>:$parameters,
    Variadic<AnyQType>:$controls,
    Variadic<AnyQTargetType>:$targets,
    OptionalAttr<DenseBoolArrayAttr>:$negated_qubit_controls
  );
  let results = (outs
    Variadic<WireType>:$wires
  );

  let builders = [
    OpBuilder<(ins "mlir::UnitAttr":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::DenseBoolArrayAttr":$negates), [{
      return build($_builder, $_state, mlir::TypeRange{}, is_adj, parameters,
                   controls, targets, negates);
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets,
                   "mlir::DenseBoolArrayAttr":$negates), [{
      return build($_builder, $_state, mlir::TypeRange{}, is_adj, parameters,
                   controls, targets, negates);
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, parameters, controls, targets,
                   {});
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$parameters,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, parameters, controls,
                   targets);
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, mlir::ValueRange{}, controls,
                   targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$controls,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, controls, targets);
    }]>,
    OpBuilder<(ins "bool":$is_adj,
                   "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, is_adj, mlir::ValueRange{}, targets);
    }]>,
    OpBuilder<(ins "mlir::ValueRange":$targets), [{
      return build($_builder, $_state, /*is_adj=*/false, targets);
    }]>
  ];

  let assemblyFormat = [{
    ( `<` `adj` $is_adj^ `>` )? ( `(` $parameters^ `)` )?
    (`[` $controls^ (`neg` $negated_qubit_controls^ )? `]`)?
    $targets `:` functional-type(operands, results) attr-dict
  }];

  let hasVerifier = 1;

  code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // MemoryEffects interface
    //===------------------------------------------------------------------===//

    void getEffectsImpl(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
      mlir::MemoryEffects::Effect>> &effects) {
      quake::getOperatorEffectsImpl(effects, getControls(), getTargets());
    }

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    bool isAdj() { return getIsAdj(); }

    //===------------------------------------------------------------------===//
    // Element Access
    //===------------------------------------------------------------------===//

    mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

    mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }

    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//

    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    void getOperatorMatrix(Matrix &matrix);
  }];
}

class OneTargetOp<string mnemonic, list<Trait> traits = []> :
    QuakeOperator<mnemonic, !listconcat([NumParameters<0>, NumTargets<1>],
                traits)>;

class OneTargetParamOp<string mnemonic, list<Trait> traits = []> :
    QuakeOperator<mnemonic, !listconcat([NumParameters<1>, NumTargets<1>],
                traits)>;

class TwoTargetOp<string mnemonic, list<Trait> traits = []> :
    QuakeOperator<mnemonic, !listconcat([NumParameters<0>, NumTargets<2>],
                traits)>;

//===----------------------------------------------------------------------===//
// Unitary quantum transformations
//===----------------------------------------------------------------------===//

def ExpPauliOp : QuakeOp<"exp_pauli"> {
  let summary = "General Pauli tensor product rotation";
  let description = [{
    This operation affects a general Pauli tensor product rotation on 
    the input qubits. The number of Pauli characters in the input Pauli word 
    string must equal the number of qubits in the veq. Mathematically, this
    operation applies exp(i theta P) where P is a general Pauli tensor product. 
  }];
  
  let arguments = (ins
    AnyFloat:$parameter,
    VeqType:$qubits,
    Optional<AnyTypeOf<[cc_PointerType, cc_StdVectorType, cc_CharSpanType]>>:$pauli,
    OptionalAttr<StrAttr>:$constantPauli
  );
  let results = (outs);

  let assemblyFormat = [{
    $parameter `,` $qubits `,` $pauli `:` functional-type(operands, results)
      attr-dict
  }];
}

def HOp : OneTargetOp<"h", [Hermitian]> {
  let summary = "Hadamard operation";
  let description = [{
    This is a π rotation about the X+Z axis, and has the effect of changing the
    computation basis from |0> (|1>) to |+> (|->) and vice-versa, meaning that
    it enables one to create a superposition of basis states.

    Matrix representation:
    ```
    H = (1 / sqrt(2)) * | 1   1 |
                        | 1  -1 |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ H ├─
     └───┘
    ```
  }];
}

def PhasedRxOp : QuakeOperator<"phased_rx",
                   [NumParameters<2>, NumTargets<1>, Rotation]> {
  let summary = "an arbitrary rotation θ around the cos(φ)x + sin(φ)y axis";
  let description = [{
    Matrix representation:
    ```
    PhasedRx(θ,φ) = |        cos(θ/2)        -iexp(-iφ) * sin(θ/2) |
                    |  -iexp(iφ) * sin(θ/2)         cos(θ/2)       |
    ```

    Circuit symbol:
    ```
     ┌───────────────┐
    ─┤ PhasedRx(θ,φ) ├─
     └───────────────┘
    ```
  }];
}

def R1Op : OneTargetParamOp<"r1", [Rotation]> {
  let summary = "an arbitrary rotation about the |1> state";
  let description = [{
    Matrix representation:
    ```
    R1(λ) = | 1     0    |
            | 0  exp(iλ) |
    ```

    Circuit symbol:
    ```
     ┌───────┐
    ─┤ R1(λ) ├─
     └───────┘
    ```
  }];
}

def RxOp : OneTargetParamOp<"rx", [Rotation]> {
  let summary = "an arbitrary rotation about the X axis";
  let description = [{
    Matrix representation:
    ```
    Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
            | -isin(θ/2)  cos(θ/2)  |
    ```

    Circuit symbol:
    ```
     ┌───────┐
    ─┤ Rx(θ) ├─
     └───────┘
    ```
  }];
  let hasCanonicalizer = 1;
}

def RyOp : OneTargetParamOp<"ry", [Rotation]> {
  let summary = "an arbitrary rotation about the Y axis";
  let description = [{
    Matrix representation:
    ```
    Ry(θ) = | cos(θ/2)  -sin(θ/2) |
            | sin(θ/2)   cos(θ/2) |
    ```

    Circuit symbol:
    ```
     ┌───────┐
    ─┤ Ry(θ) ├─
     └───────┘
    ```
  }];
  let hasCanonicalizer = 1;
}

def RzOp : OneTargetParamOp<"rz", [Rotation]> {
  let summary = "an arbitrary rotation about the Z axis";
  let description = [{
    Matrix representation:
    ```
    Rz(λ) = | exp(-iλ/2)      0     |
            |     0       exp(iλ/2) |
    ```

    Circuit symbol:
    ```
     ┌───────┐
    ─┤ Rz(λ) ├─
     └───────┘
    ```
  }];
  let hasCanonicalizer = 1;
}

def SOp : OneTargetOp<"s"> {
  let summary = "S operation (aka, P or Sqrt(Z))";
  let description = [{
    This operation applies to its target a π/2 rotation about the Z axis.

    Matrix representation:
    ```
    S = | 1   0 |
        | 0   i |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ S ├─
     └───┘
    ```
  }];
}

def SwapOp : TwoTargetOp<"swap", [Hermitian]> {
  let summary = "Swap operation";
  let description = [{
    This operation swaps the states of two qubits.

    Matrix representation:
    ```
    Swap = | 1 0 0 0 |
           | 0 0 1 0 |
           | 0 1 0 0 |
           | 0 0 0 1 |
    ```

    Circuit symbol:
    ```
    ─X─
     │
    ─X─
    ```
  }];
}

def TOp : OneTargetOp<"t"> {
  let summary = "T operation";
  let description = [{
    This operation applies to its target a π/4 rotation about the Z axis.

    Matrix representation:
    ```
    T = | 1      0     |
        | 0  exp(iπ/4) |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ T ├─
     └───┘
    ```
  }];
}

def U2Op : QuakeOperator<"u2", [NumParameters<2>, NumTargets<1>, Rotation]> {
  let summary = "generic rotation about the X+Z axis";
  let description = [{
    The two parameters are Euler angles: φ and λ.

    Matrix representation:
    ```
    U2(φ,λ) = 1/sqrt(2) * | 1        -exp(iλ)       |
                          | exp(iφ)   exp(i(λ + φ)) |
    ```

    Circuit symbol:
    ```
     ┌─────────┐
    ─┤ U2(φ,λ) ├─
     └─────────┘
    ```
  }];
}

def U3Op : QuakeOperator<"u3", [NumParameters<3>, NumTargets<1>, Rotation]> {
  let summary = "the universal three-parameters operator";
  let description = [{
    The three parameters are Euler angles: θ, φ, and λ.

    NOTE: U3 is a generalization of U2 that covers all single-qubit rotations.

    Matrix representation:
    ```
    U3(θ,φ,λ) = | cos(θ/2)            -exp(iλ) * sin(θ/2)       |
                | exp(iφ) * sin(θ/2)   exp(i(λ + φ)) * cos(θ/2) |
    ```

    Circuit symbol:
    ```
     ┌───────────┐
    ─┤ U3(θ,φ,λ) ├─
     └───────────┘
    ```
  }];
}

def XOp : OneTargetOp<"x", [Hermitian]> {
  let summary = "Pauli-X operation (aka, NOT)";
  let description = [{
    Matrix representation:
    ```
    X = | 0  1 |
        | 1  0 |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ X ├─
     └───┘
    ```
  }];
}

def YOp : OneTargetOp<"y", [Hermitian]> {
  let summary = "Pauli-Y operation";
  let description = [{
    Matrix representation:
    ```
    Y = | 0  -i |
        | i   0 |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ Y ├─
     └───┘
    ```
  }];
}

def ZOp : OneTargetOp<"z", [Hermitian]> {
  let summary = "Pauli-Z operation.";
  let description = [{
    Matrix representation:
    ```
    Z = | 1   0 |
        | 0  -1 |
    ```

    Circuit symbol:
    ```
     ┌───┐
    ─┤ Z ├─
     └───┘
    ```
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_OPS
