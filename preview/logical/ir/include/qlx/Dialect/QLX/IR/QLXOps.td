//===- QLXOps.td - QLX operation definitions ---------------*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// This file defines all operations for the QLX (Quantum Logical eXecution)
// dialect. The ISA comprises 32+ instructions plus module-level declarations,
// organized into the following categories:
//
//   - Module-level: region_decl, interconnect_decl, target, entry
//   - State preparation: pz, pp, reset
//   - Single-qubit Clifford: h, s, sdg, x, y, z
//   - Non-Clifford gates: t, tdg, ccz
//   - Two-qubit Clifford: cx, cz
//   - Code-specific extensions: cop, cop2, cop3
//   - Measurement: mz, mx, mp, mpp, rpp
//   - Resource state / scheduling: rsv, ars
//   - Inter-region: lk, tp, csw
//   - Control/sync: bar, wt, we, te, br, bra, ret
//
//===----------------------------------------------------------------------===//

#ifndef QLX_OPS_TD
#define QLX_OPS_TD

include "qlx/Dialect/QLX/IR/QLXDialect.td"
include "qlx/Dialect/QLX/IR/QLXTypes.td"
include "qlx/Dialect/QLX/IR/QLXAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// QLX Op base class
//===----------------------------------------------------------------------===//

class QLX_Op<string mnemonic, list<Trait> traits = []>
    : Op<QLX_Dialect, mnemonic, traits>;

def QLX_EstimateResultOp : QLX_Op<"estimate_result", [Symbol]> {
  let summary = "versioned resource-estimation result with exact provenance";
  let description = [{
    Carries one immutable result from the QLX estimation ladder.  The payload
    is a schema-versioned dictionary for interchange inside MLIR; public Python
    and CUDA-Q adapters construct their own typed result values from it.  The
    operation is symbol-addressable so higher tiers can cite the exact result
    they refine instead of relying on a mutable module annotation.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$tier,
    FlatSymbolRefAttr:$root,
    StrAttr:$schema,
    DictionaryAttr:$data,
    ArrayAttr:$assumptions,
    ArrayAttr:$evidence,
    OptionalAttr<FlatSymbolRefAttr>:$lower_tier,
    OptionalAttr<FlatSymbolRefAttr>:$device,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def QLX_DeviceOp : QLX_Op<"device", [Symbol]> {
  let summary = "immutable manifest for a contiguous layered device stack";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$logical,
    OptionalAttr<FlatSymbolRefAttr>:$qec,
    OptionalAttr<FlatSymbolRefAttr>:$logical_to_qec,
    OptionalAttr<ArrayAttr>:$resource_bindings,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def QLX_LogicalToQECBindingOp : QLX_Op<"logical_to_qec", [Symbol]> {
  let summary = "static refinement from logical regions to QEC block pools";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$logical,
    FlatSymbolRefAttr:$qec,
    ArrayAttr:$entries
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def QLX_QECLoweringOp : QLX_Op<"qec_lowering", [Symbol]> {
  let summary = "versioned compiler manifest for P1 logical intent to P2 QEC IR";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$manifest_name,
    StrAttr:$manifest_sha256,
    StrAttr:$objective_family,
    OptionalAttr<AnyAttr>:$objective,
    ArrayAttr:$codes,
    ArrayAttr:$requirements,
    StrAttr:$compiler_plugin,
    StrAttr:$compiler_symbol,
    StrAttr:$compiler_version,
    ArrayAttr:$dependencies,
    StrAttr:$input_stage,
    StrAttr:$output_stage,
    ArrayAttr:$provides_facets,
    OptionalAttr<DictionaryAttr>:$policy_schema,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def QLX_LoweringRecipeOp : QLX_Op<"lowering_recipe", [Symbol]> {
  let summary = "portable manifest for one target launch capability";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$capability,
    ArrayAttr:$accepted_stages,
    ArrayAttr:$required_facets,
    OptionalAttr<StrAttr>:$produced_stage,
    ArrayAttr:$provides_facets,
    ArrayAttr:$stages,
    StrAttr:$finalizer,
    OptionalAttr<StrAttr>:$result_schema,
    StrAttr:$effect,
    OptionalAttr<DictionaryAttr>:$plugin,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def QLX_TargetManifestOp : QLX_Op<"target_manifest", [Symbol]> {
  let summary = "derived portable manifest for one immutable target";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ArrayAttr:$capabilities,
    ArrayAttr:$recipes,
    StrAttr:$availability,
    OptionalAttr<DictionaryAttr>:$plugin,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def QLX_ExperimentOp : QLX_Op<"experiment", [Symbol]> {
  let summary = "immutable manifest for one reproducible compilation point";
  let description = [{
    Records the selected root, verified stage/facets, ordered semantic pass recipe,
    linked symbol closure, and exact model/policy/parameter bindings.  It does
    not contain cache locations, diagnostic sinks, retention settings, or a
    process-local artifact identity.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$root,
    StrAttr:$stage,
    ArrayAttr:$facets,
    ArrayAttr:$pass_recipe,
    ArrayAttr:$closure,
    DictionaryAttr:$bindings
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Module-level declarations
//===----------------------------------------------------------------------===//

// The first QLX operations intentionally coexist with the alpha QLX
// operations below.  QLX P0 is machine-free and therefore never uses a
// `!qlx.region` operand. Familiar built-ins remain inline typed attributes;
// custom/composite objectives normalize to symbol-backed apply/instrument ops.

def QLX_ReturnOp : QLX_Op<"return", [Terminator]> {
  let summary = "Return from a QLX unplaced logical or objective body";
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "($results^ `:` type($results))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

def QLX_ProgramOp : QLX_Op<"program", [
    IsolatedFromAbove,
    Symbol,
    SymbolTable,
    SingleBlockImplicitTerminator<"ReturnOp">
  ]> {
  let summary = "QLX P0 unplaced logical program";
  let description = [{
    Owns one machine-free, code-free, unplaced logical program.  The
    `function_type` attribute specifies the entry-block argument types and the
    `qlx.return` operand types. `qlx.stage` is `p0`; `qlx.profile = "p0"` is
    accepted as a temporary compatibility mirror. `estimate_only` marks a
    typed demand surrogate that may be analyzed and placed but not lowered to
    execution. An executable program cannot call a marked program, directly or
    through a nested helper.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<UnitAttr>:$estimate_only,
    OptionalAttr<DictionaryAttr>:$specialization
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `:` $function_type attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def QLX_ObjectiveBodyOp : QLX_Op<"objective_body", [
    IsolatedFromAbove,
    Symbol,
    SymbolTable,
    SingleBlockImplicitTerminator<"ReturnOp">
  ]> {
  let summary = "closed ideal body backing a logical objective";
  let description = [{
    Owns the non-executing ideal fragment referenced by one qlx.action or
    qlx.instrument_decl. It uses the same value semantics as a P0 program but
    is not an application entry point and cannot be launched directly.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$objective_kind,
    OptionalAttr<DictionaryAttr>:$specialization
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `:` $function_type attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def QLX_ActionOp : QLX_Op<"action", [Symbol]> {
  let summary = "Declare a canonical logical action objective";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    StrAttr:$kind,
    OptionalAttr<AnyAttr>:$semantics,
    OptionalAttr<QLX_CliffordActionAttr>:$clifford_action,
    OptionalAttr<DictionaryAttr>:$effects
  );
  let assemblyFormat = "$sym_name `:` $function_type attr-dict";
}

def QLX_InstrumentDeclOp : QLX_Op<"instrument_decl", [Symbol]> {
  let summary = "Declare a canonical logical instrument objective";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    StrAttr:$kind,
    OptionalAttr<AnyAttr>:$semantics,
    OptionalAttr<DictionaryAttr>:$ownership,
    OptionalAttr<DictionaryAttr>:$effects
  );
  let assemblyFormat = "$sym_name `:` $function_type attr-dict";
}

def QLX_ApplyOp : QLX_Op<"apply", []> {
  let summary = "Apply a canonical logical action";
  let arguments = (ins
    AnyAttr:$action,
    Variadic<AnyType>:$inputs,
    OptionalAttr<DictionaryAttr>:$parameters
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$action `(` $inputs `)` attr-dict `:` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def QLX_InstrumentOp : QLX_Op<"instrument", []> {
  let summary = "Apply a custom or nontrivial built-in logical instrument";
  let arguments = (ins
    AnyAttr:$instrument,
    Variadic<AnyType>:$inputs,
    OptionalAttr<DictionaryAttr>:$parameters
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$instrument `(` $inputs `)` attr-dict `:` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def QLX_PrepareOp : QLX_Op<"prepare", []> {
  let summary = "Prepare one unplaced logical state";
  let arguments = (ins
    StrAttr:$state,
    OptionalAttr<I64Attr>:$allocation,
    OptionalAttr<I64Attr>:$value_index
  );
  let results = (outs QLX_LogicalQubitType:$result);
  let assemblyFormat = "$state attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QLX_MeasureOp : QLX_Op<"measure", []> {
  let summary = "Destructively measure one unplaced logical Pauli";
  let arguments = (ins QLX_PauliAttr:$basis, QLX_LogicalQubitType:$input);
  let results = (outs I1:$result);
  let assemblyFormat = "$basis $input attr-dict `:` type($input) `->` type($result)";
}

def QLX_CallOp : QLX_Op<"call", []> {
  let summary = "Call a materialized unplaced logical subprogram";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$inputs
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $inputs `)` attr-dict `:` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def QLX_IdleOp : QLX_Op<"idle", []> {
  let summary = "Apply an abstract logical-memory workload";
  let arguments = (ins
    Variadic<QLX_LogicalQubitType>:$inputs,
    Index:$rounds
  );
  let results = (outs Variadic<QLX_LogicalQubitType>:$results);
  let assemblyFormat = "$inputs `rounds` $rounds attr-dict `:` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def QLX_DiscardOp : QLX_Op<"discard", []> {
  let summary = "Explicitly terminate logical quantum ownership";
  let arguments = (ins
    Variadic<QLX_LogicalQubitType>:$inputs,
    OptionalAttr<StrAttr>:$reason
  );
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

def QLX_ResourceRequestOp : QLX_Op<"resource_request", []> {
  let summary = "request one unplaced logical resource asynchronously";
  let arguments = (ins StrAttr:$kind);
  let results = (outs QLX_LogicalEventType:$event);
  let assemblyFormat = "$kind attr-dict `:` type($event)";
  let hasVerifier = 1;
}

def QLX_EventTestOp : QLX_Op<"event_test", []> {
  let summary = "observe event readiness without consuming its payload";
  let arguments = (ins QLX_LogicalEventType:$event);
  let results = (outs I1:$ready);
  let assemblyFormat = "$event attr-dict `:` type($event) `->` type($ready)";
}

def QLX_EventPollOp : QLX_Op<"event_poll", []> {
  let summary = "observe the complete event state without consuming it";
  let arguments = (ins QLX_LogicalEventType:$event);
  let results = (outs I8:$status);
  let assemblyFormat = "$event attr-dict `:` type($event) `->` type($status)";
}

def QLX_EventIsOp : QLX_Op<"event_is", []> {
  let summary = "test one portable event-state value";
  let arguments = (ins I8:$status, StrAttr:$state);
  let results = (outs I1:$matches);
  let assemblyFormat = "$status $state attr-dict `:` type($status) `->` type($matches)";
  let hasVerifier = 1;
}

def QLX_EventSelectReadyOp : QLX_Op<"event_select_ready", []> {
  let summary = "select one ready event without consuming any payload";
  let arguments = (ins
    Variadic<QLX_LogicalEventType>:$events,
    DefaultValuedAttr<StrAttr, "\"priority\"">:$policy
  );
  let results = (outs Index:$selected);
  let assemblyFormat = "`(` $events `)` attr-dict `:` type($events) `->` type($selected)";
  let hasVerifier = 1;
}

def QLX_EventTryTakeOp : QLX_Op<"event_try_take", [
  SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "branch on one nonblocking event take with linear ownership";
  let arguments = (ins
    QLX_LogicalEventType:$event,
    Variadic<AnyType>:$carries
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$ready,
                         SizedRegion<1>:$pending,
                         SizedRegion<1>:$failed);
  let hasVerifier = 1;
}

def QLX_EventCancelOp : QLX_Op<"event_cancel", []> {
  let summary = "cancel one outstanding event and consume its ownership";
  let arguments = (ins QLX_LogicalEventType:$event, OptionalAttr<StrAttr>:$reason);
  let results = (outs I8:$status);
  let assemblyFormat = "$event attr-dict `:` type($event) `->` type($status)";
}

def QLX_EventAwaitOp : QLX_Op<"event_await", []> {
  let summary = "await and consume one typed event payload";
  let arguments = (ins QLX_LogicalEventType:$event);
  let results = (outs AnyType:$payload);
  let assemblyFormat = "$event attr-dict `:` type($event) `->` type($payload)";
  let hasVerifier = 1;
}

def QLX_FenceOp : QLX_Op<"fence", []> {
  let summary = "order only the declared portable semantic effects";
  let arguments = (ins ArrayAttr:$effects);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def QLX_SelectionOp : QLX_Op<"selection", []> {
  let summary = "declare portable statistical selection over a logical outcome";
  let arguments = (ins
    I1:$predicate,
    StrAttr:$mode,
    DefaultValuedAttr<BoolAttr, "true">:$accept_when
  );
  let assemblyFormat = "$predicate attr-dict `:` type($predicate)";
  let hasVerifier = 1;
}

def QLX_ConsumeResourceOp : QLX_Op<"consume_resource", []> {
  let summary = "consume a logical resource while applying its ideal action";
  let arguments = (ins
    QLX_LogicalResourceType:$resource,
    AnyAttr:$action,
    Variadic<QLX_LogicalQubitType>:$inputs
  );
  let results = (outs Variadic<QLX_LogicalQubitType>:$results);
  let assemblyFormat = "$resource `with` $action `(` $inputs `)` attr-dict `:` type($resource) `,` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def QLX_FrameInitOp : QLX_Op<"frame_init", []> {
  let summary = "create an empty unplaced logical frame";
  let arguments = (ins StrAttr:$domain);
  let results = (outs QLX_LogicalFrameType:$frame);
  let assemblyFormat = "$domain attr-dict `:` type($frame)";
  let hasVerifier = 1;
}

def QLX_FrameUpdateOp : QLX_Op<"frame_update", []> {
  let summary = "update symbolic frame state from a logical outcome";
  let arguments = (ins
    QLX_LogicalFrameType:$frame,
    I1:$source,
    StrAttr:$update
  );
  let results = (outs QLX_LogicalFrameType:$result);
  let assemblyFormat = "$frame `from` $source attr-dict `:` type($frame) `->` type($result)";
  let hasVerifier = 1;
}

def QLX_FrameTransformOp : QLX_Op<"frame_transform", []> {
  let summary = "apply a known logical transform to symbolic frame state";
  let arguments = (ins QLX_LogicalFrameType:$frame, StrAttr:$transform);
  let results = (outs QLX_LogicalFrameType:$result);
  let assemblyFormat = "$frame attr-dict `:` type($frame) `->` type($result)";
  let hasVerifier = 1;
}

def QLX_RegionDeclOp : QLX_Op<"region_decl", [Symbol]> {
  let summary = "Declare a logical region";
  let description = [{
    Declares a logical region with a QEC code type, role, and capacity.
    Regions are the fundamental organizational unit: each represents a
    contiguous block of hardware resources running a single QEC code.

    Example:
    ```mlir
    qlx.region_decl @Cs0 {
      code = "surface_17",
      role = #qlx.role<compute>,
      capacity = 4 : i64
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$code,
    QLX_RoleAttr:$role,
    OptionalAttr<I64Attr>:$capacity,
    OptionalAttr<StrAttr>:$distill,
    OptionalAttr<QLX_CodeDescAttr>:$code_desc,
    OptionalAttr<QLX_FloorplanAttr>:$floorplan,
    OptionalAttr<QLX_RouteStrategyAttr>:$route_strategy
  );

  let results = (outs);
  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

def QLX_InterconnectDeclOp : QLX_Op<"interconnect_decl", [Symbol]> {
  let summary = "Declare an interconnect between two regions";
  let description = [{
    Declares an interconnect between two regions, specifying the protocol
    used for inter-region communication and an optional cost in nanoseconds.

    Example:
    ```mlir
    qlx.interconnect_decl @ic0 {
      from = @Cs0,
      to = @Fs0,
      proto = #qlx.protocol<teleport>,
      cost_ns = 1.0e+03 : f64
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$from,
    FlatSymbolRefAttr:$to,
    QLX_ProtocolAttr:$proto,
    OptionalAttr<F64Attr>:$cost_ns
  );

  let results = (outs);
  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

def QLX_TargetOp : QLX_Op<"target", []> {
  let summary = "Declare the target profile triple";
  let description = [{
    Specifies the QLX target profile for the module. The profile string
    constrains which operations and region configurations are legal.

    Example:
    ```mlir
    qlx.target "qlx-cmf-csw.tp"
    ```
  }];

  let arguments = (ins StrAttr:$profile);
  let results = (outs);
  let assemblyFormat = [{
    $profile attr-dict
  }];
}

def QLX_RetOp : QLX_Op<"ret", [Terminator]> {
  let summary = "Return from an entry function";
  let description = [{
    Terminates the body of a `qlx.entry` function. Optionally takes
    logical measurement bits as operands, making the program's output
    explicit.

    Examples:
    ```mlir
    qlx.ret
    qlx.ret %b0, %b1 : !qlx.lbit, !qlx.lbit
    ```
  }];

  let arguments = (ins Variadic<QLX_LBitType>:$results);
  let assemblyFormat = [{
    ($results^ `:` type($results))? attr-dict
  }];
  let builders = [
    // Default builder with no operands (for SingleBlockImplicitTerminator).
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

def QLX_EntryOp : QLX_Op<"entry", [
    IsolatedFromAbove,
    Symbol,
    SingleBlockImplicitTerminator<"RetOp">
  ]> {
  let summary = "Entry point function for a QLX program";
  let description = [{
    Defines the entry point of a QLX program. The entry function takes
    region references as arguments and contains a single block of QLX
    operations. It is analogous to `func.func` but specialized for
    the QLX execution model.

    The entry block arguments correspond to the region parameters
    declared at module level via `qlx.region_decl`.

    Example:
    ```mlir
    qlx.entry @hello_qlx(%Cs0: !qlx.region, %Fs0: !qlx.region) {
      %q0 = qlx.pz %Cs0 : !qlx.lqbit
      %q0_1 = qlx.h %q0, %Cs0 : !qlx.lqbit
      %s0 = qlx.mz %q0_1, %Cs0 : !qlx.lbit
      qlx.ret
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<ArrayAttr>:$region_params
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name,
                   "::mlir::TypeRange":$argTypes,
                   "::llvm::ArrayRef<::mlir::NamedAttribute>":$attrs), [{
      $_state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(),
                           $_builder.getStringAttr(name));
      $_state.addAttributes(attrs);
      ::mlir::Region *bodyRegion = $_state.addRegion();
      ::mlir::Block *body = new ::mlir::Block();
      bodyRegion->push_back(body);
      for (::mlir::Type ty : argTypes)
        body->addArgument(ty, $_builder.getUnknownLoc());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// State preparation ops
//===----------------------------------------------------------------------===//

def QLX_PzOp : QLX_Op<"pz", []> {
  let summary = "Allocate and prepare a logical qubit in |0>";
  let description = [{
    Allocates a logical qubit in the specified region and prepares it in
    the computational basis state |0> (Z-basis eigenstate). Consumes one
    slot of the region's capacity.

    Example:
    ```mlir
    %q = qlx.pz %R : !qlx.lqbit
    ```
  }];

  let arguments = (ins QLX_RegionType:$region);
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $region attr-dict `:` type($result)
  }];
}

def QLX_PpOp : QLX_Op<"pp", []> {
  let summary = "Allocate and prepare a logical qubit in |+>";
  let description = [{
    Allocates a logical qubit in the specified region and prepares it in
    the |+> state (X-basis eigenstate). Consumes one slot of the region's
    capacity.

    Example:
    ```mlir
    %q = qlx.pp %R : !qlx.lqbit
    ```
  }];

  let arguments = (ins QLX_RegionType:$region);
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $region attr-dict `:` type($result)
  }];
}

def QLX_PrepareInjectOp : QLX_Op<"pinj", []> {
  let summary = "Prepare logical qubit via non-Clifford state injection";
  let description = [{
    Prepares a logical qubit in state U(params)|0> using the code type's
    injection protocol. Allocates a new qubit slot in the region.

    For full-lowering architectures (LoweringStrategy::physical), this
    decomposes to pz + gridsynth rotation synthesis into rsv/ars T-gate
    sequences.

    For logical-emission architectures (LoweringStrategy::logical), the
    operation is preserved through lowering and emitted as the injection
    gate (e.g. U3) in the output format.

    Example:
    ```mlir
    %q = qlx.pinj %R, "u3"([0.5, 1.2, 0.0]) : !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_RegionType:$region,
    StrAttr:$gate,
    F64ArrayAttr:$params
  );
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $region `,` $gate `(` $params `)` attr-dict `:` type($result)
  }];
}

def QLX_ResetOp : QLX_Op<"reset", []> {
  let summary = "Reset a logical qubit to |0>";
  let description = [{
    Resets an existing logical qubit to the |0> state within its region.
    Unlike `qlx.pz`, this does not allocate a new qubit slot -- it
    reinitializes the qubit in place.

    Example:
    ```mlir
    %q_new = qlx.reset %q, %R : !qlx.lqbit
    ```
  }];

  let arguments = (ins QLX_LQBitType:$qubit, QLX_RegionType:$region);
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $region attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Single-qubit gate template
//
// All single-qubit gates take a logical qubit and its owning region as
// operands and produce a new logical qubit value (value semantics).
// Used for both Clifford and non-Clifford single-qubit gates.
//===----------------------------------------------------------------------===//

class QLX_SingleQubitGateOp<string mnemonic, string gateName>
    : QLX_Op<mnemonic, []> {
  let summary = "Apply logical " # gateName # " gate";
  let description = [{
    Applies the logical }] # gateName # [{ gate to a qubit within its region.
    The qubit must belong to the specified region. The input qubit is consumed
    and a new qubit value is produced (value semantics).
  }];

  let arguments = (ins QLX_LQBitType:$qubit, QLX_RegionType:$region);
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $region attr-dict `:` type($result)
  }];
}

// Backward-compatible alias.
class QLX_SingleQubitCliffordOp<string mnemonic, string gateName>
    : QLX_SingleQubitGateOp<mnemonic, gateName>;

def QLX_HOp   : QLX_SingleQubitGateOp<"h",   "Hadamard (H)">;
def QLX_SOp   : QLX_SingleQubitGateOp<"s",   "phase (S)">;
def QLX_SdgOp : QLX_SingleQubitGateOp<"sdg", "adjoint phase (S-dagger)">;
def QLX_XOp   : QLX_SingleQubitGateOp<"x",   "Pauli-X">;
def QLX_YOp   : QLX_SingleQubitGateOp<"y",   "Pauli-Y">;
def QLX_ZOp   : QLX_SingleQubitGateOp<"z",   "Pauli-Z">;
def QLX_SXOp  : QLX_SingleQubitGateOp<"sx",  "sqrt(X) (X^1/2)">;
def QLX_SYOp  : QLX_SingleQubitGateOp<"sy",  "sqrt(Y) (Y^1/2)">;

//===----------------------------------------------------------------------===//
// Non-Clifford gates
//===----------------------------------------------------------------------===//

def QLX_TOp   : QLX_SingleQubitGateOp<"t",   "T (pi/8)">;
def QLX_TdgOp : QLX_SingleQubitGateOp<"tdg", "T-dagger">;

def QLX_CCZOp : QLX_Op<"ccz", []> {
  let summary = "Logical controlled-controlled-Z gate";
  let description = [{
    Applies a logical controlled-controlled-Z (Toffoli-phase) gate to three
    qubits within the same region. All three qubits must belong to the
    specified region.

    Example:
    ```mlir
    %q0_out, %q1_out, %q2_out = qlx.ccz %q0, %q1, %q2, %R
        : !qlx.lqbit, !qlx.lqbit, !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit1,
    QLX_LQBitType:$qubit2,
    QLX_LQBitType:$qubit3,
    QLX_RegionType:$region
  );
  let results = (outs
    QLX_LQBitType:$qubit1_out,
    QLX_LQBitType:$qubit2_out,
    QLX_LQBitType:$qubit3_out
  );
  let assemblyFormat = [{
    $qubit1 `,` $qubit2 `,` $qubit3 `,` $region attr-dict `:` type(results)
  }];
}

//===----------------------------------------------------------------------===//
// Code-specific extension ops (cop)
//===----------------------------------------------------------------------===//

def QLX_CopOp : QLX_Op<"cop", []> {
  let summary = "1-qubit code-specific extension operation";
  let description = [{
    Applies a code-specific (Tier B) operation to a single qubit. The
    `op_name` attribute identifies the operation within the code type's
    extension set (validated against `gate_class()` at assembly time).

    Example:
    ```mlir
    %q_out = qlx.cop %q, %R, "shift" : !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit,
    QLX_RegionType:$region,
    StrAttr:$op_name
  );
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $region `,` $op_name attr-dict `:` type($result)
  }];
}

def QLX_Cop2Op : QLX_Op<"cop2", []> {
  let summary = "2-qubit code-specific extension operation";
  let description = [{
    Applies a code-specific (Tier B) operation to two qubits. Both qubits
    must belong to the specified region.

    Example:
    ```mlir
    %q0_out, %q1_out = qlx.cop2 %q0, %q1, %R, "zz_measure"
        : !qlx.lqbit, !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit1,
    QLX_LQBitType:$qubit2,
    QLX_RegionType:$region,
    StrAttr:$op_name
  );
  let results = (outs QLX_LQBitType:$qubit1_out, QLX_LQBitType:$qubit2_out);
  let assemblyFormat = [{
    $qubit1 `,` $qubit2 `,` $region `,` $op_name attr-dict `:` type(results)
  }];
}

def QLX_Cop3Op : QLX_Op<"cop3", []> {
  let summary = "3-qubit code-specific extension operation";
  let description = [{
    Applies a code-specific (Tier B) operation to three qubits. All qubits
    must belong to the specified region.

    Example:
    ```mlir
    %q0_out, %q1_out, %q2_out = qlx.cop3 %q0, %q1, %q2, %R, "toffoli_native"
        : !qlx.lqbit, !qlx.lqbit, !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit1,
    QLX_LQBitType:$qubit2,
    QLX_LQBitType:$qubit3,
    QLX_RegionType:$region,
    StrAttr:$op_name
  );
  let results = (outs
    QLX_LQBitType:$qubit1_out,
    QLX_LQBitType:$qubit2_out,
    QLX_LQBitType:$qubit3_out
  );
  let assemblyFormat = [{
    $qubit1 `,` $qubit2 `,` $qubit3 `,` $region `,` $op_name attr-dict `:` type(results)
  }];
}

def QLX_CopNOp : QLX_Op<"copn", []> {
  let summary = "Variadic code-specific extension operation";
  let description = [{
    Applies a code-specific operation to a variadic set of logical qubits in
    one region. This is intended for packed high-rate blocks where a code
    supplies an operation on a selected tuple of logical slots (for example, a
    whole-block automorphism with a pinned action on several logical qubits).

    Example:
    ```mlir
    %q0_out, %q1_out, %q2_out, %q5_out = qlx.copn
        %q0, %q1, %q2, %q5, %R, "path4_free_cnots"
        : !qlx.lqbit, !qlx.lqbit, !qlx.lqbit, !qlx.lqbit
    ```
  }];

  let arguments = (ins
    Variadic<QLX_LQBitType>:$qubits,
    QLX_RegionType:$region,
    StrAttr:$op_name
  );
  let results = (outs Variadic<QLX_LQBitType>:$qubit_results);
}

//===----------------------------------------------------------------------===//
// Two-qubit Clifford ops
//===----------------------------------------------------------------------===//

def QLX_CXOp : QLX_Op<"cx", []> {
  let summary = "Logical CNOT (controlled-X) gate";
  let description = [{
    Applies a logical CNOT gate with the first operand as control and the
    second as target. Both qubits must reside in the same region.

    Example:
    ```mlir
    %ctrl_out, %targ_out = qlx.cx %ctrl, %targ, %R : !qlx.lqbit, !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$ctrl,
    QLX_LQBitType:$targ,
    QLX_RegionType:$region
  );
  let results = (outs QLX_LQBitType:$ctrl_out, QLX_LQBitType:$targ_out);
  let assemblyFormat = [{
    $ctrl `,` $targ `,` $region attr-dict `:` type(results)
  }];
}

def QLX_CZOp : QLX_Op<"cz", []> {
  let summary = "Logical controlled-Z gate";
  let description = [{
    Applies a logical controlled-Z gate between two qubits. Both qubits
    must reside in the same region. CZ is symmetric: the two operands
    are interchangeable.

    Example:
    ```mlir
    %q1_out, %q2_out = qlx.cz %q1, %q2, %R : !qlx.lqbit, !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$q1,
    QLX_LQBitType:$q2,
    QLX_RegionType:$region
  );
  let results = (outs QLX_LQBitType:$q1_out, QLX_LQBitType:$q2_out);
  let assemblyFormat = [{
    $q1 `,` $q2 `,` $region attr-dict `:` type(results)
  }];
}

//===----------------------------------------------------------------------===//
// Measurement ops
//===----------------------------------------------------------------------===//

def QLX_MzOp : QLX_Op<"mz", []> {
  let summary = "Destructive Z-basis measurement";
  let description = [{
    Performs a destructive measurement of a logical qubit in the Z basis.
    The qubit is consumed (no further uses are valid) and a logical
    classical bit is produced.

    Example:
    ```mlir
    %bit = qlx.mz %q, %R : !qlx.lbit
    ```
  }];

  let arguments = (ins QLX_LQBitType:$qubit, QLX_RegionType:$region);
  let results = (outs QLX_LBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $region attr-dict `:` type($result)
  }];
}

def QLX_MxOp : QLX_Op<"mx", []> {
  let summary = "Destructive X-basis measurement";
  let description = [{
    Performs a destructive measurement of a logical qubit in the X basis.
    The qubit is consumed and a logical classical bit is produced.

    Example:
    ```mlir
    %bit = qlx.mx %q, %R : !qlx.lbit
    ```
  }];

  let arguments = (ins QLX_LQBitType:$qubit, QLX_RegionType:$region);
  let results = (outs QLX_LBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $region attr-dict `:` type($result)
  }];
}

def QLX_MpOp : QLX_Op<"mp", []> {
  let summary = "Destructive arbitrary-Pauli-basis measurement";
  let description = [{
    Performs a destructive measurement of a logical qubit in an arbitrary
    Pauli basis (X, Y, or Z) specified by the `pauli` attribute.
    The qubit is consumed and a logical classical bit is produced.

    Example:
    ```mlir
    %bit = qlx.mp %q, %R pauli = #qlx.pauli<Y> : !qlx.lbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit,
    QLX_RegionType:$region,
    QLX_PauliAttr:$pauli
  );
  let results = (outs QLX_LBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $region `pauli` `=` $pauli attr-dict `:` type($result)
  }];
}

def QLX_MppOp : QLX_Op<"mpp", [AttrSizedOperandSegments,
                                AttrSizedResultSegments]> {
  let summary = "Joint multi-qubit Pauli-product measurement (non-destructive)";
  let description = [{
    Performs a non-destructive joint measurement of multiple logical qubits
    in a specified Pauli-product basis. Used for cross-region stabilizer
    measurements and entanglement verification. The qubits survive the
    measurement -- new qubit values are produced (value semantics).

    `pauli` is a string of `X`/`Y`/`Z` characters, one per qubit operand.
    `pauli[i]` is the Pauli applied to `qubits[i]` in the product. The
    verifier rejects strings with length != number of qubits or strings
    that contain any non-`X`/`Y`/`Z` character.

    The number of qubits and regions must match: each qubit is associated
    with its corresponding region. The number of output qubits equals the
    number of input qubits.

    Example:
    ```mlir
    // Joint X⊗Z measurement of %q0 and %q1.
    %q0_out, %q1_out, %bit = qlx.mpp %q0, %q1, %R0, %R1
        pauli = "XZ" : !qlx.lqbit, !qlx.lqbit, !qlx.lbit
    ```
  }];

  let arguments = (ins
    Variadic<QLX_LQBitType>:$qubits,
    Variadic<QLX_RegionType>:$qubit_regions,
    StrAttr:$pauli
  );
  let results = (outs
    Variadic<QLX_LQBitType>:$qubit_results,
    Variadic<QLX_LBitType>:$bit_results
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def QLX_RppOp : QLX_Op<"rpp", [AttrSizedOperandSegments]> {
  let summary = "Logical Pauli-product rotation";
  let description = [{
    Applies a logical rotation exp(-i angle P / 2) for a Pauli product P.
    The qubits survive and are returned with value semantics. `pauli_product`
    contains one X/Y/Z character per qubit. `synthesis` is one of `auto`,
    `native`, or `decompose`.
  }];
  let arguments = (ins
    Variadic<QLX_LQBitType>:$qubits,
    Variadic<QLX_RegionType>:$qubit_regions,
    StrAttr:$pauli_product,
    F64Attr:$angle,
    StrAttr:$synthesis
  );
  let results = (outs Variadic<QLX_LQBitType>:$qubit_results);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $qubits `regions` $qubit_regions attr-dict `:` type($qubit_results)
  }];
}

//===----------------------------------------------------------------------===//
// Resource state / non-Clifford ops
//===----------------------------------------------------------------------===//

def QLX_RsvOp : QLX_Op<"rsv", []> {
  let summary = "Reserve one resource state from a factory region";
  let description = [{
    Reserves one resource state (magic state) from a factory or
    self-distilling region. The `type` attribute specifies the kind of
    resource state (T, CCZ, or CS).

    Returns a resource state promise handle that is consumed by exactly
    one `qlx.ars`. For N magic states, write N `qlx.rsv` ops.

    Example:
    ```mlir
    %rs = qlx.rsv %R type = <T> : !qlx.rs
    ```
  }];

  let arguments = (ins
    QLX_RegionType:$region,
    QLX_ResourceTypeAttr:$type
  );
  let results = (outs QLX_RSType:$result);
  let assemblyFormat = [{
    $region `type` `=` $type attr-dict `:` type($result)
  }];
}

def QLX_ArsOp : QLX_Op<"ars", []> {
  let summary = "Apply resource state to implement non-Clifford gate";
  let description = [{
    Consumes a resource state promise and applies it to a logical qubit,
    implementing a non-Clifford gate (e.g., T gate via T-state injection).
    The resource state handle is consumed by this operation. The optional
    `gate` attribute preserves the logical operation intent when the same
    resource kind can implement multiple adjoint operations, e.g. `t` and
    `tdg` both consume T states but resolve through distinct injection
    protocols.

    Example:
    ```mlir
    %q_out = qlx.ars %q, %rs, %R : !qlx.lqbit
    %q_tdg = qlx.ars %q, %rs, %R {gate = "tdg"} : !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit,
    QLX_RSType:$rs,
    QLX_RegionType:$region,
    OptionalAttr<StrAttr>:$gate
  );
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $rs `,` $region attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Inter-region ops
//===----------------------------------------------------------------------===//

def QLX_LkOp : QLX_Op<"lk", []> {
  let summary = "Create a Bell pair link between two regions";
  let description = [{
    Creates a Bell pair (entangled link) between two regions. Returns a
    resource state handle representing the Bell pair, which can be consumed
    by `qlx.tp` (teleport) for inter-region qubit transfer.

    Example:
    ```mlir
    %bell = qlx.lk %Ra, %Rb : !qlx.rs
    ```
  }];

  let arguments = (ins QLX_RegionType:$region_a, QLX_RegionType:$region_b);
  let results = (outs QLX_RSType:$result);
  let assemblyFormat = [{
    $region_a `,` $region_b attr-dict `:` type($result)
  }];
}

def QLX_TpOp : QLX_Op<"tp", []> {
  let summary = "Teleport a logical qubit using a Bell pair";
  let description = [{
    Teleports a logical qubit to another region using a pre-established
    Bell pair. The source qubit and the Bell pair resource state are both
    consumed. A new logical qubit is produced in the destination region.

    Example:
    ```mlir
    %q_dest = qlx.tp %q_src, %bell : !qlx.lqbit
    ```
  }];

  let arguments = (ins QLX_LQBitType:$qubit, QLX_RSType:$rs);
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $rs attr-dict `:` type($result)
  }];
}

def QLX_CswOp : QLX_Op<"csw", []> {
  let summary = "Code-switch a logical qubit between regions";
  let description = [{
    Performs a code switch: re-encodes a logical qubit from the source
    region's QEC code into the destination region's QEC code and transfers
    it. The source qubit is consumed and a new qubit is produced in the
    destination region.

    Requires a declared interconnect with codeswitch protocol between
    the source and destination regions.

    Example:
    ```mlir
    %q_new = qlx.csw %q, %Rs, %Rd : !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit,
    QLX_RegionType:$src_region,
    QLX_RegionType:$dst_region
  );
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $src_region `,` $dst_region attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Store/Load ops (assembly-level transfer abstraction)
//===----------------------------------------------------------------------===//

def QLX_StoreOp : QLX_Op<"store", []> {
  let summary = "Store a logical qubit from compute to memory/storage";
  let description = [{
    Assembly-level abstraction for transferring a logical qubit from a
    source region to a destination region. The assembler/lowering may
    expand this to `qlx.csw`, teleport sequences, or other protocols
    depending on the architecture descriptor.

    Example:
    ```mlir
    %q_new = qlx.store %q, %src_region, %dst_region : !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit,
    QLX_RegionType:$src_region,
    QLX_RegionType:$dst_region
  );
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $src_region `,` $dst_region attr-dict `:` type($result)
  }];
}

def QLX_LoadOp : QLX_Op<"load", []> {
  let summary = "Load a logical qubit from memory/storage to compute";
  let description = [{
    Assembly-level abstraction for transferring a logical qubit from a
    source region to a destination region. The assembler/lowering may
    expand this to `qlx.csw`, teleport sequences, or other protocols
    depending on the architecture descriptor.

    Example:
    ```mlir
    %q_new = qlx.load %q, %src_region, %dst_region : !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit,
    QLX_RegionType:$src_region,
    QLX_RegionType:$dst_region
  );
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $src_region `,` $dst_region attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Control/sync ops
//===----------------------------------------------------------------------===//

def QLX_BarOp : QLX_Op<"bar", []> {
  let summary = "Barrier across one or more regions";
  let description = [{
    Inserts a synchronization barrier across the specified regions.
    All operations on the listed regions before the barrier must complete
    before any operations after the barrier begin.

    Example:
    ```mlir
    qlx.bar %R0, %R1, %R2
    ```
  }];

  let arguments = (ins Variadic<QLX_RegionType>:$sync_regions);
  let results = (outs);
  let assemblyFormat = [{
    $sync_regions attr-dict
  }];
}

def QLX_WtOp : QLX_Op<"wt", []> {
  let summary = "Wait for a specified number of QEC rounds";
  let description = [{
    Pauses execution on the specified regions for the given number of
    QEC stabilizer rounds. This is used for timing alignment and to
    ensure sufficient error correction cycles between operations.
    Variadic over regions so a single `qlx.wt` can synchronize all
    entry regions at once (the .qlx surface emits one op covering
    every entry region).

    Example:
    ```mlir
    qlx.wt %R0, %R1 rounds = 3
    ```
  }];

  let arguments = (ins
    Variadic<QLX_RegionType>:$wait_regions,
    I64Attr:$rounds
  );
  let results = (outs);
  let assemblyFormat = [{
    $wait_regions `rounds` `=` $rounds attr-dict
  }];
}

def QLX_WeOp : QLX_Op<"we", []> {
  let summary = "Wait on an event (resource state completion)";
  let description = [{
    Blocks until the specified resource state event is ready. This is
    used to synchronize on the completion of asynchronous resource state
    production (e.g., waiting for a distillation round to complete).

    Example:
    ```mlir
    qlx.we %evt
    ```
  }];

  let arguments = (ins QLX_RSType:$event);
  let results = (outs);
  let assemblyFormat = [{
    $event attr-dict
  }];
}

def QLX_TeOp : QLX_Op<"te", []> {
  let summary = "Non-blocking test of an event";
  let description = [{
    Tests whether a resource state event is ready without blocking.
    Returns a logical classical bit indicating the result: 1 if the
    event is complete, 0 otherwise.

    Example:
    ```mlir
    %ready = qlx.te %evt : !qlx.lbit
    ```
  }];

  let arguments = (ins QLX_RSType:$event);
  let results = (outs QLX_LBitType:$result);
  let assemblyFormat = [{
    $event attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Structured loop
//===----------------------------------------------------------------------===//

def QLX_RepeatOp : QLX_Op<"repeat", [
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Repeat a body N times with loop-carried logical values";
  let description = [{
    Executes its body region `count` times, threading logical values
    (`!qlx.lqbit` / `!qlx.lbit` / `!qlx.rs`) through iterations via
    iter-args. Each iteration receives the previous iteration's yielded
    values as block arguments; the op's results are the values yielded by
    the final iteration.

    This is the QLX-level structured loop. It is the natural, faithful
    encoding for the repeated blocks of large fault-tolerant algorithms
    (e.g. the modular-arithmetic loops of Gidney-Ekera factoring): the
    body is authored once with real gates and the trip count is carried
    explicitly, so downstream passes (lowering to `fabric.repeat`, the
    resource counter, the scheduler) multiply through the loop analytically
    instead of materialising every iteration.

    Example:
    ```mlir
    %q_out = qlx.repeat 64
        iter(%qi : !qlx.lqbit = %q_in) {
      %qn = qlx.h %qi, %Cs0 : !qlx.lqbit
      qlx.yield %qn : !qlx.lqbit
    }
    ```
  }];

  let arguments = (ins
    I64Attr:$count,
    Variadic<AnyType>:$inits
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def QLX_YieldOp : QLX_Op<"yield", [
    Terminator,
    ParentOneOf<["RepeatOp", "IfOp", "WhileOp", "EventTryTakeOp"]>
  ]> {
  let summary = "Yield values from a qlx.repeat or qlx.if body";
  let description = [{
    Terminates the body of a `qlx.repeat` or branch of a `qlx.if`, yielding
    values back to the parent op. For `qlx.repeat`, the operands are the
    loop-carried values for the next iteration. For `qlx.if`, the operands
    are the branch results.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

def QLX_WhileConditionOp : QLX_Op<"while_condition", [
    Terminator, HasParent<"WhileOp">
  ]> {
  let summary = "decide whether a portable dynamic loop continues";
  let description = [{
    Terminates the `before` region of `qlx.while`.  `condition` controls
    entry into the body and `forwarded` supplies the body's complete carried
    state.  The forwarded values have exactly the while result types, so every
    linear owner has one successor on both the continue and exit edges.
  }];
  let arguments = (ins I1:$condition, Variadic<AnyType>:$forwarded);
}

def QLX_WhileOp : QLX_Op<"while", []> {
  let summary = "dynamic portable loop with typed linear carries";
  let description = [{
    Models the same two-region contract as `scf.while`.  The `before` region
    receives the current carries and ends in `qlx.while_condition`; when true,
    its forwarded values enter the `after` body.  The body ends in `qlx.yield`
    and feeds the next condition evaluation.  `max_iterations` is an optional
    semantic bound, not an instruction to unroll the loop.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inits,
    OptionalAttr<I64Attr>:$max_iterations
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$before_region,
    SizedRegion<1>:$after_region
  );
  let hasVerifier = 1;
}

def QLX_IfOp : QLX_Op<"if", [
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Conditional execution on a logical classical bit";
  let description = [{
    Executes one of two single-block regions depending on a logical classical
    bit. The op returns the values yielded by the selected branch. It is the
    logical-layer form for measurement-dependent feed-forward, such as Pauli
    byproduct corrections.

    Branch regions may reference values defined outside the op. When a branch
    updates a logical qubit, the updated qubit should be yielded and threaded
    through the op result.
  }];

  // `AnyType` temporarily accepts both the alpha !qlx.lbit and QLX i1.
  // The custom verifier rejects every other type.  Migration eventually
  // removes the legacy alternative and leaves builtin i1 only.
  let arguments = (ins AnyType:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$then_region,
    SizedRegion<1>:$else_region
  );
  let hasVerifier = 1;
}

def QLX_XorOp : QLX_Op<"xor", [AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Classical parity (XOR) of logical bits";
  // QLX uses builtin i1. AnyType retains the alpha !qlx.lbit parser
  // surface until cutover; the verifier accepts only those two exact forms.
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QLX_BrOp : QLX_Op<"br", [
    Terminator,
    AttrSizedOperandSegments
  ]> {
  let summary = "Conditional branch on a logical classical bit";
  let description = [{
    Conditional branch terminator. Takes a logical classical bit as
    condition and branches to one of two successor blocks: the first
    successor if the condition is true (1), the second if false (0).

    Each successor may take block arguments forwarded from this op.

    Example:
    ```mlir
    qlx.br %cond, ^bb_true(%arg0 : !qlx.lqbit), ^bb_false(%arg1 : !qlx.lqbit)
    ```
  }];

  let arguments = (ins
    QLX_LBitType:$condition,
    Variadic<AnyType>:$trueOperands,
    Variadic<AnyType>:$falseOperands
  );
  let results = (outs);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueOperands^ `:` type($trueOperands) `)`)? `,`
    $falseDest (`(` $falseOperands^ `:` type($falseOperands) `)`)?
    attr-dict
  }];
}

def QLX_BraOp : QLX_Op<"bra", [Terminator]> {
  let summary = "Unconditional branch";
  let description = [{
    Unconditional branch terminator. Transfers control to the specified
    successor block, optionally forwarding operands as block arguments.

    Example:
    ```mlir
    qlx.bra ^bb_next(%q : !qlx.lqbit)
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let results = (outs);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}
def QLX_RouteOp : QLX_Op<"route", []> {
  let summary = "Route a logical qubit between floorplan slots";
  let description = [{
    Routes a logical qubit from one floorplan slot to another. This op
    is inserted by the scheduler or routing pass — never written by the
    user. It carries the full corridor path and routing strategy.

    The op returns a new `!qlx.lqbit` value (value semantics), replacing
    the input qubit in subsequent uses.

    Example:
    ```mlir
    %q1 = qlx.route %q0, %from to %to {
      via = [1 : i32, 3 : i32],
      strategy = #qlx.route_strategy<patch_deform>,
      region = "Cs0"
    } : !qlx.lqbit
    ```
  }];

  let arguments = (ins
    QLX_LQBitType:$qubit,
    I32:$from_slot,
    I32:$to_slot,
    I32ArrayAttr:$via,
    QLX_RouteStrategyAttr:$strategy,
    OptionalAttr<StrAttr>:$region
  );
  let results = (outs QLX_LQBitType:$result);
  let assemblyFormat = [{
    $qubit `,` $from_slot `to` $to_slot attr-dict `:` type($result)
  }];
}

// Factory production is expressed by the SSA edge from `qlx.rsv` (reserve)
// to `qlx.ars` (apply); see the resource-state ops above. No explicit
// async-launch / await ops appear at this layer — the dataflow edge is
// the only required synchronization point. A downstream scheduler may
// introduce its own launch/await ops in a target-specific dialect.

// Note: QLX_RetOp is defined earlier (before QLX_EntryOp) because
// SingleBlockImplicitTerminator references it.

#endif // QLX_OPS_TD
