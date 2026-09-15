/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// This file defines all operations for the QLX (Quantum Logical eXecution)
// dialect: a machine-free P0 logical program stratum plus the target-manifest
// and analysis-result strata that sit alongside it, organized into the
// following categories:
//
//   - Analysis-result: estimate_result
//   - Target-manifest: device, logical_to_qec, qec_to_physical,
//     qec_lowering, lowering_recipe, target_manifest,
//     experiment
//   - P0 program: program, objective_body, action, instrument_decl, apply,
//     instrument, prepare, measure, call, idle, discard
//   - P0 resource events: resource_request (stays dialect-local; returns
//     the shared !event.handle<...>), consume_resource. The rest of the
//     observe/select/cancel/await/fence/selection family is embedded
//     directly from the shared `event` dialect (event.test, event.poll,
//     event.is, event.select_ready, event.try_take, event.cancel,
//     event.await, event.fence, event.selection, event.yield) rather than
//     redefined locally.
//   - P0 frames and control: frame_init, frame_update, frame_transform
//   - Structured control flow: xor (repeat/if/while/while_condition/yield
//     live in the `cflow` dialect)
//
//===----------------------------------------------------------------------===//

#ifndef QLX_OPS_TD
#define QLX_OPS_TD

include "qlx/Dialect/Event/IR/EventTypes.td"
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
    OptionalAttr<FlatSymbolRefAttr>:$physical,
    OptionalAttr<FlatSymbolRefAttr>:$logical_to_qec,
    OptionalAttr<FlatSymbolRefAttr>:$qec_to_physical,
    OptionalAttr<FlatSymbolRefAttr>:$operating_point,
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

def QLX_QECToPhysicalBindingOp : QLX_Op<"qec_to_physical", [Symbol]> {
  let summary = "static refinement from QEC block pools to physical resources";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$qec,
    FlatSymbolRefAttr:$physical,
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
    OptionalAttr<ArrayAttr>:$consumes_resources,
    StrAttr:$input_stage,
    StrAttr:$output_stage,
    ArrayAttr:$provides_facets,
    OptionalAttr<DictionaryAttr>:$policy_schema,
    OptionalAttr<DictionaryAttr>:$metadata,
    OptionalAttr<DictionaryAttr>:$rpp_selections
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
// P0 program
//===----------------------------------------------------------------------===//

// QLX P0 is machine-free and therefore no op below takes a `!qlx.region`
// operand. Familiar built-ins remain inline typed attributes; custom or
// composite objectives normalize to symbol-backed apply/instrument ops.

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
  let results = (outs Event_HandleType:$event);
  let assemblyFormat = "$kind attr-dict `:` type($event)";
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

def QLX_XorOp : QLX_Op<"xor", [AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Classical parity (XOR) of logical bits";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

#endif // QLX_OPS_TD
