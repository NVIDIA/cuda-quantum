/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_LVM_OPS_TD
#define QLX_LVM_OPS_TD

include "qlx/Dialect/LVM/IR/LVMDialect.td"
include "qlx/Dialect/LVM/IR/LVMTypes.td"
include "qlx/Dialect/LVM/IR/LVMAttrs.td"
include "qlx/Dialect/Event/IR/EventTypes.td"
include "qlx/Dialect/QLX/IR/QLXAttrs.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class LVM_Op<string mnemonic, list<Trait> traits = []>
    : Op<LVM_Dialect, mnemonic, traits>;

def LVM_DomainOp : LVM_Op<"domain", [Symbol, SymbolTable, NoTerminator]> {
  let summary = "immutable logical machine domain";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def LVM_SpaceOp : LVM_Op<"space", [Symbol, HasParent<"DomainOp">]> {
  let summary = "logical capacity and capability domain";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ArrayAttr:$capabilities,
    OptionalAttr<I64Attr>:$capacity,
    OptionalAttr<ArrayAttr>:$tags
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def LVM_StreamOp : LVM_Op<"stream", [Symbol, HasParent<"DomainOp">]> {
  let summary = "logical resource producer";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$produces,
    OptionalAttr<I64Attr>:$capacity,
    OptionalAttr<FlatSymbolRefAttr>:$produced_by,
    OptionalAttr<FlatSymbolRefAttr>:$transfer,
    OptionalAttr<FlatSymbolRefAttr>:$backing_region,
    OptionalAttr<UnitAttr>:$external
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def LVM_ChannelOp : LVM_Op<"channel", [Symbol, HasParent<"DomainOp">]> {
  let summary = "structurally available logical communication channel";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$from,
    FlatSymbolRefAttr:$to,
    ArrayAttr:$capabilities,
    OptionalAttr<I64Attr>:$capacity,
    OptionalAttr<StrAttr>:$direction
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def LVM_PlacementOp : LVM_Op<"placement", [Symbol, HasParent<"DomainOp">]> {
  let summary = "nontrivial logical support binding";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$space,
    I64Attr:$slot,
    OptionalAttr<AnyAttr>:$source_value,
    OptionalAttr<DictionaryAttr>:$binding
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def LVM_ActionSiteOp : LVM_Op<"action_site", [
    Symbol, HasParent<"DomainOp">
  ]> {
  let summary = "placed logical intent awaiting deterministic P2 selection";
  let description = [{
    Names one P1 action, instrument, preparation, or memory site independently
    of the executable SSA operation that references it.  P2 selection evidence
    refers back to this code-agnostic site; it is never stored on the P1 site.
    Compiler-materialized local action sites retain the deterministic `site`
    ordinal of their source `lvm.apply` as `source_site`.  Authored declarations
    may omit this link.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind,
      OptionalAttr<AnyAttr>:$objective, ArrayAttr:$placements,
      OptionalAttr<DictionaryAttr>:$parameters,
      OptionalAttr<I64Attr>:$source_site,
      OptionalAttr<SymbolRefAttr>:$channel,
      OptionalAttr<LVM_CapabilityAttr>:$channel_capability,
      OptionalAttr<ArrayAttr>:$endpoints, OptionalAttr<StrAttr>:$direction);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def LVM_ReturnOp : LVM_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "($results^ `:` type($results))? attr-dict";
  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, ::mlir::ValueRange{}); }]>];
}

def LVM_KernelOp : LVM_Op<"kernel", [
    Symbol, IsolatedFromAbove, SingleBlockImplicitTerminator<"ReturnOp">
  ]> {
  let summary = "P1 logical kernel statically bound to one domain";
  let description = [{
    Refines one P0 program onto a logical domain. For compiler-produced
    kernels, `estimate_only` presence and `specialization` exactly match
    `input_p0`; a marked kernel prohibits P2 or execution lowering. Detached
    unmarked executable kernels may omit `input_p0`.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$domain,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<FlatSymbolRefAttr>:$input_p0,
    OptionalAttr<StrAttr>:$placement_witness_sha256,
    OptionalAttr<UnitAttr>:$estimate_only,
    OptionalAttr<DictionaryAttr>:$specialization
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `on` $domain `:` $function_type attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def LVM_CallOp : LVM_Op<"call", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "placed instance of one retained P0 helper call";
  let description = [{
    Preserves the semantic P0 callee while materializing this particular call
    occurrence at the caller's placements.  The inline region is not a second
    reusable function: its block arguments and yield are the placement-refined
    call signature, so the same P0 helper may be instantiated at different
    logical locations without assigning machine facts to the P0 symbol.
    `scope` is a nonnegative call-occurrence identity unique within the
    containing kernel. It gives later stages typed provenance for this exact
    placed instance; it is not an arithmetic classification.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    I64Attr:$scope,
    Variadic<AnyType>:$inputs
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$callee `(` $inputs `)` attr-dict `:` functional-type($inputs, $results) $body";
  let hasVerifier = 1;
}

def LVM_PrepareOp : LVM_Op<"prepare", []> {
  let arguments = (ins
    StrAttr:$state,
    SymbolRefAttr:$at,
    OptionalAttr<I64Attr>:$site,
    OptionalAttr<DictionaryAttr>:$parameters
  );
  let results = (outs LVM_LogicalQubitType:$result);
  let assemblyFormat = "$state `at` $at attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

def LVM_ApplyOp : LVM_Op<"apply", []> {
  let arguments = (ins
    AnyAttr:$action,
    Variadic<AnyType>:$inputs,
    ArrayAttr:$placements,
    OptionalAttr<I64Attr>:$site,
    OptionalAttr<DictionaryAttr>:$parameters
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$action `(` $inputs `)` `at` $placements attr-dict `:` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def LVM_InstrumentOp : LVM_Op<"instrument", []> {
  let arguments = (ins
    AnyAttr:$instrument,
    Variadic<AnyType>:$inputs,
    ArrayAttr:$placements,
    OptionalAttr<I64Attr>:$site,
    OptionalAttr<DictionaryAttr>:$parameters,
    OptionalAttr<SymbolRefAttr>:$channel,
    OptionalAttr<LVM_CapabilityAttr>:$channel_capability,
    OptionalAttr<ArrayAttr>:$endpoints
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$instrument `(` $inputs `)` `at` $placements attr-dict `:` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def LVM_MeasureOp : LVM_Op<"measure", []> {
  let arguments = (ins
    QLX_PauliAttr:$basis,
    LVM_LogicalQubitType:$input,
    SymbolRefAttr:$at,
    OptionalAttr<I64Attr>:$site
  );
  let results = (outs I1:$result);
  let assemblyFormat = "$basis $input `at` $at attr-dict `:` qualified(type($input)) `->` type($result)";
  let hasVerifier = 1;
}

def LVM_IdleOp : LVM_Op<"idle", []> {
  let arguments = (ins
    Variadic<LVM_LogicalQubitType>:$inputs,
    Index:$rounds,
    ArrayAttr:$placements
  );
  let results = (outs Variadic<LVM_LogicalQubitType>:$results);
  let assemblyFormat = "$inputs `rounds` $rounds `at` $placements attr-dict `:` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def LVM_DiscardOp : LVM_Op<"discard", []> {
  let arguments = (ins
    Variadic<LVM_LogicalQubitType>:$inputs,
    ArrayAttr:$placements,
    OptionalAttr<StrAttr>:$reason
  );
  let assemblyFormat = "$inputs `at` $placements attr-dict `:` type($inputs)";
}

def LVM_ResourceRequestOp : LVM_Op<"resource_request", []> {
  let summary = "request a logical resource from one selected machine stream";
  let arguments = (ins StrAttr:$kind, SymbolRefAttr:$stream);
  let results = (outs Event_HandleType:$event);
  let assemblyFormat = "$kind `from` $stream attr-dict `:` qualified(type($event))";
}

def LVM_ConsumeResourceOp : LVM_Op<"consume_resource", []> {
  let arguments = (ins
    LVM_LogicalResourceType:$resource,
    AnyAttr:$action,
    Variadic<LVM_LogicalQubitType>:$inputs,
    FlatSymbolRefAttr:$resource_kind,
    SymbolRefAttr:$resource_stream,
    ArrayAttr:$placements,
    OptionalAttr<I64Attr>:$site
  );
  let results = (outs Variadic<LVM_LogicalQubitType>:$results);
  let assemblyFormat = "$resource `with` $action `(` $inputs `)` `at` $placements attr-dict `:` qualified(type($resource)) `,` functional-type($inputs, $results)";
  let hasVerifier = 1;
}

def LVM_FrameInitOp : LVM_Op<"frame_init", []> {
  let arguments = (ins StrAttr:$domain);
  let results = (outs LVM_LogicalFrameType:$frame);
  let assemblyFormat = "$domain attr-dict `:` qualified(type($frame))";
}

def LVM_FrameUpdateOp : LVM_Op<"frame_update", []> {
  let arguments = (ins LVM_LogicalFrameType:$frame, I1:$source, StrAttr:$update);
  let results = (outs LVM_LogicalFrameType:$result);
  let assemblyFormat = "$frame `from` $source attr-dict `:` qualified(type($frame)) `->` qualified(type($result))";
}

def LVM_FrameTransformOp : LVM_Op<"frame_transform", []> {
  let arguments = (ins LVM_LogicalFrameType:$frame, StrAttr:$transform);
  let results = (outs LVM_LogicalFrameType:$result);
  let assemblyFormat = "$frame attr-dict `:` qualified(type($frame)) `->` qualified(type($result))";
}

def LVM_XorOp : LVM_Op<"xor", []> {
  let summary = "placed classical parity of two runtime bits";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def LVM_YieldOp : LVM_Op<"yield", [
  Terminator, ParentOneOf<["CallOp"]>
]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, ::mlir::ValueRange{}); }]>];
}

#endif
