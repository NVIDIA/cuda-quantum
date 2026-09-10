/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_PHYS_OPS_TD
#define QLX_PHYS_OPS_TD

include "qlx/Dialect/Phys/IR/PhysDialect.td"
include "qlx/Dialect/Phys/IR/PhysTypes.td"
include "qlx/Dialect/QLX/IR/QLXAttrs.td"
include "qlx/Dialect/Event/IR/EventTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class Phys_Op<string mnemonic, list<Trait> traits = []>
    : Op<Phys_Dialect, mnemonic, traits>;

def Phys_ActionOp : Phys_Op<"action", [Symbol]> {
  let summary = "reusable physical instruction with a quantum process";
  let description = [{
    Names the physical instruction consumed by `phys.apply`. `process` is a
    deterministic qlx.simulation process payload and is independent of any
    backend. Optional controller bindings are transport spellings, not
    simulation semantics.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$arity,
    StrAttr:$process,
    OptionalAttr<DictionaryAttr>:$controller_bindings,
    OptionalAttr<ArrayAttr>:$parameters,
    OptionalAttr<UnitAttr>:$broadcast,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_InstrumentOp : Phys_Op<"instrument", [Symbol]> {
  let summary = "typed physical instruction that produces a classical record";
  let description = [{
    Instruments are separate from state-only actions.  A fixed positive arity
    or the `variadic` marker defines the nonempty physical-state boundary;
    `record_schema` defines the produced record family.  Resource classes
    explicitly advertise supported instruments.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    OptionalAttr<I64Attr>:$arity,
    OptionalAttr<UnitAttr>:$variadic,
    StrAttr:$record_schema,
    OptionalAttr<UnitAttr>:$preserves_inputs,
    StrAttr:$process,
    OptionalAttr<DictionaryAttr>:$controller_bindings,
    OptionalAttr<ArrayAttr>:$parameters,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_ArchitectureOp : Phys_Op<"machine", [
    Symbol, SymbolTable, NoTerminator
  ]> {
  let summary = "independent P3 physical carrier and action machine";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword regions";
  let hasVerifier = 1;
}

def Phys_ResourceClassOp : Phys_Op<"resource_class", [
    Symbol, HasParent<"ArchitectureOp">
  ]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    I64Attr:$count,
    ArrayAttr:$native_actions,
    OptionalAttr<StrAttr>:$granularity,
    OptionalAttr<StrAttr>:$physical_unit_kind,
    OptionalAttr<I64Attr>:$physical_units,
    OptionalAttr<StrAttr>:$footprint_evidence,
    OptionalAttr<ArrayAttr>:$native_instruments,
    OptionalAttr<ArrayAttr>:$capabilities
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_TopologyOp : Phys_Op<"topology", [
    Symbol, HasParent<"ArchitectureOp">
  ]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    OptionalAttr<DictionaryAttr>:$parameters,
    OptionalAttr<I64Attr>:$num_nodes,
    OptionalAttr<ArrayAttr>:$edges,
    OptionalAttr<ArrayAttr>:$coordinates,
    OptionalAttr<UnitAttr>:$strict
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_PatchTopologyOp : Phys_Op<"patch_topology", [
    Symbol, HasParent<"ArchitectureOp">
  ]> {
  let summary = "carrier-derived topology for implicit region patch slots";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$capacity,
    ArrayAttr:$carrier_groups,
    ArrayAttr:$categories,
    ArrayAttr:$edges,
    FlatSymbolRefAttr:$carrier_topology,
    FlatSymbolRefAttr:$resource_class
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_QECBindingOp : Phys_Op<"qec_binding", [
    Symbol, HasParent<"ArchitectureOp">
  ]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SymbolRefAttr:$qec_region,
    ArrayAttr:$resources,
    OptionalAttr<FlatSymbolRefAttr>:$topology,
    OptionalAttr<FlatSymbolRefAttr>:$patch_topology
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_QECChannelBindingOp : Phys_Op<"qec_channel_binding", [
    Symbol, HasParent<"ArchitectureOp">
  ]> {
  let summary = "bind one P2 interconnect to P3 physical resource pools";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SymbolRefAttr:$qec_channel,
    ArrayAttr:$resources,
    OptionalAttr<ArrayAttr>:$transport_claims,
    OptionalAttr<I64Attr>:$source_endpoint_occupancy,
    OptionalAttr<I64Attr>:$destination_endpoint_occupancy
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_OperatingPointOp : Phys_Op<"operating_point", [Symbol]> {
  let summary = "replaceable physical timing, calibration, and costs";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$machine,
    OptionalAttr<DictionaryAttr>:$timing,
    OptionalAttr<StrAttr>:$timing_source,
    OptionalAttr<DictionaryAttr>:$calibration,
    OptionalAttr<DictionaryAttr>:$costs,
    OptionalAttr<ArrayAttr>:$target_compatibility
  );
  let assemblyFormat = "$sym_name `for` $machine attr-dict";
  let hasVerifier = 1;
}

def Phys_ReturnOp : Phys_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, ::mlir::ValueRange{}); }]>];
}

def Phys_GraphOp : Phys_Op<"graph", [
    Symbol, IsolatedFromAbove, SingleBlockImplicitTerminator<"ReturnOp">
  ]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    FlatSymbolRefAttr:$architecture,
    OptionalAttr<FlatSymbolRefAttr>:$operating_point,
    OptionalAttr<FlatSymbolRefAttr>:$source_protocol,
    OptionalAttr<I64Attr>:$linear_value_count,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `on` $architecture `:` $function_type attr-dict-with-keyword regions";
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def Phys_SpacetimePlanOp : Phys_Op<"spacetime_plan", [
    Symbol, SymbolTable, NoTerminator
  ]> {
  let summary = "provider-authenticated physical spacetime DAG";
  let description = [{
    A compiler-produced P3 plan derived from one retained P2 protocol and one
    selected physical machine/operating point.  Provider-specific derivations
    populate this common representation; consumers inspect only its typed
    phases or events, dependencies, and resource claims.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$architecture,
    FlatSymbolRefAttr:$source_protocol,
    FlatSymbolRefAttr:$operating_point,
    StrAttr:$provider,
    StrAttr:$provider_version,
    StrAttr:$derivation,
    I64Attr:$derivation_version,
    OptionalAttr<DenseI64ArrayAttr>:$geometry,
    StrAttr:$evidence,
    OptionalAttr<F64Attr>:$forwarding_latency_ns,
    OptionalAttr<F64Attr>:$initiation_interval_ns,
    OptionalAttr<StrAttr>:$interval_semantics,
    OptionalAttr<StrAttr>:$policy,
    OptionalAttr<StrAttr>:$source_protocol_sha256,
    OptionalAttr<StrAttr>:$source_selected_protocol_sha256,
    OptionalAttr<StrAttr>:$source_objective_sha256,
    OptionalAttr<StrAttr>:$source_boundary_sha256,
    OptionalAttr<StrAttr>:$source_architecture_sha256,
    OptionalAttr<StrAttr>:$source_timing_source,
    OptionalAttr<DictionaryAttr>:$source_timing_profile,
    OptionalAttr<DenseI64ArrayAttr>:$source_code_distances,
    OptionalAttr<StrAttr>:$model_sha256,
    OptionalAttr<StrAttr>:$model_commitment,
    OptionalAttr<StrAttr>:$source_build_sha256,
    OptionalAttr<StrAttr>:$source_schedule_sha256,
    OptionalAttr<FlatSymbolRefAttr>:$recurrence_resource_kind,
    OptionalAttr<StrAttr>:$recurrence_output_event
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword regions";
  let hasVerifier = 1;
}

def Phys_SpacetimePhaseOp : Phys_Op<"spacetime_phase", [
    Symbol, HasParent<"SpacetimePlanOp">
  ]> {
  let summary = "one derived phase in a generic physical spacetime plan";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$steps,
    F64Attr:$step_duration_ns,
    ArrayAttr:$resource_classes,
    OptionalAttr<ArrayAttr>:$resource_claims,
    ArrayAttr:$factory_models,
    ArrayAttr:$after
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_SpacetimeEventOp : Phys_Op<"spacetime_event", [
    Symbol, HasParent<"SpacetimePlanOp">
  ]> {
  let summary = "one scheduled physical event in a recurring P3 layout";
  let description = [{
    A provider-produced physical event used to prove recurring capacity.  The
    event names an exact physical resource slice, a selected compact input
    factory when applicable, its dependencies, and an optional typed output.
    A recurrence carries several output events; characterization derives the
    cadence from their scheduled separation instead of accepting an authored
    interval scalar.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    I64Attr:$iteration,
    F64Attr:$start_ns,
    F64Attr:$duration_ns,
    OptionalAttr<SymbolRefAttr>:$resource_class,
    OptionalAttr<I64Attr>:$resource_offset,
    OptionalAttr<I64Attr>:$resource_count,
    OptionalAttr<FlatSymbolRefAttr>:$factory_model,
    OptionalAttr<I64Attr>:$factory_units,
    ArrayAttr:$after,
    OptionalAttr<FlatSymbolRefAttr>:$output_resource_kind
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_ResourceOp : Phys_Op<"resource", [
    Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    OptionalAttr<FlatSymbolRefAttr>:$architecture,
    FlatSymbolRefAttr:$resource_class,
    I64Attr:$index,
    OptionalAttr<SymbolRefAttr>:$qec_region,
    OptionalAttr<I64Attr>:$code_distance,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_AcquireOp : Phys_Op<"acquire", []> {
  let summary = "begin one ownership lifetime for declared resources";
  let arguments = (ins
    ArrayAttr:$resources,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$states);
  let assemblyFormat = "$resources attr-dict `:` type($states)";
  let hasVerifier = 1;
}

def Phys_ReleaseOp : Phys_Op<"release", []> {
  let summary = "release final ownership of declared physical resources";
  let arguments = (ins
    Variadic<Phys_StateType>:$states,
    OptionalAttr<StrAttr>:$event_id
  );
  let assemblyFormat = "$states attr-dict `:` type($states)";
  let hasVerifier = 1;
}

def Phys_PrepareOp : Phys_Op<"prepare", []> {
  let summary = "prepare physical resource states";
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    StrAttr:$state,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Phys_ResetOp : Phys_Op<"reset", []> {
  let summary = "reset physical resource states in a declared basis";
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    DefaultValuedAttr<StrAttr, "\"zero\"">:$state,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Phys_MoveOp : Phys_Op<"move", []> {
  let summary = "move carriers through an architecture route";
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    FlatSymbolRefAttr:$route,
    OptionalAttr<StrAttr>:$trajectory,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$route `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Phys_ApplyOp : Phys_Op<"apply", []> {
  let arguments = (ins
    FlatSymbolRefAttr:$action,
    Variadic<Phys_StateType>:$inputs,
    OptionalAttr<ArrayAttr>:$resources,
    OptionalAttr<FlatSymbolRefAttr>:$topology,
    OptionalAttr<I64Attr>:$batch_lanes,
    OptionalAttr<StrAttr>:$event_id,
    OptionalAttr<ArrayAttr>:$reservations
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$action `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Phys_MappingOp : Phys_Op<"mapping", [Symbol]> {
  let summary = "carrier embedding evidence for one physical graph";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$graph,
    FlatSymbolRefAttr:$source_graph,
    ArrayAttr:$initial,
    ArrayAttr:$final
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_RecordProjectionOp : Phys_Op<"record_projection", [Symbol]> {
  let summary = "map stable Fabric records to concrete physical record IDs";
  let description = [{
    Retains the finite, instance-qualified record lanes used by detached P3
    sidecars after P2-to-P3 lowering. Each entry contains `instance`,
    `source_record`, and `physical_record` strings and must be claimed by an
    aligned sidecar lane. A lane produced under folded physical control also
    carries aligned `repeat_events` and positive `repeat_counts`; this exact
    context denotes the bounded family of dynamic records without cloning the
    sidecar or spelling every iteration. Multiple physical records may project
    one source record across folded occurrences, and one physical record may
    carry multiple stable aliases such as an assembled-syndrome name.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$graph,
    FlatSymbolRefAttr:$source_protocol,
    ArrayAttr:$entries
  );
  let assemblyFormat = "$sym_name `for` $graph `from` $source_protocol attr-dict";
  let hasVerifier = 1;
}

def Phys_AllocationMappingOp : Phys_Op<"allocation_mapping", [Symbol]> {
  let summary = "bind invocation-local allocation lifetimes to physical carriers";
  let description = [{
    Each entry names one unique allocation, its acquire/release event IDs, and
    the concrete resource-class indices assigned for that lifetime. Distinct
    allocations may bind the same indices; the verified physical schedule
    must then prove that their resource intervals do not overlap. An optional
    `after` array names earlier mapped release events that provide explicit
    resource-order edges when graph SSA does not otherwise order reuse.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$graph,
    ArrayAttr:$entries
  );
  let assemblyFormat = "$sym_name `for` $graph attr-dict";
  let hasVerifier = 1;
}

def Phys_RoutingOp : Phys_Op<"routing", [Symbol]> {
  let summary = "explicit route evidence for inserted physical events";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$graph,
    FlatSymbolRefAttr:$topology,
    ArrayAttr:$steps
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_MeasureOp : Phys_Op<"measure", []> {
  let arguments = (ins
    FlatSymbolRefAttr:$measurement,
    Phys_StateType:$input,
    StrAttr:$record_id,
    OptionalAttr<StrAttr>:$event_id,
    OptionalAttr<UnitAttr>:$destructive
  );
  let results = (outs
    Optional<Phys_StateType>:$output,
    Phys_RecordType:$record
  );
  let assemblyFormat = "$measurement `(` $input `)` attr-dict `:` functional-type($input, results)";
  let hasVerifier = 1;
}

def Phys_MeasureProductOp : Phys_Op<"measure_product", []> {
  let summary = "native physical Pauli-product measurement event";
  let description = [{
    Measures one multi-qubit Pauli product natively (one label per state
    operand) and returns one record. The `invert` unit attribute marks a
    negated product (a Fabric-level `pauli_product = "-..."` projects here):
    the recorded bit is the complement of the unsigned product's outcome,
    matching Stim's inverted-target `MPP !P0*P1` semantics. Rotations carry
    no such flag because physical projection folds the product sign into the
    rotation angle.

    Legal only when every participating resource class advertises the typed
    instrument in `native_instruments` (conformance case 83). The verifier
    resolves each state operand's resource declaration and its resource
    class inside the architecture bound by the enclosing `phys.graph`;
    an unresolvable link is tolerated for partially-linked modules, but a
    resolved class that does not advertise the instrument is rejected.
  }];
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    FlatSymbolRefAttr:$instrument,
    ArrayAttr:$paulis,
    StrAttr:$record_id,
    OptionalAttr<UnitAttr>:$invert,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs
    Variadic<Phys_StateType>:$outputs,
    Phys_RecordType:$record
  );
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, results)";
  let hasVerifier = 1;
}

def Phys_RotateProductOp : Phys_Op<"rotate_product", []> {
  let summary = "native physical Pauli-product rotation event";
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    ArrayAttr:$paulis,
    F64Attr:$angle,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Phys_ResourceRotateProductOp : Phys_Op<"resource_rotate_product", []> {
  let summary = "physical Pauli-product rotation powered by one raw resource";
  let arguments = (ins
    Phys_ResourcePayloadType:$resource,
    Variadic<Phys_StateType>:$inputs,
    ArrayAttr:$paulis,
    F64Attr:$angle,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = [{
    $resource `on` $inputs attr-dict `:` functional-type(operands, $outputs)
  }];
  let hasVerifier = 1;
}

def Phys_DelayOp : Phys_Op<"delay", []> {
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    F64Attr:$duration_ns,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Phys_BarrierOp : Phys_Op<"barrier", []> {
  let summary = "synchronize declared physical states or clock domains";
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    OptionalAttr<ArrayAttr>:$domains,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def Phys_ProduceResourceOp : Phys_Op<"produce_resource", []> {
  let summary = "produce one typed physical resource payload";
  let arguments = (ins
    FlatSymbolRefAttr:$region,
    FlatSymbolRefAttr:$resource_kind,
    AnyAttr:$protocol,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Phys_ResourcePayloadType:$resource);
  let assemblyFormat = "$resource_kind `at` $region attr-dict `:` type($resource)";
  let hasVerifier = 1;
}

def Phys_TransportResourceOp : Phys_Op<"transport_resource", []> {
  let summary = "transport ownership through a selected physical resource route";
  let arguments = (ins
    Phys_ResourcePayloadType:$resource,
    FlatSymbolRefAttr:$source,
    FlatSymbolRefAttr:$destination,
    AnyAttr:$protocol,
    OptionalAttr<SymbolRefAttr>:$route,
    OptionalAttr<FlatSymbolRefAttr>:$model,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Phys_ResourcePayloadType:$result);
  let assemblyFormat = "$resource `from` $source `to` $destination attr-dict `:` type($resource) `->` type($result)";
  let hasVerifier = 1;
}

def Phys_TransportModelOp : Phys_Op<"transport_model", [Symbol]> {
  let summary = "authenticated compact physical transport model";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$architecture,
    FlatSymbolRefAttr:$operating_point,
    SymbolRefAttr:$qec_binding,
    SymbolRefAttr:$qec_channel,
    FlatSymbolRefAttr:$protocol,
    F64Attr:$latency_ns,
    F64Attr:$initiation_interval_ns,
    StrAttr:$interval_semantics,
    StrAttr:$policy,
    I64Attr:$source_endpoint_occupancy,
    I64Attr:$destination_endpoint_occupancy,
    ArrayAttr:$resource_claims,
    StrAttr:$provider,
    StrAttr:$provider_version,
    StrAttr:$evidence,
    StrAttr:$channel_sha256,
    StrAttr:$realization_sha256,
    StrAttr:$protocol_sha256,
    OptionalAttr<StrAttr>:$selected_protocol_sha256,
    StrAttr:$binding_sha256,
    StrAttr:$architecture_sha256,
    StrAttr:$model_sha256,
    OptionalAttr<StrAttr>:$model_commitment,
    OptionalAttr<StrAttr>:$timing_source,
    DictionaryAttr:$timing_profile,
    OptionalAttr<StrAttr>:$source_build_sha256,
    OptionalAttr<StrAttr>:$source_schedule_sha256
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_FactoryModelOp : Phys_Op<"factory_model", [Symbol]> {
  let summary = "authenticated deterministic compact physical factory model";
  let description = [{
    Closes one backed P1 resource stream at P3. The model binds the retained
    P2 producer protocol to its physical factory region, a dedicated compact
    physical-resource group, and deterministic startup/output timing from
    the selected operating point. `lane_count` is the number of identical
    provider instances; `output_interval_ns` is their aggregate bank cadence,
    obtained from the per-lane interval divided by `lane_count`;
    `physical_units` is the resource class's total
    authenticated base-unit footprint (including patch weights). `evidence`
    records the provenance of a declared or compiler-characterized model.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$resource_kind,
    SymbolRefAttr:$stream,
    FlatSymbolRefAttr:$provider,
    StrAttr:$provider_sha256,
    SymbolRefAttr:$region,
    FlatSymbolRefAttr:$qec_binding,
    FlatSymbolRefAttr:$physical_resource_class,
    I64Attr:$lane_count,
    I64Attr:$buffer_capacity,
    I64Attr:$physical_units,
    F64Attr:$startup_ns,
    F64Attr:$output_interval_ns,
    FlatSymbolRefAttr:$operating_point,
    StrAttr:$policy,
    StrAttr:$evidence,
    OptionalAttr<StrAttr>:$source_provider,
    OptionalAttr<StrAttr>:$source_provider_sha256,
    OptionalAttr<StrAttr>:$source_startup_cycles,
    OptionalAttr<StrAttr>:$source_output_interval_cycles,
    OptionalAttr<StrAttr>:$source_build_sha256,
    OptionalAttr<StrAttr>:$source_schedule_sha256,
    OptionalAttr<StrAttr>:$source_operating_point,
    OptionalAttr<StrAttr>:$source_timing_source,
    OptionalAttr<DictionaryAttr>:$source_timing_profile,
    OptionalAttr<ArrayAttr>:$source_output_events,
    OptionalAttr<ArrayAttr>:$source_selection_events,
    OptionalAttr<I64Attr>:$source_physical_units,
    OptionalAttr<StrAttr>:$source_physical_unit_kind,
    OptionalAttr<DenseI64ArrayAttr>:$source_code_distances
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Phys_FactoryStartOp : Phys_Op<"factory_start", []> {
  let summary = "start one deterministic compact physical factory model";
  let arguments = (ins FlatSymbolRefAttr:$factory_model, StrAttr:$event_id);
  let assemblyFormat = "$factory_model attr-dict";
  let hasVerifier = 1;
}

def Phys_UnpackResourceOp : Phys_Op<"unpack_resource", []> {
  let summary = "transfer a payload into exact selected-block carrier segments";
  let arguments = (ins
    Phys_ResourcePayloadType:$resource,
    Variadic<Phys_StateType>:$inputs,
    FlatSymbolRefAttr:$encoding,
    OptionalAttr<QLX_BuiltinActionAttr>:$payload_action,
    OptionalAttr<DenseI64ArrayAttr>:$payload_carrier_segments,
    OptionalAttr<ArrayAttr>:$payload_logical_block_ids,
    OptionalAttr<DenseI64ArrayAttr>:$payload_logical_blocks,
    OptionalAttr<DenseI64ArrayAttr>:$payload_logical_ports,
    OptionalAttr<ArrayAttr>:$payload_roles,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = [{
    $resource `into` `(` $inputs `)` attr-dict `:` functional-type(operands, $outputs)
  }];
  let hasVerifier = 1;
}

def Phys_PackResourceOp : Phys_Op<"pack_resource", []> {
  let summary = "transfer physical carrier ownership into a resource payload";
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    FlatSymbolRefAttr:$resource_kind,
    FlatSymbolRefAttr:$encoding,
    OptionalAttr<DenseI64ArrayAttr>:$payload_carrier_segments,
    OptionalAttr<ArrayAttr>:$payload_roles,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Phys_ResourcePayloadType:$resource);
  let assemblyFormat = [{
    `(` $inputs `)` `as` $resource_kind attr-dict `:` functional-type($inputs, $resource)
  }];
  let hasVerifier = 1;
}

def Phys_ResourceRequestOp : Phys_Op<"resource_request", []> {
  let summary = "submit one request to a placed physical resource provider";
  let arguments = (ins
    StrAttr:$kind,
    SymbolRefAttr:$stream,
    OptionalAttr<FlatSymbolRefAttr>:$factory_model,
    OptionalAttr<SymbolRefAttr>:$provider,
    OptionalAttr<SymbolRefAttr>:$region,
    OptionalAttr<SymbolRefAttr>:$transfer,
    OptionalAttr<SymbolRefAttr>:$physical_binding,
    OptionalAttr<F64Attr>:$duration_ns,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Event_HandleType:$event);
  let assemblyFormat = "$kind `from` $stream attr-dict `:` qualified(type($event))";
  let hasVerifier = 1;
}

def Phys_ConsumeResourceOp : Phys_Op<"consume_resource", []> {
  let summary = "legacy unresolved physical resource-action intent";
  let description = [{
    Compatibility syntax for an older projection that carried a logical action
    symbol into P3. Canonical QLX P3 uses `phys.unpack_resource` to hand
    the resource payload to allocated carriers and represents the selected
    realization as ordinary physical actions, instruments, decoding, and
    control. New lowering pipelines must not emit this operation.
  }];
  let arguments = (ins
    Phys_ResourcePayloadType:$resource,
    Variadic<Phys_StateType>:$inputs,
    FlatSymbolRefAttr:$action,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$resource `with` $action `(` $inputs `)` attr-dict `:` functional-type(operands, $outputs)";
  let hasVerifier = 1;
}

def Phys_DiscardResourcePayloadOp : Phys_Op<"discard_resource_payload", []> {
  let arguments = (ins Phys_ResourcePayloadType:$resource,
                       OptionalAttr<StrAttr>:$event_id);
  let assemblyFormat = "$resource attr-dict `:` type($resource)";
}

def Phys_ConditionOp : Phys_Op<"condition", []> {
  let summary = "derive a causally tracked control predicate from a record";
  let arguments = (ins
    AnyType:$source,
    DefaultValuedAttr<BoolAttr, "true">:$expected,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs I1:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
  let hasVerifier = 1;
}

def Phys_XorOp : Phys_Op<"xor", []> {
  let summary = "causally tracked parity of two physical runtime predicates";
  let arguments = (ins I1:$lhs, I1:$rhs, OptionalAttr<StrAttr>:$event_id);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Phys_AllFalseOp : Phys_Op<"all_false", []> {
  let summary = "causally tracked conjunction that every predicate is false";
  let arguments = (ins Variadic<I1>:$events, OptionalAttr<StrAttr>:$event_id);
  let results = (outs I1:$result);
  let assemblyFormat = "$events attr-dict `:` functional-type($events, $result)";
}

def Phys_EpochTransitionOp : Phys_Op<"epoch_transition", [
    SameVariadicOperandSize,
    SameVariadicResultSize
  ]> {
  let summary = "physical realization boundary for a QEC epoch transition";
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    FlatSymbolRefAttr:$source_epoch,
    FlatSymbolRefAttr:$destination_epoch,
    StrAttr:$evidence,
    OptionalAttr<DictionaryAttr>:$logical_map,
    StrAttr:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = [{
    $source_epoch `to` $destination_epoch `(` $inputs `)` attr-dict
    `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

def Phys_YieldOp : Phys_Op<"yield", [
    Terminator, ParentOneOf<["CallOp"]>
  ]> {
  let summary = "yield physical state and record versions from a folded call body";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ::mlir::ValueRange{}); }]>
  ];
}

def Phys_CallOp : Phys_Op<"call", []> {
  let summary = "folded physical realization of one Fabric call site";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    FlatSymbolRefAttr:$callee,
    StrAttr:$instance,
    OptionalAttr<FlatSymbolRefAttr>:$profile,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def Phys_CallTemplateOp : Phys_Op<"call_template", []> {
  let summary = "invoke one earlier exact-compatible physical call body";
  let description = [{
    Represents one real dynamic invocation of an earlier region-bearing
    `phys.call` without cloning its body.  The first form is deliberately
    strict: callee, compatible selected profile, and every non-state input and
    output type must match exactly. Concrete physical state resources may
    differ only through a complete, bijective, resource-class-compatible
    `state_aliases` mapping from the canonical body to this invocation.
    When the selected compiler generated physically equivalent per-site
    callees, `resource_substitutions` carries any nonidentity resource mapping
    as explicit `from`/`to` provenance and authorizes reuse of the canonical
    internal allocation plan. Equivalent callees that use the exact same
    physical resources require no substitution entries.
    Profile-bearing invocations may carry explicit
    `record_aliases` from invocation-local record IDs to records produced by
    the canonical body; aliases never imply positional rewriting.
    `state_boundary_elided` is the compact all-state form. It is valid
    only when the canonical call has an all-state, type-identical input/output
    boundary. The invocation either uses those exact physical resources or
    supplies a complete nonidentity `state_aliases` bijection. The canonical
    resource-effect summary then advances the resolved owners without
    repeating a potentially wide SSA boundary at every invocation.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    StrAttr:$template_event,
    FlatSymbolRefAttr:$callee,
    StrAttr:$instance,
    OptionalAttr<FlatSymbolRefAttr>:$profile,
    OptionalAttr<ArrayAttr>:$state_aliases,
    OptionalAttr<ArrayAttr>:$resource_substitutions,
    OptionalAttr<ArrayAttr>:$record_aliases,
    OptionalAttr<UnitAttr>:$state_boundary_elided,
    StrAttr:$event_id
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}
def Phys_SpacetimeCallOp : Phys_Op<"spacetime_call", []> {
  let summary = "folded invocation of one compiler-derived spacetime plan";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    FlatSymbolRefAttr:$plan,
    OptionalAttr<FlatSymbolRefAttr>:$source_protocol,
    StrAttr:$instance,
    StrAttr:$event_id
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}
def Phys_RetryOp : Phys_Op<"retry", []> {
  let summary = "bounded runtime replay policy over a projected attempt";
  let arguments = (ins
    Variadic<Phys_StateType>:$inputs,
    I1:$success,
    I64Attr:$max_attempts,
    DefaultValuedAttr<StrAttr, "\"report_failure\"">:$exhaustion,
    OptionalAttr<StrAttr>:$commit_point,
    OptionalAttr<F64Attr>:$success_probability,
    OptionalAttr<FlatSymbolRefAttr>:$success_probability_source,
    OptionalAttr<StrAttr>:$success_probability_evidence,
    FlatSymbolRefAttr:$attempt,
    FlatSymbolRefAttr:$profile,
    StrAttr:$attempt_event,
    StrAttr:$decision_event,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<Phys_StateType>:$outputs);
  let assemblyFormat = "$success `carries` `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

// Selection metadata remains separate from executable physical events. It
// binds stable Fabric success declarations to concrete physical record IDs.
def Phys_SelectionSidecarOp : Phys_Op<"selection_sidecar", [Symbol]> {
  let summary = "bind one Fabric success predicate to physical records";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$graph,
    OptionalAttr<FlatSymbolRefAttr>:$source_profile,
    OptionalAttr<StrAttr>:$source_kind,
    OptionalAttr<I64Attr>:$source_row,
    OptionalAttr<StrAttr>:$source_instance,
    OptionalAttr<FlatSymbolRefAttr>:$record_projection,
    OptionalAttr<DenseI64ArrayAttr>:$projection_indices,
    ArrayAttr:$records,
    OptionalAttr<ArrayAttr>:$source_records,
    OptionalAttr<ArrayAttr>:$input_syndromes,
    OptionalAttr<BoolAttr>:$constant,
    DefaultValuedAttr<BoolAttr, "false">:$expected,
    OptionalAttr<StrAttr>:$label,
    OptionalAttr<StrAttr>:$scope
  );
  let assemblyFormat = "$sym_name `for` $graph attr-dict";
  let hasVerifier = 1;
}

def Phys_HierarchyProjectionOp : Phys_Op<"hierarchy_projection", [Symbol]> {
  let summary = "map one folded encoding hierarchy to concrete resources";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$graph,
    FlatSymbolRefAttr:$source_hierarchy,
    ArrayAttr:$entries
  );
  let assemblyFormat = "$sym_name `for` $graph attr-dict";
  let hasVerifier = 1;
}

def Phys_ScheduleOp : Phys_Op<"schedule", [Symbol]> {
  let summary = "verified concrete schedule for a physical event graph";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$graph,
    StrAttr:$strategy,
    StrAttr:$strategy_domain,
    StrAttr:$provider,
    StrAttr:$provider_version,
    StrAttr:$constraint_profile,
    ArrayAttr:$constraints,
    DictionaryAttr:$timing_profile,
    StrAttr:$tie_break,
    StrAttr:$optimization_status,
    OptionalAttr<F64Attr>:$objective_value,
    ArrayAttr:$entries,
    F64Attr:$makespan_ns
  );
  let assemblyFormat = "$sym_name `for` $graph attr-dict";
  let hasVerifier = 1;
}

#endif
