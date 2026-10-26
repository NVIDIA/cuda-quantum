/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// Operation definitions for the Fabric dialect. ~40 operations covering:
//
//   - Device/region/code declarations
//   - Gadget definition and invocation
//   - Patch lifecycle (alloc, prep, dealloc)
//   - Physical gate operations (bulk single-qubit, two-qubit, transversal)
//   - Lattice surgery (merge, split, multi-body measure)
//   - Measurement and syndrome-record formation
//   - Success predicates and syndrome decoding
//   - Syndrome submission and decoding
//   - Pauli frame tracking
//   - Magic state production and injection
//   - Inter-region transport
//   - Timing, synchronization, and classical control
//
//===----------------------------------------------------------------------===//

#ifndef FABRIC_OPS_TD
#define FABRIC_OPS_TD

include "qlx/Dialect/Fabric/IR/FabricDialect.td"
include "qlx/Dialect/Fabric/IR/FabricTypes.td"
include "qlx/Dialect/Fabric/IR/FabricAttrs.td"
include "qlx/Dialect/Fabric/IR/FabricInterfaces.td"
include "qlx/Dialect/QLX/IR/QLXAttrs.td"

// Cross-dialect: fabric.resource_request returns the shared event handle type.
include "qlx/Dialect/Event/IR/EventTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class
//===----------------------------------------------------------------------===//

class Fabric_Op<string mnemonic, list<Trait> traits = []>
    : Op<Fabric_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Device and Code Declarations
//===----------------------------------------------------------------------===//

def Fabric_DeviceOp : Fabric_Op<"machine", [
    Symbol, IsolatedFromAbove, NoTerminator, SymbolTable
  ]> {
  let summary = "Declare a P2 QEC machine";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword regions";
}

def Fabric_RegionOp : Fabric_Op<"region", [
    Symbol, HasParent<"DeviceOp">
  ]> {
  let summary = "Declare a region within a device";
  let description = [{
    Declares a named region bound to a code type. Factory regions specify
    a `protocol` that produces the region's output resource state. The
    `protocol` attribute is either a `FlatSymbolRefAttr` pointing at a
    `fabric.gadget` (the compiled production protocol) or a
    `#fabric.spec_only<"name">` attribute naming an out-of-tree
    `ProductionSpec` (cost model only, no compiled body).
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$code,
    OptionalAttr<FlatSymbolRefAttr>:$encoding,
    OptionalAttr<FlatSymbolRefAttr>:$epoch,
    Fabric_RoleAttr:$role,
    Fabric_FloorplanAttr:$floorplan,
    OptionalAttr<I64Attr>:$block_capacity,
    OptionalAttr<StrAttr>:$packing,
    OptionalAttr<DenseI64ArrayAttr>:$ports,
    OptionalAttr<Fabric_RouteAttr>:$route,
    OptionalAttr<AnyAttrOf<[FlatSymbolRefAttr, Fabric_SpecOnlyAttr]>>:$protocol
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_InterconnectOp : Fabric_Op<"interconnect", [
    Symbol, HasParent<"DeviceOp">
  ]> {
  let summary = "Declare a selected P2 inter-region channel realization";
  let description = [{
    Declares a named link between two persistent region channel ports. When
    present, `logical_channel`, `capabilities`, `direction`, `concurrency`, and
    `provider` retain the selected refinement of a P1 `lvm.channel`.
    Each persistent endpoint retains its complete typed capability set,
    concurrency, provider identity, and provider-owned metadata. Metadata may
    describe lanes or other QEC geometry without adding provider-specific nouns
    to the Fabric dialect.

    The optional `protocol` attribute remains the executable transport
    realization: either a symbol reference to a typed `fabric.protocol` (the
    ordinary active-adapter form), a `fabric.gadget`, or a
    `#fabric.spec_only<"name">` attribute naming an out-of-tree
    `TransportSpec` used only for cost modeling.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$region_a,
    I64Attr:$port_a,
    FlatSymbolRefAttr:$region_b,
    I64Attr:$port_b,
    OptionalAttr<SymbolRefAttr>:$logical_channel,
    OptionalAttr<ArrayAttr>:$capabilities,
    OptionalAttr<StrAttr>:$direction,
    OptionalAttr<I64Attr>:$concurrency,
    OptionalAttr<StrAttr>:$provider,
    OptionalAttr<StrAttr>:$port_a_name,
    OptionalAttr<StrAttr>:$port_b_name,
    OptionalAttr<I64Attr>:$port_a_concurrency,
    OptionalAttr<I64Attr>:$port_b_concurrency,
    OptionalAttr<ArrayAttr>:$port_a_capabilities,
    OptionalAttr<ArrayAttr>:$port_b_capabilities,
    OptionalAttr<StrAttr>:$port_a_provider,
    OptionalAttr<StrAttr>:$port_b_provider,
    OptionalAttr<DictionaryAttr>:$port_a_metadata,
    OptionalAttr<DictionaryAttr>:$port_b_metadata,
    OptionalAttr<DictionaryAttr>:$metadata,
    OptionalAttr<AnyAttrOf<[FlatSymbolRefAttr, Fabric_SpecOnlyAttr]>>:$protocol,
    OptionalAttr<F64Attr>:$latency_ns
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_CodeOp : Fabric_Op<"code", [Symbol]> {
  let summary = "Declare a QEC code type";
  let description = [{
    Module-level declaration of a QEC code type. Defines partition structure,
    and optionally maps logical gates to gadget implementations.

    The `k` attribute is the number of logical qubits encoded in one block
    of this code. Defaults to 1. Codes with `k > 1` (qLDPC codes such as
    toric or bivariate bicycle) must declare canonical logical operator
    supports via `lx` and `lz`: each is an array of `k` inner arrays of
    data-qubit indices (same shape as `hx` / `hz`). These canonical logical
    supports are available to logical-operation lowering and decoding.

    A declaration that carries only this CSS support-array form (no
    canonical symplectic attributes) is verified directly over GF(2):
    every X-type support (`hx`/`gx`/`lx`) must intersect every Z-type
    support (`hz`/`gz`/`lz`) evenly, except that `lx[i]`/`lz[i]` and
    `gx[i]`/`gz[i]` partners must intersect oddly exactly on the diagonal;
    indices must lie in `[0, n)`; and when `n` (with optional `k`/`r`) is
    declared, the GF(2) ranks of `hx` plus `hz` must equal `n - k - r`.

    Example (k = 1, Steane):
    ```mlir
    fabric.code @steane {
      distance = 3 : i64,
      partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64}
    }
    ```

    Example (k = 2, toric L=3):
    ```mlir
    fabric.code @toric_3 {
      distance = 3 : i64,
      partitions = {data = 18 : i64, sx = 9 : i64, sz = 9 : i64},
      k = 2 : i64,
      lx = [[0, 1, 2], [9, 12, 15]],
      lz = [[0, 3, 6], [9, 10, 11]],
      hx = [...], hz = [...]
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$distance,
    DictionaryAttr:$partitions,
    OptionalAttr<DictionaryAttr>:$gate_map,
    OptionalAttr<ArrayAttr>:$hx,
    OptionalAttr<ArrayAttr>:$hz,
    OptionalAttr<ArrayAttr>:$gx,
    OptionalAttr<ArrayAttr>:$gz,
    OptionalAttr<I64Attr>:$stabilize_rounds,
    OptionalAttr<I64Attr>:$k,
    OptionalAttr<ArrayAttr>:$lx,
    OptionalAttr<ArrayAttr>:$lz,
    OptionalAttr<I64Attr>:$n,
    OptionalAttr<I64Attr>:$r,
    OptionalAttr<ArrayAttr>:$stabilizers,
    OptionalAttr<ArrayAttr>:$gauges,
    OptionalAttr<AnyAttr>:$stabilizer_basis,
    OptionalAttr<AnyAttr>:$logical_x_basis,
    OptionalAttr<AnyAttr>:$logical_z_basis,
    OptionalAttr<AnyAttr>:$gauge_x_basis,
    OptionalAttr<AnyAttr>:$gauge_z_basis,
    OptionalAttr<AnyAttr>:$anti_stabilizers,
    OptionalAttr<AnyAttr>:$encoding_clifford,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_CodeProfileOp : Fabric_Op<"code_profile", [Symbol]> {
  let summary = "evidence-bearing operating profile for one QEC code";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$code,
    OptionalAttr<I64Attr>:$distance_claim,
    OptionalAttr<StrAttr>:$distance_status,
    OptionalAttr<AnyAttr>:$effective_stabilizers,
    OptionalAttr<AnyAttr>:$decomposition,
    OptionalAttr<AnyAttr>:$effective_metachecks,
    OptionalAttr<AnyIntElementsAttr>:$kept_from_effective,
    OptionalAttr<DictionaryAttr>:$metachecks,
    OptionalAttr<DictionaryAttr>:$gauge_measurements,
    OptionalAttr<ArrayAttr>:$dynamic_phases,
    OptionalAttr<AnyIntElementsAttr>:$period_closure,
    OptionalAttr<AnyAttr>:$temporal_recovery,
    OptionalAttr<AnyAttr>:$temporal_recovery_targets,
    OptionalAttr<DictionaryAttr>:$record_logicals,
    OptionalAttr<ArrayAttr>:$evidence,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_EncodingOp : Fabric_Op<"encoding", [Symbol]> {
  let summary = "logical interface and block view over a code profile";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$code,
    FlatSymbolRefAttr:$profile,
    StrAttr:$block,
    ArrayAttr:$logical_ports,
    OptionalAttr<DictionaryAttr>:$layout,
    OptionalAttr<FlatSymbolRefAttr>:$hierarchy,
    OptionalAttr<FlatSymbolRefAttr>:$epoch_schema,
    OptionalAttr<FlatSymbolRefAttr>:$initial_epoch,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_PatchGraphOp : Fabric_Op<"patch_graph", [Symbol]> {
  let summary = "derived structured interaction graph for one P2 root";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$root,
    ArrayAttr:$nodes,
    ArrayAttr:$interactions
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_PatchMappingOp : Fabric_Op<"patch_mapping", [Symbol]> {
  let summary = "chosen mapping from P2 patch instances to implicit slots";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$graph,
    ArrayAttr:$assignments
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_EncodingEpochSchemaOp : Fabric_Op<"encoding_epoch_schema", [Symbol]> {
  let summary = "finite phase and transition schema for one encoding";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ArrayAttr:$phases,
    StrAttr:$initial,
    ArrayAttr:$transitions,
    OptionalAttr<DictionaryAttr>:$logical_maps,
    OptionalAttr<UnitAttr>:$periodic,
    OptionalAttr<AnyIntElementsAttr>:$closure
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_EncodingEpochOp : Fabric_Op<"encoding_epoch", [Symbol]> {
  let summary = "one immutable encoding phase instance";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$encoding,
    FlatSymbolRefAttr:$schema,
    StrAttr:$phase,
    I64Attr:$index
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Fabric_EpochTransitionOp : Fabric_Op<"epoch_transition", []> {
  let summary = "apply one verified dynamic-encoding phase transition";
  let arguments = (ins
    Fabric_PatchType:$patch,
    FlatSymbolRefAttr:$to_epoch,
    StrAttr:$evidence,
    OptionalAttr<DictionaryAttr>:$logical_map
  );
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = [{
    $patch `to` $to_epoch attr-dict `:` functional-type($patch, $result)
  }];
  let hasVerifier = 1;
}

def Fabric_EncodingHierarchyOp : Fabric_Op<"encoding_hierarchy", [Symbol]> {
  let summary = "folded child-slot structure for one hierarchical encoding";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$code,
    FlatSymbolRefAttr:$outer,
    FlatSymbolRefAttr:$child,
    I64Attr:$multiplicity,
    ArrayAttr:$carrier_map,
    OptionalAttr<ArrayAttr>:$exposed_ports,
    OptionalAttr<ArrayAttr>:$gauge_ports,
    OptionalAttr<ArrayAttr>:$fixed_ports,
    FlatSymbolRefAttr:$flat_encoding,
    I64Attr:$depth,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name `for` $code attr-dict";
  let hasVerifier = 1;
}

def Fabric_EncodingProjectionOp : Fabric_Op<"encoding_projection", [Symbol]> {
  let summary = "algebraic projection between equivalent encoding views";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$source,
    FlatSymbolRefAttr:$destination,
    DenseI64ArrayAttr:$carrier_map,
    DenseI64ArrayAttr:$logical_map,
    StrAttr:$evidence,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name `from` $source `to` $destination attr-dict";
}

def Fabric_PatchTransformOp : Fabric_Op<"patch_transform", [Symbol]> {
  let summary = "linear code/encoding/support transformation contract";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$source,
    FlatSymbolRefAttr:$destination,
    DictionaryAttr:$frame_partitions,
    DenseI64ArrayAttr:$source_support,
    DenseI64ArrayAttr:$destination_support,
    DictionaryAttr:$source_roles,
    DictionaryAttr:$destination_roles,
    DenseI64ArrayAttr:$logical_map,
    StrAttr:$evidence,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name `from` $source `to` $destination attr-dict";
  let hasVerifier = 1;
}

def Fabric_TransformBeginOp : Fabric_Op<"transform_begin", []> {
  let summary = "consume a source patch and expose its local carrier frame";
  let arguments = (ins
    Fabric_PatchType:$source,
    FlatSymbolRefAttr:$transform
  );
  let results = (outs Fabric_PatchFrameType:$frame);
  let assemblyFormat = [{
    $source `using` $transform attr-dict `:` functional-type($source, $frame)
  }];
  let hasVerifier = 1;
}

def Fabric_TransformEndOp : Fabric_Op<"transform_end", []> {
  let summary = "commit a carrier frame as the destination encoded patch";
  let arguments = (ins
    Fabric_PatchFrameType:$frame,
    FlatSymbolRefAttr:$transform
  );
  let results = (outs Fabric_PatchType:$destination);
  let assemblyFormat = [{
    $frame `using` $transform attr-dict `:` functional-type($frame, $destination)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Gadget Definition and Invocation
//===----------------------------------------------------------------------===//

def Fabric_ObjectiveOp : Fabric_Op<"objective", [Symbol]> {
  let summary = "typed logical or resource-flow objective referenced by a realization";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<FlatSymbolRefAttr>:$logical,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictionaryAttr>:$subsystem_fragment,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name (`implements` $logical^)? `:` $function_type attr-dict";
  let hasVerifier = 1;
}

def Fabric_GadgetSpecOp : Fabric_Op<"gadget_spec", [Symbol]> {
  let summary = "objective and typed QEC boundary, separate from realization";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$objective,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<UnitAttr>:$entrypoint,
    ArrayAttr:$encodings,
    OptionalAttr<ArrayAttr>:$ports,
    OptionalAttr<ArrayAttr>:$flows,
    OptionalAttr<DictionaryAttr>:$logical_ports,
    OptionalAttr<StrAttr>:$action_equivalence,
    OptionalAttr<StrAttr>:$epoch_map,
    OptionalAttr<FlatSymbolRefAttr>:$transform,
    OptionalAttr<ArrayAttr>:$record_schema,
    OptionalAttr<DictionaryAttr>:$outcome_map,
    OptionalAttr<DictionaryAttr>:$parameter_map,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let assemblyFormat = "$sym_name `for` $objective `:` $function_type attr-dict";
  let hasVerifier = 1;
}

def Fabric_ProfileEndOp : Fabric_Op<"profile_end", [
    Terminator, HasParent<"GadgetProfileOp">
  ]> {
  let summary = "terminate a declarative gadget-analysis profile";
  let assemblyFormat = "attr-dict";
}

def Fabric_GadgetProfileOp : Fabric_Op<"gadget_profile", [
    Symbol, IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ProfileEndOp">
  ]> {
  let summary = "success and boundary analysis overlay for one gadget";
  let description = [{
    References one executable gadget realization while separately declaring a
    success and boundary interpretation. More than one profile may reference
    the same gadget. Profile bodies are declarative: success and output
    syndrome operations refer to stable measurement records through `records`
    attributes rather than capturing SSA values from the gadget body.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$gadget,
    OptionalAttr<ArrayAttr>:$inputs,
    OptionalAttr<ArrayAttr>:$outputs,
    OptionalAttr<UnitAttr>:$boundary_complete,
    OptionalAttr<AnyAttr>:$bit_order,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name `for` $gadget attr-dict-with-keyword regions";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Gadget analysis declarations
//===----------------------------------------------------------------------===//

def Fabric_SuccessOp : Fabric_Op<"success", []> {
  let summary = "declare a success or postselection predicate over records";
  let description = [{
    Declares a mismatch parity over stable measurement records in a
    `fabric.gadget_profile`: zero means accepted. It is analysis semantics, not
    executable branch control. Retry, discard, and abort behavior belong to
    the enclosing protocol's explicit selection or retry operation.
  }];
  let arguments = (ins
    OptionalAttr<ArrayAttr>:$records,
    OptionalAttr<ArrayAttr>:$input_syndromes,
    OptionalAttr<BoolAttr>:$constant
  );
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

def Fabric_OutputSyndromeOp : Fabric_Op<"output_syndrome", []> {
  let summary = "assign one effective syndrome bit at an output profile port";
  let description = [{
    Compiler-normalized scalar row defining one output effective-syndrome
    component as an affine parity over incoming syndrome bits and stable
    realization records. Source frontends bind whole typed syndrome bundles;
    ``port_index`` is the output endpoint ordinal in the gadget interface.
  }];
  let arguments = (ins
    I64Attr:$port_index,
    I64Attr:$index,
    OptionalAttr<ArrayAttr>:$records,
    OptionalAttr<ArrayAttr>:$input_syndromes,
    OptionalAttr<BoolAttr>:$constant
  );
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

def Fabric_ReturnOp : Fabric_Op<"return", [
    Terminator, ParentOneOf<["GadgetOp", "CircuitOp"]>
  ]> {
  let summary = "Return from a gadget or reusable circuit body";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

def Fabric_CircuitOp : Fabric_Op<"circuit", [
    IsolatedFromAbove, Symbol,
    SingleBlockImplicitTerminator<"ReturnOp">
  ]> {
  let summary = "reusable physical/QEC realization body";
  let description = [{
    A callable realization-side body with no logical objective, gadget claim,
    analysis profile, schedule, or noise semantics of its own. One or more
    gadgets may reference the circuit through their `realization` attribute.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictionaryAttr>:$metadata
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name,
                   "::mlir::FunctionType":$type,
                   CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>",
                        "{}">:$attrs)>
  ];
}

def Fabric_GadgetOp : Fabric_Op<"gadget", [
    IsolatedFromAbove, Symbol,
    SingleBlockImplicitTerminator<"ReturnOp">
  ]> {
  let summary = "Define a gadget";
  let description = [{
    Defines a composable gadget — the universal abstraction for physical
    QEC operations on patches. Gadgets can represent syndrome extraction,
    gates, factory protocols, or full programs.

    Entry gadgets have `{entry}` and bind to a device.
    The optional `flow` attribute declares the stabilizer transformation.

    Example:
    ```mlir
    fabric.gadget @logical_h(
        %p: !fabric.patch<@sc>, %syn: !fabric.syndrome<@sc>)
        -> (!fabric.patch<@sc>, !fabric.syndrome<@sc>)
        flow #fabric.flow<{x = "z", z = "x"}>
    {
      ...
      fabric.return %p_out, %syn_out : ...
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<UnitAttr>:$entry,
    OptionalAttr<FlatSymbolRefAttr>:$device,
    OptionalAttr<Fabric_FlowAttr>:$flow,
    OptionalAttr<FlatSymbolRefAttr>:$spec,
    OptionalAttr<DictionaryAttr>:$realization_boundary,
    OptionalAttr<FlatSymbolRefAttr>:$realization,
    OptionalAttr<StrAttr>:$realization_kind,
    OptionalAttr<FlatSymbolRefAttr>:$generated_by,
    OptionalAttr<FlatSymbolRefAttr>:$transform,
    OptionalAttr<SymbolRefAttr>:$action_site,
    OptionalAttr<DictionaryAttr>:$specialization
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name,
                   "::mlir::FunctionType":$type,
                   CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>",
                        "{}">:$attrs)>
  ];
}

def Fabric_CallOp : Fabric_Op<"call", []> {
  let summary = "Invoke a gadget";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    OptionalAttr<FlatSymbolRefAttr>:$profile,
    Variadic<AnyType>:$operands,
    OptionalAttr<SymbolRefAttr>:$channel,
    OptionalAttr<AnyAttr>:$channel_capability,
    OptionalAttr<ArrayAttr>:$endpoints,
    OptionalAttr<SymbolRefAttr>:$action_site,
    OptionalAttr<FlatSymbolRefAttr>:$generated_by
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];
  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "::mlir::FlatSymbolRefAttr":$callee,
                   "::mlir::ValueRange":$operands), [{
      build($_builder, $_state, resultTypes, callee,
            ::mlir::FlatSymbolRefAttr{}, operands, ::mlir::SymbolRefAttr{},
            ::mlir::Attribute{}, ::mlir::ArrayAttr{},
            ::mlir::SymbolRefAttr{}, ::mlir::FlatSymbolRefAttr{});
    }]>
  ];
  let hasVerifier = 1;
}

def Fabric_RelocateOp : Fabric_Op<"relocate", [
    AllTypesMatch<["patch", "result"]>
  ]> {
  let summary = "apply one selected nonlocal placement transition";
  let description = [{
    Consumes one encoded owner at a trajectory boundary and returns its unique
    successor after invoking the selected gadget/protocol realization. Source
    and destination are nested machine-space references retained from P1;
    `transition` and `continuity_witness` preserve the authored worldline
    evidence. This is executable placement lowering, not a placement hint.
  }];
  let arguments = (ins
    Fabric_PatchType:$patch,
    FlatSymbolRefAttr:$callee,
    SymbolRefAttr:$source,
    SymbolRefAttr:$destination,
    StrAttr:$transition,
    StrAttr:$continuity_witness,
    I64Attr:$step,
    OptionalAttr<FlatSymbolRefAttr>:$generated_by
  );
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = [{
    $patch `using` $callee `from` $source `to` $destination attr-dict
    `:` type($patch)
  }];
  let hasVerifier = 1;
}

def Fabric_EstablishSupportOp : Fabric_Op<"establish_support", [
    AllTypesMatch<["patch", "result"]>
  ]> {
  let summary = "establish one selected distributed encoding support";
  let description = [{
    Consumes the unique encoded owner initially resident in `primary`, invokes
    a selected gadget or protocol that realizes the declared links/support,
    and returns the same encoded owner with support distributed across the
    ordered `spaces`.  `support_views` is aligned one-to-one with `spaces`.
    The ownership witness and link obligations are semantic evidence retained
    from P1; this operation is never a free metadata-only placement rewrite.
  }];
  let arguments = (ins
    Fabric_PatchType:$patch,
    FlatSymbolRefAttr:$callee,
    SymbolRefAttr:$primary,
    ArrayAttr:$spaces,
    ArrayAttr:$support_views,
    StrAttr:$ownership_witness,
    ArrayAttr:$link_obligations,
    OptionalAttr<FlatSymbolRefAttr>:$generated_by
  );
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = [{
    $patch `using` $callee `primary` $primary `across` $spaces attr-dict
    `:` type($patch)
  }];
  let hasVerifier = 1;
}

def Fabric_EstablishTopologicalRecordOp : Fabric_Op<
    "establish_topological_record", [AllTypesMatch<["patch", "result"]>]> {
  let summary = "establish one selected topological-record placement";
  let description = [{
    Invokes the selected gadget or protocol that establishes or validates a
    logical degree represented by a worldline/correlation record at the given
    frontier.  The operation preserves one encoded owner and retains separate
    support and logical-observable witnesses.  It is executable placement
    lowering, not an assertion that an arbitrary local patch is topological.
  }];
  let arguments = (ins
    Fabric_PatchType:$patch,
    FlatSymbolRefAttr:$callee,
    SymbolRefAttr:$space,
    StrAttr:$record,
    ArrayAttr:$frontier,
    StrAttr:$support_witness,
    StrAttr:$observable_witness,
    OptionalAttr<FlatSymbolRefAttr>:$generated_by
  );
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = [{
    $patch `using` $callee `at` $space `record` $record attr-dict
    `:` type($patch)
  }];
  let hasVerifier = 1;
}

def Fabric_RetryOp : Fabric_Op<"retry", []> {
  let summary = "bounded folded replay policy for one selection-bearing attempt";
  let description = [{
    Consumes the current attempt's linear outputs and returns the same boundary
    under a bounded retry policy.  `success` is the attempt predicate.  The
    referenced call/profile remains explicit in the surrounding protocol; this
    op does not clone its body.  Runtime lowering expands or orchestrates the
    attempt according to commit-point and exhaustion semantics.
  }];
  let arguments = (ins
    Variadic<Fabric_PatchType>:$carries,
    I1:$success,
    I64Attr:$max_attempts,
    DefaultValuedAttr<StrAttr, "\"report_failure\"">:$exhaustion,
    OptionalAttr<StrAttr>:$commit_point,
    OptionalAttr<F64Attr>:$success_probability,
    OptionalAttr<FlatSymbolRefAttr>:$success_probability_source,
    OptionalAttr<StrAttr>:$success_probability_evidence,
    OptionalAttr<FlatSymbolRefAttr>:$attempt,
    OptionalAttr<FlatSymbolRefAttr>:$profile
  );
  let results = (outs Variadic<Fabric_PatchType>:$results);
  let assemblyFormat = [{
    $success `carries` `(` $carries `)` attr-dict
    `:` functional-type($carries, $results)
  }];
  let hasVerifier = 1;
}

def Fabric_ProtocolReturnOp : Fabric_Op<"protocol_return", [
    Terminator, HasParent<"ProtocolOp">
  ]> {
  let summary = "return typed values from a folded QEC protocol";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

def Fabric_ProtocolOp : Fabric_Op<"protocol", [
    Symbol, IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ProtocolReturnOp">
  ]> {
  let summary = "folded typed network of gadget and protocol calls";
  let description = [{
    Defines a reusable P2N protocol graph.  Its region may contain typed
    `fabric.call` operations, structured control, selection/retry policy,
    resource flows, and explicit ownership-changing operations.  A protocol
    is not an implementation registry and does not copy its callees.

    A protocol derived from one placed P1 helper occurrence carries
    `input_p1_kernel`, `input_p1_callee`, and `input_p1_scope` together. The
    verifier resolves that tuple to exactly one retained `lvm.call`; these
    attributes preserve source provenance and do not classify the helper by
    name. Arbitrary helper semantics remain in the retained callable body;
    Fabric does not attach workload-specific role or shape metadata.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<AnyAttr>:$objective,
    OptionalAttr<FlatSymbolRefAttr>:$generated_by,
    OptionalAttr<SymbolRefAttr>:$action_site,
    OptionalAttr<DictionaryAttr>:$specialization,
    OptionalAttr<DictionaryAttr>:$metadata,
    OptionalAttr<FlatSymbolRefAttr>:$predicate_gadget,
    OptionalAttr<FlatSymbolRefAttr>:$predicate_profile,
    OptionalAttr<I64Attr>:$predicate_result,
    OptionalAttr<FlatSymbolRefAttr>:$input_p1_kernel,
    OptionalAttr<FlatSymbolRefAttr>:$input_p1_callee,
    OptionalAttr<I64Attr>:$input_p1_scope
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `:` $function_type attr-dict-with-keyword regions";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Folded encoding-hierarchy ownership
//===----------------------------------------------------------------------===//

def Fabric_EncodingUnpackOp : Fabric_Op<"encoding_unpack", []> {
  let summary = "consume a hierarchical parent patch and expose one folded child group";
  let arguments = (ins
    Fabric_PatchType:$parent,
    FlatSymbolRefAttr:$hierarchy,
    FlatSymbolRefAttr:$slot_group
  );
  let results = (outs Fabric_PatchBundleType:$children);
  let assemblyFormat = [{
    $parent `using` $hierarchy `group` $slot_group attr-dict
    `:` functional-type($parent, $children)
  }];
  let hasVerifier = 1;
}

def Fabric_MapChildrenOp : Fabric_Op<"map_children", [
    AllTypesMatch<["children", "result"]>
  ]> {
  let summary = "apply one verified callee uniformly to a folded child group";
  let arguments = (ins
    Fabric_PatchBundleType:$children,
    FlatSymbolRefAttr:$callee,
    FlatSymbolRefAttr:$slot_group,
    OptionalAttr<FlatSymbolRefAttr>:$record_schema
  );
  let results = (outs Fabric_PatchBundleType:$result);
  let assemblyFormat = [{
    $callee `to` $children `group` $slot_group attr-dict
    `:` functional-type($children, $result)
  }];
  let hasVerifier = 1;
}

def Fabric_EncodingPackOp : Fabric_Op<"encoding_pack", []> {
  let summary = "consume one complete child bundle and restore its parent owner";
  let arguments = (ins
    Fabric_PatchBundleType:$children,
    FlatSymbolRefAttr:$encoding
  );
  let results = (outs Fabric_PatchType:$parent);
  let assemblyFormat = [{
    $children `as` $encoding attr-dict
    `:` functional-type($children, $parent)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Patch Lifecycle
//===----------------------------------------------------------------------===//

def Fabric_AllocOp : Fabric_Op<"alloc", []> {
  let summary = "Allocate a patch";
  let arguments = (ins
    FlatSymbolRefAttr:$code,
    FlatSymbolRefAttr:$region,
    OptionalAttr<UnitAttr>:$strict_region,
    OptionalAttr<Fabric_PrepAttr>:$prep
  );
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Fabric_PrepZOp : Fabric_Op<"prep_z", [AllTypesMatch<["patch", "result"]>]> {
  let summary = "Prepare patch in Z basis (logical |0>)";
  let arguments = (ins Fabric_PatchType:$patch);
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = "$patch attr-dict `:` type($result)";
}

def Fabric_PrepXOp : Fabric_Op<"prep_x", [AllTypesMatch<["patch", "result"]>]> {
  let summary = "Prepare patch in X basis (logical |+>)";
  let arguments = (ins Fabric_PatchType:$patch);
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = "$patch attr-dict `:` type($result)";
}

def Fabric_DeallocOp : Fabric_Op<"dealloc", []> {
  let summary = "Release a patch";
  let arguments = (ins Fabric_PatchType:$patch);
  let assemblyFormat = "$patch attr-dict `:` type($patch)";
}

//===----------------------------------------------------------------------===//
// Physical Gate Operations — Bulk single-patch gates
//===----------------------------------------------------------------------===//

class Fabric_BulkGateOp<string mnemonic, string gateName>
    : Fabric_Op<mnemonic, [AllTypesMatch<["patch", "result"]>]> {
  let summary = "Apply bulk " # gateName # " to patch partition";
  let arguments = (ins
    Fabric_PatchLikeType:$patch,
    Fabric_PartitionAttr:$partition,
    OptionalAttr<DenseI64ArrayAttr>:$indices
  );
  let results = (outs Fabric_PatchLikeType:$result);
  let assemblyFormat = [{
    $patch custom<PartitionKeyword>($partition)
    custom<OptionalIndices>($indices)
    attr-dict `:` type($result)
  }];
}

def Fabric_HOp   : Fabric_BulkGateOp<"h",     "Hadamard">;
def Fabric_SOp   : Fabric_BulkGateOp<"s",     "phase (S)">;
def Fabric_SdgOp : Fabric_BulkGateOp<"sdg",   "S-dagger">;
def Fabric_XOp   : Fabric_BulkGateOp<"x",     "Pauli-X">;
def Fabric_ZOp   : Fabric_BulkGateOp<"z",     "Pauli-Z">;
def Fabric_TOp   : Fabric_BulkGateOp<"t",     "T (pi/8)">;
def Fabric_TdgOp : Fabric_BulkGateOp<"tdg",   "T-dagger">;
def Fabric_ResetOp : Fabric_BulkGateOp<"reset", "reset">;

def Fabric_InitBasisOp : Fabric_Op<"init_basis", [
    AllTypesMatch<["patch", "result"]>
  ]> {
  let summary = "Atomically initialize a carrier partition in a typed basis";
  let description = [{
    P2 atomic ancilla-initialization step. `basis = #fabric.prep<x>` prepares
    |+> and `basis = #fabric.prep<z>` prepares |0>. A target may realize this
    operation natively or decompose it at P3/target lowering; that realization
    does not change the authored P2 syndrome-extraction depth.
  }];
  let arguments = (ins
    Fabric_PatchLikeType:$patch,
    Fabric_PartitionAttr:$partition,
    OptionalAttr<DenseI64ArrayAttr>:$indices,
    Fabric_PrepAttr:$basis
  );
  let results = (outs Fabric_PatchLikeType:$result);
  let assemblyFormat = [{
    $patch custom<PartitionKeyword>($partition)
    custom<OptionalIndices>($indices)
    `basis` $basis attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Permutation automorphism
//===----------------------------------------------------------------------===//

def Fabric_PermuteOp : Fabric_Op<"permute", [AllTypesMatch<["patch", "result"]>]> {
  let summary = "Relabel a patch's data qubits by a code-automorphism permutation";
  let description = [{
    Noiseless permutation of the physical data qubits of a logical patch by a
    code automorphism (one that preserves the X- and Z-stabilizer groups). On
    trapped-ion hardware this is a relabeling — a "CNOT for free" in the sense of
    Reichardt et al. (arXiv:2409.04628), where two of the three Path-4 CNOTs are
    realized purely by permuting qubits. In simulation it lowers to a noiseless
    SWAP network. `perm` is a permutation of the data-qubit indices: the qubit at
    position i moves to position `perm[i]`. The walker transforms logical
    supports through the automorphism into the post-permute frame.
    The verifier independently derives the action from the referenced
    canonical code algebra.  The typed action attributes are evidence to
    compare, not author assertions trusted by downstream consumers.
  }];
  let arguments = (ins
    Fabric_PatchType:$patch,
    DenseI64ArrayAttr:$perm,
    OptionalAttr<ArrayAttr>:$logical_x_action,
    OptionalAttr<ArrayAttr>:$logical_z_action,
    OptionalAttr<I64Attr>:$protected_logicals,
    OptionalAttr<I64Attr>:$gauge_qubits,
    OptionalAttr<StrAttr>:$derivation,
    OptionalAttr<DictionaryAttr>:$objective_binding
  );
  let results = (outs Fabric_PatchType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $patch `perm` $perm attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Measurement
//===----------------------------------------------------------------------===//

def Fabric_MzOp : Fabric_Op<"mz", [
    AllTypesMatch<["patch", "patch_out"]>,
    DeclareOpInterfaceMethods<Fabric_MeasurementBundleOpInterface>
  ]> {
  let summary = "Bulk Z-basis measurement";
  let arguments = (ins
    Fabric_PatchLikeType:$patch,
    Fabric_PartitionAttr:$partition,
    OptionalAttr<DenseI64ArrayAttr>:$indices,
    OptionalAttr<StrAttr>:$record
  );
  let results = (outs
    Fabric_PatchLikeType:$patch_out,
    AnyRankedTensor:$bits
  );
  let assemblyFormat = [{
    $patch custom<PartitionKeyword>($partition)
    custom<OptionalIndices>($indices)
    attr-dict `:` type($patch_out) `->` type($bits)
  }];
}

def Fabric_MeasureBasisOp : Fabric_Op<"measure_basis", [
    AllTypesMatch<["patch", "patch_out"]>,
    DeclareOpInterfaceMethods<Fabric_MeasurementBundleOpInterface>
  ]> {
  let summary = "Atomically measure a carrier partition in a typed basis";
  let description = [{
    P2 atomic ancilla-measurement step. The `basis` attribute is retained
    through P3 and target lowering. A target without a native X-basis read may
    realize `#fabric.prep<x>` as a basis change followed by Z readout, without
    changing the authored P2 extraction depth.
  }];
  let arguments = (ins
    Fabric_PatchLikeType:$patch,
    Fabric_PartitionAttr:$partition,
    OptionalAttr<DenseI64ArrayAttr>:$indices,
    Fabric_PrepAttr:$basis,
    OptionalAttr<StrAttr>:$record
  );
  let results = (outs
    Fabric_PatchLikeType:$patch_out,
    AnyRankedTensor:$bits
  );
  let assemblyFormat = [{
    $patch custom<PartitionKeyword>($partition)
    custom<OptionalIndices>($indices)
    `basis` $basis attr-dict `:` type($patch_out) `->` type($bits)
  }];
}

def Fabric_MppOp : Fabric_Op<"mpp", [AllTypesMatch<["patch", "patch_out"]>]> {
  let summary = "Measure one physical Pauli product over a patch carrier view";
  let description = [{
    Advanced P2 primitive for an explicitly selected physical Pauli-product
    measurement. `indices` and `paulis` are aligned and the operation returns
    one measurement bit. Logical MPP authoring remains a higher-level action
    lowered through device-owned MPP protocols.

    `paulis` may carry a single optional leading `-` encoding a negated
    product (e.g. `"-XX"` measures -XX): the measurement projectors are those
    of the negated operator, so the recorded bit is the complement of the
    unsigned product's outcome (Stim's inverted-target `MPP !X0*X1` form).
    A `+` sign is never written; the unsigned string is the +1 product.
  }];
  let arguments = (ins
    Fabric_PatchLikeType:$patch,
    Fabric_PartitionAttr:$partition,
    DenseI64ArrayAttr:$indices,
    StrAttr:$paulis,
    OptionalAttr<StrAttr>:$record
  );
  let results = (outs
    Fabric_PatchLikeType:$patch_out,
    AnyRankedTensor:$bits
  );
  let assemblyFormat = [{
    $patch custom<PartitionKeyword>($partition)
    `indices` $indices `paulis` $paulis attr-dict
    `:` type($patch_out) `->` type($bits)
  }];
  let hasVerifier = 1;
}

def Fabric_ReadSyndromeAncillasOp : Fabric_Op<"read_syndrome_ancillas", [
    AllTypesMatch<["patch", "patch_out"]>
  ]> {
  let summary = "read already-entangled syndrome ancillas";
  let description = [{
    Measures the patch's `sx` partition followed by its `sz` partition and
    returns the unchanged encoded owner plus a qualified syndrome record.
    This operation never prepares ancillas, applies extraction entangling
    gates, or infers a check schedule. Those operations must already be
    explicit in the realization. `qlx.extract_syndrome` expands the standard
    CSS ancilla circuit before emitting this read.
  }];
  let arguments = (ins
    Fabric_PatchType:$patch,
    OptionalAttr<StrAttr>:$record
  );
  let results = (outs
    Fabric_PatchType:$patch_out,
    Fabric_SyndromeType:$syndrome
  );
  let assemblyFormat = [{
    $patch attr-dict `:` type($patch_out) `->` type($syndrome)
  }];
  let hasVerifier = 1;
}

def Fabric_AssembleSyndromeOp : Fabric_Op<"assemble_syndrome", [
    AllTypesMatch<["patch", "patch_out"]>
  ]> {
  let summary = "assemble a typed syndrome from explicit ancilla measurements";
  let description = [{
    Associates already-authored `sx` and `sz` measurement bundles with the
    patch's code, encoding, and epoch and returns the canonical full syndrome
    order (`sx` followed by `sz`). This operation performs no measurement and
    inserts no timing boundary. It exists for extraction realizations whose
    basis measurements occur in different authored moments, such as the
    bivariate-bicycle depth-8 cycle.
  }];
  let arguments = (ins
    Fabric_PatchType:$patch,
    AnyRankedTensor:$sx_bits,
    AnyRankedTensor:$sz_bits,
    StrAttr:$record
  );
  let results = (outs
    Fabric_PatchType:$patch_out,
    Fabric_SyndromeType:$syndrome
  );
  let assemblyFormat = [{
    $patch `from` $sx_bits `,` $sz_bits attr-dict
    `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Fabric_MeasureGaugesOp : Fabric_Op<"measure_gauges", [
    AllTypesMatch<["patch", "patch_out"]>
  ]> {
  let summary = "measure one ordered gauge basis without inventing a syndrome";
  let description = [{
    Measures the declared binary-symplectic operators and returns their raw
    outcomes. Gauge outcomes are not implicitly promoted to stabilizer
    syndrome. `phase` identifies an optional dynamic-code measurement phase;
    `record` is the stable relative record base.
  }];
  let arguments = (ins
    Fabric_PatchType:$patch,
    AnyAttr:$operators,
    OptionalAttr<AnyAttr>:$stabilizer_map,
    OptionalAttr<StrAttr>:$phase,
    StrAttr:$record
  );
  let results = (outs
    Fabric_PatchType:$patch_out,
    Fabric_GaugeRecordsType:$records
  );
  let assemblyFormat = [{
    $patch attr-dict `:` functional-type($patch, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Intra-patch two-qubit gates
//===----------------------------------------------------------------------===//

def Fabric_CXOp : Fabric_Op<"cx", [
    SameVariadicOperandSize,
    SameVariadicResultSize
  ]> {
  let summary = "CX between qubits in one or two patches";
  let description = [{
    Single-patch (intra) form takes one patch operand; `$ctrl` and
    `$targ` are partitions of that patch.

    Two-patch (cross) form takes two patch operands; `$ctrl` is a
    partition of `patches[0]` and `$targ` is a partition of
    `patches[1]`. Used for lattice-surgery seam-stabilizer extraction
    where a single seam ancilla in one patch needs to CX into data
    qubits living in a neighbouring patch. The two patches must be
    floorplan-adjacent (verified at construction time by the Python
    builder).

    For the two-patch form `pairs` is required (the schedule keywords
    `"hx"` / `"hz"` are intra-patch concepts).
  }];
  let arguments = (ins
    Variadic<Fabric_PatchLikeType>:$patches,
    Fabric_PartitionAttr:$ctrl,
    Fabric_PartitionAttr:$targ,
    OptionalAttr<StrAttr>:$schedule,
    OptionalAttr<StrAttr>:$pairs
  );
  let results = (outs Variadic<Fabric_PatchLikeType>:$results);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def Fabric_CZOp : Fabric_Op<"cz", [
    SameVariadicOperandSize,
    SameVariadicResultSize
  ]> {
  let summary = "CZ between qubits in one or two patches";
  let description = [{
    Single-patch (intra) form takes one patch operand; `$ctrl` and
    `$targ` are partitions of that patch.

    Two-patch (cross) form takes two patch operands; `$ctrl` is a
    partition of `patches[0]` and `$targ` is a partition of
    `patches[1]`.
  }];
  let arguments = (ins
    Variadic<Fabric_PatchLikeType>:$patches,
    Fabric_PartitionAttr:$ctrl,
    Fabric_PartitionAttr:$targ,
    OptionalAttr<StrAttr>:$schedule,
    OptionalAttr<StrAttr>:$pairs
  );
  let results = (outs Variadic<Fabric_PatchLikeType>:$results);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def Fabric_TransversalCXOp : Fabric_Op<"transversal_cx", []> {
  let summary = "Pairwise CX between two patches";
  let arguments = (ins
    Fabric_PatchType:$ctrl,
    Fabric_PatchType:$targ,
    OptionalAttr<DenseI64ArrayAttr>:$perm
  );
  let results = (outs
    Fabric_PatchType:$ctrl_out,
    Fabric_PatchType:$targ_out
  );
  let assemblyFormat = [{
    $ctrl `,` $targ attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Lattice Surgery
//===----------------------------------------------------------------------===//

def Fabric_MergeOp : Fabric_Op<"merge", []> {
  let summary = "Merge two patches via lattice surgery";
  let arguments = (ins
    Fabric_PatchType:$patch_a,
    Fabric_PatchType:$patch_b,
    Fabric_SlotType:$corridor,
    Fabric_MergeBasisAttr:$basis,
    Fabric_BoundaryAttr:$boundary,
    OptionalAttr<StrAttr>:$continuity_phase,
    OptionalAttr<UnitAttr>:$continuity_reset
  );
  let results = (outs Fabric_PatchType:$merged);
  let assemblyFormat = [{
    $patch_a `,` $patch_b `,` $corridor
    `basis` `=` custom<MergeBasisKeyword>($basis)
    `boundary` `=` custom<BoundaryKeyword>($boundary)
    attr-dict
    `:` functional-type(operands, $merged)
  }];
}

def Fabric_SplitOp : Fabric_Op<"split", []> {
  let summary = "Split a merged patch";
  let arguments = (ins Fabric_PatchType:$merged);
  let results = (outs
    Fabric_PatchType:$patch_a,
    Fabric_PatchType:$patch_b,
    I1:$outcome
  );
  let assemblyFormat = [{
    $merged attr-dict `:` functional-type($merged, results)
  }];
}

def Fabric_MultiMeasureOp : Fabric_Op<"multi_measure", [
    AttrSizedOperandSegments
  ]> {
  let summary = "Multi-patch Pauli measurement via lattice surgery";
  let arguments = (ins
    Variadic<Fabric_PatchType>:$patches,
    Variadic<Fabric_SlotType>:$corridors,
    StrAttr:$pauli_product
  );
  let results = (outs
    Variadic<Fabric_PatchType>:$patches_out,
    I1:$outcome
  );
  let hasCustomAssemblyFormat = 1;
}

def Fabric_MeasureProductOp : Fabric_Op<"measure_product", []> {
  let summary = "Measure a logical Pauli product over encoded logical refs";
  let description = [{
    First-class high-rate-code product measurement intent. Operands are
    unique encoded blocks. The term arrays have one entry per logical
    product term:

      * `patch_indices[i]` selects the operand patch for term i.
      * `logical_indices[i]` selects the encoded logical inside that block.
      * `pauli_product[i]` is the Pauli label for that term.

    `pauli_product` may carry a single optional leading `-` encoding a
    negated product (sign -1, e.g. `"-ZZ"` measures -Z⊗Z). The measurement
    projectors are those of the negated operator, so the recorded `outcome`
    bit is the complement of the unsigned product's outcome; backends
    realize this as an inverted measurement record (Stim `MPP !Z0*Z1`) or a
    classical flip. A `+` sign is never written. Term arrays index only the
    Pauli labels, not the sign character.

    The op returns updated patch values for every input patch plus the
    decoded product outcome as i1. Physical realization is target/protocol
    dependent; emitters that cannot expand this op must reject it explicitly.
  }];
  let arguments = (ins
    Variadic<Fabric_PatchType>:$patches,
    DenseI64ArrayAttr:$patch_indices,
    DenseI64ArrayAttr:$logical_indices,
    StrAttr:$pauli_product,
    OptionalAttr<StrAttr>:$record,
    OptionalAttr<AnyAttrOf<[FlatSymbolRefAttr, Fabric_SpecOnlyAttr]>>:$protocol
  );
  let results = (outs
    Variadic<Fabric_PatchType>:$patch_results,
    I1:$outcome
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $patches attr-dict `:` functional-type($patches, results)
  }];
}

def Fabric_RotateProductOp : Fabric_Op<"rotate_product", []> {
  let summary = "Apply a logical Pauli-product rotation intent";
  let description = [{
    First-class high-rate-code product rotation intent. Term encoding matches
    `fabric.measure_product`, including the optional leading `-` on
    `pauli_product`: a negated product rotates about -P, which is equivalent
    to rotating by `-angle` about the unsigned product. The `angle` is in
    radians. `synthesis` is a strict policy string emitted by the Python enum
    surface (`auto`, `native`, or `decompose`) until this is promoted to a
    dialect enum.
  }];
  let arguments = (ins
    Variadic<Fabric_PatchType>:$patches,
    DenseI64ArrayAttr:$patch_indices,
    DenseI64ArrayAttr:$logical_indices,
    StrAttr:$pauli_product,
    F64Attr:$angle,
    StrAttr:$synthesis,
    OptionalAttr<StrAttr>:$record,
    OptionalAttr<AnyAttrOf<[FlatSymbolRefAttr, Fabric_SpecOnlyAttr]>>:$protocol
  );
  let results = (outs Variadic<Fabric_PatchType>:$patch_results);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $patches attr-dict `:` functional-type($patches, $patch_results)
  }];
}

def Fabric_ResourceRotateProductOp : Fabric_Op<"resource_rotate_product", []> {
  let summary = "apply one exact Pauli-product rotation using a raw resource";
  let description = [{
    Consumes one linear raw resource and applies the fully specified product
    rotation to typed patch ports. Unlike legacy `fabric.inject`, this is not
    unresolved logical intent: the Pauli support, angle, resource kind, and
    successor patch boundary are explicit. Physical projection retains the
    resource dependency in `phys.resource_rotate_product`. Term encoding
    matches `fabric.measure_product`, including the optional leading `-` on
    `pauli_product` (rotation about -P == rotation by `-angle` about P).
  }];
  let arguments = (ins
    Fabric_ResourceStateType:$resource,
    Variadic<Fabric_PatchType>:$patches,
    DenseI64ArrayAttr:$patch_indices,
    DenseI64ArrayAttr:$logical_indices,
    StrAttr:$pauli_product,
    F64Attr:$angle
  );
  let results = (outs Variadic<Fabric_PatchType>:$patch_results);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $resource `on` $patches attr-dict `:` functional-type(operands, $patch_results)
  }];
}

//===----------------------------------------------------------------------===//
// Timing annotations
//===----------------------------------------------------------------------===//

def Fabric_TickOp : Fabric_Op<"tick", []> {
  let summary = "Round boundary marker (structure / timing annotation)";
  let description = [{
    Marks a logical time-step boundary for scheduling and circuit readability.
    Produces no SSA value.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Runtime ops (move data, return values)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pauli Frame Tracking
//===----------------------------------------------------------------------===//

def Fabric_FrameCreateOp : Fabric_Op<"frame_create", []> {
  let summary = "create empty symbolic frame state for one domain";
  let arguments = (ins StrAttr:$domain);
  let results = (outs Fabric_FrameType:$frame);
  let assemblyFormat = "$domain attr-dict `:` qualified(type($frame))";
}

def Fabric_FrameUpdateOp : Fabric_Op<"frame_update", [AllTypesMatch<["frame", "result"]>]> {
  let summary = "update symbolic frame state from one outcome";
  let arguments = (ins Fabric_FrameType:$frame, I1:$source, StrAttr:$update);
  let results = (outs Fabric_FrameType:$result);
  let assemblyFormat = "$frame `from` $source attr-dict `:` qualified(type($frame)) `->` qualified(type($result))";
}

def Fabric_FrameTransformOp : Fabric_Op<"frame_transform", [AllTypesMatch<["frame", "result"]>]> {
  let summary = "transform symbolic frame state through a known action";
  let arguments = (ins Fabric_FrameType:$frame, StrAttr:$transform);
  let results = (outs Fabric_FrameType:$result);
  let assemblyFormat = "$frame attr-dict `:` qualified(type($frame)) `->` qualified(type($result))";
}

def Fabric_FrameInitOp : Fabric_Op<"frame_init", []> {
  let summary = "Initialize Pauli frame";
  let arguments = (ins Fabric_PatchType:$patch);
  let results = (outs Fabric_FrameType:$frame);
  let assemblyFormat = "$patch attr-dict `:` functional-type($patch, $frame)";
}

def Fabric_FramePropagateOp : Fabric_Op<"frame_propagate", [AllTypesMatch<["frame", "result"]>]> {
  let summary = "Propagate frame through gate";
  let arguments = (ins
    Fabric_FrameType:$frame,
    StrAttr:$gate,
    Fabric_PartitionAttr:$partition
  );
  let results = (outs Fabric_FrameType:$result);
  let assemblyFormat = "$frame attr-dict `:` type($result)";
}

def Fabric_FrameResolveOp : Fabric_Op<"frame_resolve", []> {
  let summary = "Resolve measurement through frame";
  let arguments = (ins Fabric_FrameType:$frame, I1:$raw_bit);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $frame `,` $raw_bit attr-dict `:` functional-type(operands, $result)
  }];
}

//===----------------------------------------------------------------------===//
// Resource State and Injection
//===----------------------------------------------------------------------===//

def Fabric_ResourceRequestOp : Fabric_Op<"resource_request", []> {
  let summary = "request one typed resource from a placed logical stream";
  let arguments = (ins StrAttr:$kind, SymbolRefAttr:$stream);
  let results = (outs Event_HandleType:$event);
  let assemblyFormat = "$kind `from` $stream attr-dict `:` qualified(type($event))";
  let hasVerifier = 1;
}

def Fabric_ProduceResourceOp : Fabric_Op<"produce_resource", []> {
  let summary = "Produce a resource state from a factory region";
  let description = [{
    Produces one distilled / cultivated / natively-prepared resource state
    by invoking the production protocol bound to `region`. The `protocol`
    attribute is either a symbol reference to a `fabric.gadget` (the
    compiled production protocol body) or `#fabric.spec_only<"name">`
    when only the cost model is available.

    Example:
    ```mlir
    %r = fabric.produce_resource {region = @F0, protocol = @distill_15to1,
                                  resource = #fabric.resource<T>}
        : !fabric.resource<T>
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$region,
    OptionalAttr<Fabric_ResourceAttr>:$resource,
    OptionalAttr<FlatSymbolRefAttr>:$resource_kind,
    OptionalAttr<AnyAttrOf<[FlatSymbolRefAttr, Fabric_SpecOnlyAttr]>>:$protocol
  );
  let results = (outs Fabric_ResourceStateType:$resource_state);
  let assemblyFormat = "attr-dict `:` type($resource_state)";
  let hasVerifier = 1;
}

def Fabric_UnpackResourceOp : Fabric_Op<"unpack_resource", []> {
  let summary = "transfer an encoded resource payload into one or more patches";
  let description = [{
    Consumes one linear resource payload and one or more distinct live anchor
    patches. The first N results are the ownership successors of the N anchors;
    the final N results are explicitly typed payload patches, one per anchor.
    `payload_logical_blocks` and `payload_logical_ports` map each logical in the
    resource action to a payload result and a code-logical port. Repeated
    logical operands in one packed block are therefore port mappings, never
    duplicate patch owners or duplicate resource payloads.
    `payload_logical_block_ids` carries the exact selected QEC block identity
    for every mapping row; selected three-qubit magic-state handoffs require
    it and replay authenticates it against the detached QEC selection witness.
    `payload_roles` gives every payload position its resource-provider-defined
    semantic role. The generic dialect checks only ordered coverage and
    uniqueness; a registered resource provider may enforce a stronger schema
    and state-preparation proof.

    The payloads may use a different code/encoding from their placement
    anchors; their types are the resource-delivery contract and the physical
    projection must realize those encodings. The operation is a QEC
    ownership/placement boundary, not a logical gate and not an injection
    primitive. A concrete injection protocol must subsequently express its
    coupling, measurement, and feedback realization explicitly.
  }];
  let arguments = (ins
    Fabric_ResourceStateType:$resource,
    Variadic<Fabric_PatchType>:$anchors,
    OptionalAttr<QLX_BuiltinActionAttr>:$payload_action,
    OptionalAttr<ArrayAttr>:$payload_logical_block_ids,
    OptionalAttr<DenseI64ArrayAttr>:$payload_logical_blocks,
    OptionalAttr<DenseI64ArrayAttr>:$payload_logical_ports,
    OptionalAttr<ArrayAttr>:$payload_roles
  );
  let results = (outs Variadic<Fabric_PatchType>:$outputs);
  let assemblyFormat = [{
    $resource `like` `(` $anchors `)` attr-dict `:` functional-type(operands, $outputs)
  }];
  let hasVerifier = 1;
}

def Fabric_PackResourceOp : Fabric_Op<"pack_resource", []> {
  let summary = "transfer encoded patches into one linear resource payload";
  let description = [{
    Consumes one or more distinct encoded patches and packages their ownership
    as one typed resource payload. Payload order is semantic and is preserved
    by a matching `fabric.unpack_resource`. The resource kind states the
    producer's payload claim; verification evidence on the producer establishes
    that the patches actually have the promised joint state. `payload_roles`
    names every position's resource-kind-defined semantic role.
  }];
  let arguments = (ins
    Variadic<Fabric_PatchType>:$payloads,
    FlatSymbolRefAttr:$resource_kind,
    ArrayAttr:$payload_encodings,
    OptionalAttr<ArrayAttr>:$payload_roles
  );
  let results = (outs Fabric_ResourceStateType:$resource);
  let assemblyFormat = [{
    `(` $payloads `)` `as` $resource_kind attr-dict `:`
      functional-type($payloads, $resource)
  }];
  let hasVerifier = 1;
}

def Fabric_InjectOp : Fabric_Op<"inject", []> {
  let summary = "legacy unresolved resource-injection intent";
  let description = [{
    Compatibility syntax for alpha/imported modules that combined a linear
    resource boundary with an unresolved logical action. It is not legal in a
    canonical QLX P2 body. P1-to-P2 selection must instead link a concrete
    code-specific protocol whose body uses `fabric.unpack_resource` followed by
    explicit encoded operations, measurements, decoding, and feedback.

    P2-to-P3 rejects this operation. The `gate` and `protocol` attributes exist
    only so import tooling can diagnose or migrate the original intent; they
    never authorize carrierwise gate synthesis.
  }];
  let arguments = (ins
    Fabric_PatchType:$patch,
    Fabric_ResourceStateType:$resource,
    AnyAttrOf<[FlatSymbolRefAttr, Fabric_SpecOnlyAttr]>:$protocol,
    OptionalAttr<StrAttr>:$gate
  );
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = [{
    $patch `,` $resource attr-dict `:` functional-type(operands, $result)
  }];
  let hasVerifier = 1;
}

def Fabric_DiscardResourceOp : Fabric_Op<"discard_resource", []> {
  let summary = "Discard unused resource state";
  let arguments = (ins Fabric_ResourceStateType:$resource);
  let assemblyFormat = "$resource attr-dict `:` type($resource)";
}

def Fabric_TransportOp : Fabric_Op<"transport", []> {
  let summary = "Transport a resource state between region ports";
  let description = [{
    Moves a resource state from a source region's port to a destination
    region's port via an interconnect protocol. The result has the same
    resource kind as the input. The `protocol` attribute is either a
    symbol reference to a `fabric.gadget` implementing the transfer or a
    `#fabric.spec_only<"name">` attribute naming a `TransportSpec`.
    When `route` is present it names the selected `fabric.interconnect`; the
    endpoints and typed protocol must exactly match that interconnect.

    Example:
    ```mlir
    %r2 = fabric.transport %r from @F0 to @C0 {protocol = @ls_handoff}
        : !fabric.resource<T> -> !fabric.resource<T>
    ```
  }];
  let arguments = (ins
    Fabric_ResourceStateType:$resource,
    FlatSymbolRefAttr:$src_region,
    FlatSymbolRefAttr:$dst_region,
    AnyAttrOf<[FlatSymbolRefAttr, Fabric_SpecOnlyAttr]>:$protocol,
    OptionalAttr<SymbolRefAttr>:$route
  );
  let results = (outs Fabric_ResourceStateType:$result);
  let assemblyFormat = [{
    $resource `from` $src_region `to` $dst_region attr-dict
    `:` type($resource) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Inter-Region Transport
//===----------------------------------------------------------------------===//

def Fabric_MoveOp : Fabric_Op<"move", []> {
  let summary = "Intra-region patch movement";
  let arguments = (ins
    Fabric_PatchType:$patch,
    Fabric_SlotType:$from,
    Fabric_SlotType:$to,
    OptionalAttr<DenseI64ArrayAttr>:$via
  );
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = [{
    $patch `,` $from `,` $to attr-dict `:` functional-type(operands, $result)
  }];
}

def Fabric_SendOp : Fabric_Op<"send", []> {
  let summary = "Begin inter-region transfer";
  let arguments = (ins
    Fabric_PatchType:$patch,
    Fabric_SlotType:$port,
    FlatSymbolRefAttr:$dest_region
  );
  let assemblyFormat = "$patch `,` $port attr-dict `:` type(operands)";
}

def Fabric_RecvOp : Fabric_Op<"recv", []> {
  let summary = "Complete inter-region transfer";
  let arguments = (ins
    Fabric_SlotType:$port,
    FlatSymbolRefAttr:$src_region
  );
  let results = (outs Fabric_PatchType:$patch);
  let assemblyFormat = "$port attr-dict `:` functional-type($port, $patch)";
}

//===----------------------------------------------------------------------===//
// Timing, Synchronization, and Classical Control
//===----------------------------------------------------------------------===//

def Fabric_BarrierOp : Fabric_Op<"barrier", []> {
  let summary = "Synchronize patches (value-semantic)";
  let description = [{
    Synchronization point. Consumes and returns patches, maintaining
    the SSA dependency chain.
  }];
  let arguments = (ins Variadic<Fabric_PatchType>:$patches);
  let results = (outs Variadic<Fabric_PatchType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = [{
    ($patches^ `:` type($patches))? attr-dict (`->` type($results)^)?
  }];
}

def Fabric_IdleOp : Fabric_Op<"idle", [AllTypesMatch<["patch", "result"]>]> {
  let summary = "Explicit idle (stabilizer rounds with no logical op)";
  let arguments = (ins
    Fabric_PatchType:$patch,
    I64Attr:$rounds
  );
  let results = (outs Fabric_PatchType:$result);
  let assemblyFormat = "$patch attr-dict `:` type($result)";
}

def Fabric_XorOp : Fabric_Op<"xor", []> {
  let summary = "Classical parity (XOR)";
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Fabric_AllZeroOp : Fabric_Op<"all_zero", []> {
  let summary = "true when every bit in a measurement bundle is zero";
  let description = [{
    Reduces one statically shaped measurement record bundle to the runtime
    success predicate used by postselection and bounded retry.  It is a
    classical operation and does not alter the emitted quantum circuit.
  }];
  let arguments = (ins AnyRankedTensor:$bits);
  let results = (outs I1:$result);
  let assemblyFormat = "$bits attr-dict `:` type($bits) `->` type($result)";
  let hasVerifier = 1;
}

def Fabric_ParityOp : Fabric_Op<"parity", []> {
  let summary = "XOR one or more scalar measurement bundles";
  let description = [{
    Computes the classical parity of one or more measurement bundles. This
    operation does not emit a quantum instruction; it makes an explicitly
    authored parity available to selection logic.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$bits);
  let results = (outs I1:$result);
  let assemblyFormat = "$bits attr-dict `:` functional-type($bits, $result)";
  let hasVerifier = 1;
}

def Fabric_AllFalseOp : Fabric_Op<"all_false", []> {
  let summary = "true when no classical event is set";
  let arguments = (ins Variadic<I1>:$events);
  let results = (outs I1:$result);
  let assemblyFormat = "$events attr-dict `:` functional-type($events, $result)";
  let hasVerifier = 1;
}

#endif // FABRIC_OPS_TD
