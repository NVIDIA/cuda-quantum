//===- FabricTypes.td - Fabric type definitions -----------*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// Type definitions for the Fabric dialect.
//
//   !fabric.patch<@code, @encoding, @epoch> - encoded QEC block owner
//   !fabric.syndrome<@code, @encoding, @epoch> - epoch-qualified syndrome
//   builtin i1                - scalar measurement, branch, and parity bit
//   !fabric.frame             - Pauli frame tracker
//   !fabric.resource<T|CCZ|CS> - distilled resource state (T magic, CCZ, CS)
//   !fabric.slot              - floorplan slot reference
//   !fabric.machine            - QEC machine reference
//
//===----------------------------------------------------------------------===//

#ifndef FABRIC_TYPES_TD
#define FABRIC_TYPES_TD

include "qlx/Dialect/Fabric/IR/FabricDialect.td"
include "qlx/Dialect/Fabric/IR/FabricAttrs.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Parameterized types
//===----------------------------------------------------------------------===//

def Fabric_PatchType : TypeDef<Fabric_Dialect, "Patch"> {
  let mnemonic = "patch";
  let summary = "QEC block owner parameterized by code and optional encoding";
  let description = [{
    A handle to a QEC code block (patch) containing named qubit partitions:
    `data`, `sx` (X-syndrome ancillas), and `sz` (Z-syndrome ancillas).
    Parameterized by a code symbol and, for canonical QLX artifacts, an
    encoding symbol. The code-only form remains accepted for alpha IR and
    partially migrated external producers.

    Uses value semantics: operations consume the input patch and produce a
    new SSA value.

    Example:
    ```mlir
    %p = fabric.alloc {code = @surface_17, region = @Cs0}
        : !fabric.patch<@surface_17>
    ```
  }];

  let parameters = (ins
    "mlir::FlatSymbolRefAttr":$code_type,
    OptionalParameter<"mlir::FlatSymbolRefAttr">:$encoding,
    OptionalParameter<"mlir::FlatSymbolRefAttr">:$epoch
  );
  let assemblyFormat = [{ `<` $code_type (`,` $encoding^ (`,` $epoch^)?)? `>` }];
  let extraClassDeclaration = [{
    static PatchType get(mlir::MLIRContext *context,
                         mlir::FlatSymbolRefAttr codeType) {
      return get(context, codeType, mlir::FlatSymbolRefAttr{},
                 mlir::FlatSymbolRefAttr{});
    }
    static PatchType get(mlir::MLIRContext *context,
                         mlir::FlatSymbolRefAttr codeType,
                         mlir::FlatSymbolRefAttr encoding) {
      return get(context, codeType, encoding, mlir::FlatSymbolRefAttr{});
    }
  }];
}

def Fabric_PatchFrameType : TypeDef<Fabric_Dialect, "PatchFrame"> {
  let mnemonic = "patch_frame";
  let summary = "linear carrier frame during one code/encoding transformation";
  let description = [{
    A temporary, linearly owned carrier frame used inside a code-changing
    gadget. The referenced `fabric.patch_transform` defines the source and
    destination patch boundaries, persistent reservation, active-support
    embeddings, and carrier roles. Physical operations act on this frame; only
    `fabric.transform_end` may publish the destination encoded patch.
  }];
  let parameters = (ins "mlir::FlatSymbolRefAttr":$transform);
  let assemblyFormat = "`<` $transform `>`";
}

def Fabric_PatchLikeType : AnyTypeOf<[
  Fabric_PatchType,
  Fabric_PatchFrameType
]>;

def Fabric_SyndromeType : TypeDef<Fabric_Dialect, "Syndrome"> {
  let mnemonic = "syndrome";
  let summary = "syndrome measurement qualified by code and encoding";
  let description = [{
    Opaque value representing the full syndrome measurement from one
    round of stabilizer extraction (both X-type and Z-type checks).
    Parameterized by the code type, making syndromes self-describing.

    Produced by `fabric.read_syndrome_ancillas` or
    `fabric.assemble_syndrome` and available to protocol success predicates.

    The code type parameter enables composition passes to verify that
    syndromes from different gadgets are compatible without inspecting
    gadget internals.

    Example:
    ```mlir
    %p_out, %syn = fabric.read_syndrome_ancillas %p
        : !fabric.patch<@steane> -> !fabric.syndrome<@steane>
    ```
  }];

  let parameters = (ins
    "mlir::FlatSymbolRefAttr":$code_type,
    OptionalParameter<"mlir::FlatSymbolRefAttr">:$encoding,
    OptionalParameter<"mlir::FlatSymbolRefAttr">:$epoch
  );
  let assemblyFormat = [{ `<` $code_type (`,` $encoding^ (`,` $epoch^)?)? `>` }];
  let extraClassDeclaration = [{
    static SyndromeType get(mlir::MLIRContext *context,
                            mlir::FlatSymbolRefAttr codeType) {
      return get(context, codeType, mlir::FlatSymbolRefAttr{},
                 mlir::FlatSymbolRefAttr{});
    }
    static SyndromeType get(mlir::MLIRContext *context,
                            mlir::FlatSymbolRefAttr codeType,
                            mlir::FlatSymbolRefAttr encoding) {
      return get(context, codeType, encoding, mlir::FlatSymbolRefAttr{});
    }
  }];
}

def Fabric_GaugeRecordsType : TypeDef<Fabric_Dialect, "GaugeRecords"> {
  let mnemonic = "gauge_records";
  let summary = "epoch-qualified outcomes from one declared gauge basis";
  let description = [{
    A copyable classical record bundle produced by `fabric.measure_gauges`.
    The producing operation carries the ordered binary-symplectic generator
    matrix and stable record base.  Code, encoding, and epoch parameters keep
    dynamic-code records from being combined across incompatible phases.
  }];
  let parameters = (ins
    "mlir::FlatSymbolRefAttr":$code_type,
    "mlir::FlatSymbolRefAttr":$encoding,
    "mlir::FlatSymbolRefAttr":$epoch
  );
  let assemblyFormat = [{ `<` $code_type `,` $encoding `,` $epoch `>` }];
}

def Fabric_PatchBundleType : TypeDef<Fabric_Dialect, "PatchBundle"> {
  let mnemonic = "patch_bundle";
  let summary = "linear folded child-patch aggregate for an encoding hierarchy";
  let description = [{
    Owns one complete ordered child-slot group after `fabric.encoding_unpack`.
    The parent patch is consumed while this value is live.  The aggregate is
    folded: its IR size is independent of the hierarchy multiplicity.
  }];
  let parameters = (ins
    "mlir::FlatSymbolRefAttr":$hierarchy,
    "mlir::FlatSymbolRefAttr":$slot_group
  );
  let assemblyFormat = "`<` $hierarchy `,` $slot_group `>`";
}

def Fabric_RecordBundleType : TypeDef<Fabric_Dialect, "RecordBundle"> {
  let mnemonic = "record_bundle";
  let summary = "folded hierarchy record aggregate with one stable schema";
  let parameters = (ins "mlir::FlatSymbolRefAttr":$schema);
  let assemblyFormat = "`<` $schema `>`";
}

def Fabric_ResourceStateType : TypeDef<Fabric_Dialect, "ResourceState"> {
  let mnemonic = "resource";
  let summary = "distilled resource state for non-Clifford gates";
  let description = [{
    A distilled resource state parameterized by resource kind (`T`, `CCZ`,
    or `CS`). Produced by factory production protocols and consumed by
    injection protocols. The umbrella term "resource state" subsumes
    informal usage of "magic state" (T) as well as CCZ and CS.

    Example:
    ```mlir
    %r = fabric.produce_resource {region = @F0, protocol = @distill_15to1}
        : !fabric.resource<T>
    ```
  }];

  let parameters = (ins "mlir::Attribute":$kind);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    static ResourceStateType get(mlir::MLIRContext *context,
                                 qlx::fabric::ResourceType resourceType) {
      return get(context,
                 qlx::fabric::ResourceTypeAttr::get(context, resourceType));
    }
    bool hasLegacyResourceType() const {
      return llvm::isa<qlx::fabric::ResourceTypeAttr>(getKind());
    }
    qlx::fabric::ResourceType getResourceType() const {
      return llvm::cast<qlx::fabric::ResourceTypeAttr>(getKind()).getValue();
    }
  }];
}

def Fabric_EventType : TypeDef<Fabric_Dialect, "Event"> {
  let mnemonic = "event";
  let summary = "typed asynchronous Fabric resource result";
  let parameters = (ins
    "mlir::Type":$payload,
    StringRefParameter<"payload ownership">:$ownership
  );
  let assemblyFormat = "`<` $payload `,` $ownership `>`";
}

//===----------------------------------------------------------------------===//
// Simple (unparameterized) types
//===----------------------------------------------------------------------===//

class Fabric_SimpleType<string name, string typeMnemonic>
    : TypeDef<Fabric_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Fabric_BitType : Fabric_SimpleType<"Bit", "bit"> {
  let summary = "deprecated legacy measurement bit; normalize to builtin i1";
  let description = [{
    Accepted only for legacy import compatibility. Canonical Fabric uses
    builtin `i1` for scalar measurement, branch, parity, and success values.
    Multiple bits from bulk measurement are returned as `tensor<Nxi1>`.
  }];
}

def Fabric_FrameType : TypeDef<Fabric_Dialect, "Frame"> {
  let mnemonic = "frame";
  let summary = "Pauli frame tracker";
  let description = [{
    Tracks accumulated Pauli corrections classically for a patch.
    Uses value semantics: consumed and updated through frame operations.
  }];
  let parameters = (ins
    OptionalParameter<"mlir::FlatSymbolRefAttr">:$domain
  );
  let assemblyFormat = [{ (`<` $domain^ `>`)? }];
  let extraClassDeclaration = [{
    static FrameType get(mlir::MLIRContext *context) {
      return get(context, mlir::FlatSymbolRefAttr{});
    }
  }];
}

def Fabric_SlotType : Fabric_SimpleType<"Slot", "slot"> {
  let summary = "floorplan slot reference";
  let description = [{
    Reference to a floorplan slot (data, corridor, port, or factory).
    Used in placement annotations, lattice surgery, and transport.
  }];
}

def Fabric_MachineType : Fabric_SimpleType<"Machine", "machine"> {
  let summary = "QEC machine reference";
  let description = [{
    Reference to a `fabric.machine` QEC virtual-machine declaration. Passed
    implicitly to QEC-machine-bound entry gadgets.
  }];
}

#endif // FABRIC_TYPES_TD
