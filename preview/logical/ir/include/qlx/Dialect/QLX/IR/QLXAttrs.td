//===- QLXAttrs.td - QLX attribute definitions -------------*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// This file defines the attributes for the QLX dialect, including enum
// attributes for region roles, interconnect protocols, Pauli bases, and
// resource state types, as well as a string attribute for the target profile.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_ATTRS_TD
#define QLX_ATTRS_TD

include "qlx/Dialect/QLX/IR/QLXDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// QLX Enum Attributes
//===----------------------------------------------------------------------===//

// Portable built-in logical actions are semantic values, not module-level
// declarations.  Custom and composite actions still use qlx.action symbols.
def QLX_BuiltinAction_H             : I32EnumAttrCase<"h", 0>;
def QLX_BuiltinAction_S             : I32EnumAttrCase<"s", 1>;
def QLX_BuiltinAction_Sdg           : I32EnumAttrCase<"sdg", 2>;
def QLX_BuiltinAction_X             : I32EnumAttrCase<"x", 3>;
def QLX_BuiltinAction_Y             : I32EnumAttrCase<"y", 4>;
def QLX_BuiltinAction_Z             : I32EnumAttrCase<"z", 5>;
def QLX_BuiltinAction_T             : I32EnumAttrCase<"t", 6>;
def QLX_BuiltinAction_Tdg           : I32EnumAttrCase<"tdg", 7>;
def QLX_BuiltinAction_CX            : I32EnumAttrCase<"cx", 8>;
def QLX_BuiltinAction_CZ            : I32EnumAttrCase<"cz", 9>;
def QLX_BuiltinAction_CCZ           : I32EnumAttrCase<"ccz", 10>;
def QLX_BuiltinAction_Idle          : I32EnumAttrCase<"idle", 11>;
def QLX_BuiltinAction_PauliRotation : I32EnumAttrCase<"pauli_rotation", 12>;

def QLX_BuiltinActionEnum : I32EnumAttr<"BuiltinAction",
    "Canonical built-in unplaced logical action",
    [QLX_BuiltinAction_H, QLX_BuiltinAction_S, QLX_BuiltinAction_Sdg,
     QLX_BuiltinAction_X, QLX_BuiltinAction_Y, QLX_BuiltinAction_Z,
     QLX_BuiltinAction_T, QLX_BuiltinAction_Tdg, QLX_BuiltinAction_CX,
     QLX_BuiltinAction_CZ, QLX_BuiltinAction_CCZ, QLX_BuiltinAction_Idle,
     QLX_BuiltinAction_PauliRotation]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_BuiltinActionAttr : EnumAttr<QLX_Dialect, QLX_BuiltinActionEnum,
                                      "action"> {
  let assemblyFormat = "`<` $value `>`";
}

// General instruments remain extensible symbols.  This attribute covers only
// built-ins whose complete semantics are owned by the QLX dialect.
def QLX_BuiltinInstrument_MPP : I32EnumAttrCase<"mpp", 0>;

def QLX_BuiltinInstrumentEnum : I32EnumAttr<"BuiltinInstrument",
    "Canonical built-in unplaced logical instrument",
    [QLX_BuiltinInstrument_MPP]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_BuiltinInstrumentAttr
    : EnumAttr<QLX_Dialect, QLX_BuiltinInstrumentEnum, "instrument"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Region role: compute, memory, factory
//===----------------------------------------------------------------------===//

def QLX_Role_Compute : I32EnumAttrCase<"compute", 0>;
def QLX_Role_Memory  : I32EnumAttrCase<"memory",  1>;
def QLX_Role_Factory : I32EnumAttrCase<"factory", 2>;

def QLX_RoleEnum : I32EnumAttr<"Role",
    "Region role in the QLX architecture",
    [QLX_Role_Compute, QLX_Role_Memory, QLX_Role_Factory]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_RoleAttr : EnumAttr<QLX_Dialect, QLX_RoleEnum, "role"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Interconnect protocol: codeswitch, teleport, joint_measurement, inject
//===----------------------------------------------------------------------===//

def QLX_Protocol_Codeswitch        : I32EnumAttrCase<"codeswitch",        0>;
def QLX_Protocol_Teleport          : I32EnumAttrCase<"teleport",          1>;
def QLX_Protocol_JointMeasurement  : I32EnumAttrCase<"joint_measurement", 2>;
def QLX_Protocol_Inject            : I32EnumAttrCase<"inject",            3>;

def QLX_ProtocolEnum : I32EnumAttr<"Protocol",
    "Interconnect protocol for inter-region communication",
    [QLX_Protocol_Codeswitch, QLX_Protocol_Teleport,
     QLX_Protocol_JointMeasurement, QLX_Protocol_Inject]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_ProtocolAttr : EnumAttr<QLX_Dialect, QLX_ProtocolEnum, "protocol"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Pauli basis: X, Y, Z
//===----------------------------------------------------------------------===//

def QLX_Pauli_X : I32EnumAttrCase<"X", 0>;
def QLX_Pauli_Y : I32EnumAttrCase<"Y", 1>;
def QLX_Pauli_Z : I32EnumAttrCase<"Z", 2>;

def QLX_PauliEnum : I32EnumAttr<"Pauli",
    "Pauli operator basis for measurements",
    [QLX_Pauli_X, QLX_Pauli_Y, QLX_Pauli_Z]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_PauliAttr : EnumAttr<QLX_Dialect, QLX_PauliEnum, "pauli"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Resource state type: T, CCZ, CS
//===----------------------------------------------------------------------===//

def QLX_ResourceType_T   : I32EnumAttrCase<"T",   0>;
def QLX_ResourceType_CCZ : I32EnumAttrCase<"CCZ", 1>;
def QLX_ResourceType_CS  : I32EnumAttrCase<"CS",  2>;

def QLX_ResourceTypeEnum : I32EnumAttr<"ResourceType",
    "Type of resource state (magic state)",
    [QLX_ResourceType_T, QLX_ResourceType_CCZ, QLX_ResourceType_CS]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_ResourceTypeAttr : EnumAttr<QLX_Dialect, QLX_ResourceTypeEnum,
                                     "resource_type"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Gate implementation: native, resource
//===----------------------------------------------------------------------===//

def QLX_GateImpl_Native   : I32EnumAttrCase<"native",   0>;
def QLX_GateImpl_Resource : I32EnumAttrCase<"resource",  1>;

def QLX_GateImplEnum : I32EnumAttr<"GateImpl",
    "How a logical gate is realized on a code type",
    [QLX_GateImpl_Native, QLX_GateImpl_Resource]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_GateImplAttr : EnumAttr<QLX_Dialect, QLX_GateImplEnum, "gate_impl"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Routing strategy: patch_deform, automorphism, swap
//===----------------------------------------------------------------------===//

def QLX_RouteStrategy_PatchDeform  : I32EnumAttrCase<"patch_deform",  0>;
def QLX_RouteStrategy_Automorphism : I32EnumAttrCase<"automorphism",  1>;
def QLX_RouteStrategy_Swap         : I32EnumAttrCase<"swap",          2>;

def QLX_RouteStrategyEnum : I32EnumAttr<"RouteStrategy",
    "Routing strategy for moving logical qubits within a region",
    [QLX_RouteStrategy_PatchDeform, QLX_RouteStrategy_Automorphism,
     QLX_RouteStrategy_Swap]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_RouteStrategyAttr : EnumAttr<QLX_Dialect, QLX_RouteStrategyEnum,
                                      "route_strategy"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Floorplan slot kind: data, corridor, port, factory
//===----------------------------------------------------------------------===//

def QLX_SlotKind_Data     : I32EnumAttrCase<"data",     0>;
def QLX_SlotKind_Corridor : I32EnumAttrCase<"corridor", 1>;
def QLX_SlotKind_Port     : I32EnumAttrCase<"port",     2>;
def QLX_SlotKind_Factory  : I32EnumAttrCase<"factory",  3>;

def QLX_SlotKindEnum : I32EnumAttr<"SlotKind",
    "Type of a floorplan slot",
    [QLX_SlotKind_Data, QLX_SlotKind_Corridor,
     QLX_SlotKind_Port, QLX_SlotKind_Factory]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_SlotKindAttr : EnumAttr<QLX_Dialect, QLX_SlotKindEnum, "slot_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// QLX Composite Attributes
//===----------------------------------------------------------------------===//

def QLX_CliffordActionAttr : AttrDef<QLX_Dialect, "CliffordAction"> {
  let mnemonic = "clifford_action";
  let summary = "complete signed binary-symplectic logical action";
  let description = [{
    Stores a flattened row-major 2k-by-2k binary symplectic matrix, one
    binary sign phase per generator image, and the ordered logical port
    signature.  Matrix rows are the images of X_0..X_(k-1), Z_0..Z_(k-1);
    columns are x_0..x_(k-1), z_0..z_(k-1).
  }];
  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$matrix,
    "::mlir::DenseI64ArrayAttr":$phases,
    "::mlir::ArrayAttr":$ports
  );
  let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<` `matrix` `=` $matrix `,` `phases` `=` $phases `,`
        `ports` `=` $ports `>`
  }];
}

//===----------------------------------------------------------------------===//
// CodeDesc: JSON-encoded code descriptor (compiler-visible capabilities)
//===----------------------------------------------------------------------===//

def QLX_CodeDescAttr : AttrDef<QLX_Dialect, "CodeDesc"> {
  let mnemonic = "code_desc";
  let summary = "JSON-encoded code-type descriptor";
  let description = [{
    Declarative code-type descriptor encoding the compiler-visible
    capabilities of a QEC code: physical qubit count, distance, per-gate
    costs, error model, and routing parameters. Encoded as a JSON string
    that is parsed by C++ passes using nlohmann_json.
  }];

  let parameters = (ins StringRefParameter<"JSON-encoded descriptor">:$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Floorplan: JSON-encoded typed spatial graph
//===----------------------------------------------------------------------===//

def QLX_FloorplanAttr : AttrDef<QLX_Dialect, "Floorplan"> {
  let mnemonic = "floorplan";
  let summary = "JSON-encoded floorplan (typed spatial graph)";
  let description = [{
    Describes how code blocks are arranged within a region: which slots
    hold data patches, which are corridors (for lattice surgery boundaries
    and routing), and how they connect. Encoded as a JSON string.
  }];

  let parameters = (ins StringRefParameter<"JSON-encoded floorplan">:$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// QLX String Attributes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Profile attribute: wraps the target triple string (e.g. "qlx-cmf-csw.tp")
//===----------------------------------------------------------------------===//

def QLX_ProfileAttr : AttrDef<QLX_Dialect, "Profile"> {
  let mnemonic = "profile";
  let summary = "QLX target profile triple string";
  let description = [{
    Encodes the target profile triple for a QLX program. The profile string
    specifies the architectural capabilities available, following the format:

      `qlx-<regions>-<interconnects>`

    For example:
    - `"qlx-cmf-csw.tp"` - compute + memory + factory regions, codeswitch
      and teleport interconnects
    - `"qlx-c"` - compute-only (no inter-region operations)
    - `"qlx-cm-tp"` - compute + memory, teleport only

    The profile constrains which QLX operations are legal in a program.
  }];

  let parameters = (ins StringRefParameter<"the target triple string">:$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // QLX_ATTRS_TD
