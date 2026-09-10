/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// Attribute definitions for the Fabric dialect:
//
//   Enum attrs:
//     #fabric.partition<data|sx|sz|all>
//     #fabric.merge_basis<X|Z>
//     #fabric.boundary<north|east|south|west>
//     #fabric.prep<z|x>
//     #fabric.layout<checkerboard|linear|direct|custom>
//     #fabric.role<compute|memory|factory|scratch>
//     #fabric.route<patch_deform|automorphism|swap>
//
//   Composite attrs:
//     #fabric.floorplan<layout, [params]>
//     #fabric.flow<{x = "z", z = "x"}>       -- stabilizer flow
//     #fabric.spec_only<"name">              -- protocol identified by spec only,
//                                               no gadget body (estimation-only)
//
//===----------------------------------------------------------------------===//

#ifndef FABRIC_ATTRS_TD
#define FABRIC_ATTRS_TD

include "qlx/Dialect/Fabric/IR/FabricDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Partition enum: data, sx, sz, all
//===----------------------------------------------------------------------===//

def Fabric_Partition_data : I32EnumAttrCase<"data", 0>;
def Fabric_Partition_sx   : I32EnumAttrCase<"sx",   1>;
def Fabric_Partition_sz   : I32EnumAttrCase<"sz",   2>;
def Fabric_Partition_all  : I32EnumAttrCase<"all",  3>;

def Fabric_PartitionEnum : I32EnumAttr<"Partition",
    "Qubit partition within a patch",
    [Fabric_Partition_data, Fabric_Partition_sx,
     Fabric_Partition_sz, Fabric_Partition_all]> {
  let cppNamespace = "qlx::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_PartitionAttr : EnumAttr<Fabric_Dialect, Fabric_PartitionEnum,
                                    "partition"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// MergeBasis enum: X, Z
//===----------------------------------------------------------------------===//

def Fabric_MergeBasis_X : I32EnumAttrCase<"X", 0>;
def Fabric_MergeBasis_Z : I32EnumAttrCase<"Z", 1>;

def Fabric_MergeBasisEnum : I32EnumAttr<"MergeBasis",
    "Lattice surgery merge basis",
    [Fabric_MergeBasis_X, Fabric_MergeBasis_Z]> {
  let cppNamespace = "qlx::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_MergeBasisAttr : EnumAttr<Fabric_Dialect, Fabric_MergeBasisEnum,
                                     "merge_basis"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Boundary enum: north, east, south, west
//===----------------------------------------------------------------------===//

def Fabric_Boundary_north : I32EnumAttrCase<"north", 0>;
def Fabric_Boundary_east  : I32EnumAttrCase<"east",  1>;
def Fabric_Boundary_south : I32EnumAttrCase<"south", 2>;
def Fabric_Boundary_west  : I32EnumAttrCase<"west",  3>;

def Fabric_BoundaryEnum : I32EnumAttr<"Boundary",
    "Patch boundary direction",
    [Fabric_Boundary_north, Fabric_Boundary_east,
     Fabric_Boundary_south, Fabric_Boundary_west]> {
  let cppNamespace = "qlx::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_BoundaryAttr : EnumAttr<Fabric_Dialect, Fabric_BoundaryEnum,
                                   "boundary"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Prep enum: z, x
//===----------------------------------------------------------------------===//

def Fabric_Prep_z : I32EnumAttrCase<"z", 0>;
def Fabric_Prep_x : I32EnumAttrCase<"x", 1>;

def Fabric_PrepEnum : I32EnumAttr<"Prep",
    "Initialization basis for state preparation",
    [Fabric_Prep_z, Fabric_Prep_x]> {
  let cppNamespace = "qlx::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_PrepAttr : EnumAttr<Fabric_Dialect, Fabric_PrepEnum,
                               "prep"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Layout enum: checkerboard, linear, direct, custom
//===----------------------------------------------------------------------===//

def Fabric_Layout_checkerboard : I32EnumAttrCase<"checkerboard", 0>;
def Fabric_Layout_linear       : I32EnumAttrCase<"linear",       1>;
def Fabric_Layout_direct       : I32EnumAttrCase<"direct",       2>;
def Fabric_Layout_custom       : I32EnumAttrCase<"custom",       3>;

def Fabric_LayoutEnum : I32EnumAttr<"Layout",
    "Floorplan layout generator kind",
    [Fabric_Layout_checkerboard, Fabric_Layout_linear,
     Fabric_Layout_direct, Fabric_Layout_custom]> {
  let cppNamespace = "qlx::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_LayoutAttr : EnumAttr<Fabric_Dialect, Fabric_LayoutEnum,
                                 "layout"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Role enum: compute, memory, factory, scratch
//===----------------------------------------------------------------------===//

def Fabric_Role_compute : I32EnumAttrCase<"compute", 0>;
def Fabric_Role_memory  : I32EnumAttrCase<"memory",  1>;
def Fabric_Role_factory : I32EnumAttrCase<"factory", 2>;
def Fabric_Role_scratch : I32EnumAttrCase<"scratch", 3>;

def Fabric_RoleEnum : I32EnumAttr<"Role",
    "Region role in device",
    [Fabric_Role_compute, Fabric_Role_memory, Fabric_Role_factory,
     Fabric_Role_scratch]> {
  let cppNamespace = "qlx::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_RoleAttr : EnumAttr<Fabric_Dialect, Fabric_RoleEnum,
                               "role"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Route enum: patch_deform, automorphism, swap
//===----------------------------------------------------------------------===//

def Fabric_Route_patch_deform  : I32EnumAttrCase<"patch_deform",  0>;
def Fabric_Route_automorphism  : I32EnumAttrCase<"automorphism",  1>;
def Fabric_Route_swap          : I32EnumAttrCase<"swap",          2>;

def Fabric_RouteEnum : I32EnumAttr<"Route",
    "Routing strategy for patch movement",
    [Fabric_Route_patch_deform, Fabric_Route_automorphism,
     Fabric_Route_swap]> {
  let cppNamespace = "qlx::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_RouteAttr : EnumAttr<Fabric_Dialect, Fabric_RouteEnum,
                                "route"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ResourceType enum: T, CCZ, CS
//===----------------------------------------------------------------------===//

def Fabric_ResourceType_T   : I32EnumAttrCase<"T",   0, "T">;
def Fabric_ResourceType_CCZ : I32EnumAttrCase<"CCZ", 1, "CCZ">;
def Fabric_ResourceType_CS  : I32EnumAttrCase<"CS",  2, "CS">;

def Fabric_ResourceTypeEnum : I32EnumAttr<"ResourceType",
    "Magic state resource type",
    [Fabric_ResourceType_T, Fabric_ResourceType_CCZ,
     Fabric_ResourceType_CS]> {
  let cppNamespace = "qlx::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_ResourceAttr : EnumAttr<Fabric_Dialect, Fabric_ResourceTypeEnum,
                                   "resource"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Floorplan composite attribute: #fabric.floorplan<checkerboard, [11, 11]>
//===----------------------------------------------------------------------===//

def Fabric_FloorplanAttr : AttrDef<Fabric_Dialect, "Floorplan"> {
  let mnemonic = "floorplan";
  let summary = "Compact floorplan specification";
  let description = [{
    Specifies a floorplan layout generator and its integer parameters.

    Example:
    ```mlir
    #fabric.floorplan<checkerboard, [11, 11]>
    #fabric.floorplan<direct, [6]>
    ```
  }];

  let parameters = (ins
    "qlx::fabric::Layout":$layout,
    ArrayRefParameter<"int64_t">:$params
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Flow attribute: #fabric.flow<{x = "z", z = "x"}>
//===----------------------------------------------------------------------===//

def Fabric_FlowAttr : AttrDef<Fabric_Dialect, "Flow"> {
  let mnemonic = "flow";
  let summary = "Stabilizer flow declaration";
  let description = [{
    Declares how a gadget transforms stabilizer bases. The `x` field says
    what X-stabilizers become at the output; the `z` field says what
    Z-stabilizers become.

    Common flows:
    - Identity:  `#fabric.flow<{x = "x", z = "z"}>`
    - Hadamard:  `#fabric.flow<{x = "z", z = "x"}>`
    - S-gate:    `#fabric.flow<{x = "y", z = "z"}>`

    Used on `fabric.gadget` as an optional verification assertion for its
    declared Clifford action. Syndrome propagation derives basis changes from
    patch dataflow and does not consume this attribute.

    Example:
    ```mlir
    fabric.gadget @logical_h(...)
        -> (...)
        flow #fabric.flow<{x = "z", z = "x"}>
    { ... }
    ```
  }];

  let parameters = (ins
    "mlir::StringAttr":$x_to,
    "mlir::StringAttr":$z_to
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// SpecOnly attribute: #fabric.spec_only<"distill-15to1-T">
//===----------------------------------------------------------------------===//

def Fabric_SpecOnlyAttr : AttrDef<Fabric_Dialect, "SpecOnly"> {
  let mnemonic = "spec_only";
  let summary = "Protocol reference by spec name only, no gadget body";
  let description = [{
    A reference to a protocol identified solely by its `ProductionSpec` /
    `TransportSpec` / `InjectionSpec` name. Used in `protocol` attribute
    slots (on `fabric.region`, `fabric.interconnect`, `fabric.produce_resource`,
    `fabric.transport`, `fabric.inject`) when the architecture declares a
    cost model but has no compiled gadget body. Downstream consumers
    (resource estimation, scheduling) read the name and look up the
    associated spec in a side table. Passes that require a circuit body
    must error on `SpecOnly`.

    Example:
    ```mlir
    fabric.region @F0 {
      code = @surface_15,
      role = #fabric.role<factory>,
      protocol = #fabric.spec_only<"distill-15to1-T">
    }
    ```
  }];

  let parameters = (ins StringRefParameter<>:$name);
  let assemblyFormat = "`<` $name `>`";
}

#endif // FABRIC_ATTRS_TD
