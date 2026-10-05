/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx-c/Dialect/Fabric.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

#include "qlx/Dialect/Fabric/IR/FabricAttrs.h"
#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricTypes.h"

#include "mlir/IR/BuiltinAttributes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Fabric, fabric,
                                      qlx::fabric::FabricDialect)

//===----------------------------------------------------------------------===//
// Parameterized types
//===----------------------------------------------------------------------===//

bool fabricTypeIsAPatch(MlirType type) {
  return llvm::isa<qlx::fabric::PatchType>(unwrap(type));
}

MlirType fabricPatchTypeGet(MlirContext ctx, MlirStringRef codeSymbol) {
  auto *mlirCtx = unwrap(ctx);
  auto sym = mlir::FlatSymbolRefAttr::get(mlirCtx, unwrap(codeSymbol));
  return wrap(qlx::fabric::PatchType::get(mlirCtx, sym));
}

MlirTypeID fabricPatchTypeGetTypeID(void) {
  return wrap(qlx::fabric::PatchType::getTypeID());
}

bool fabricTypeIsASyndrome(MlirType type) {
  return llvm::isa<qlx::fabric::SyndromeType>(unwrap(type));
}

MlirType fabricSyndromeTypeGet(MlirContext ctx, MlirStringRef codeSymbol) {
  auto *mlirCtx = unwrap(ctx);
  auto sym = mlir::FlatSymbolRefAttr::get(mlirCtx, unwrap(codeSymbol));
  return wrap(qlx::fabric::SyndromeType::get(mlirCtx, sym));
}

MlirTypeID fabricSyndromeTypeGetTypeID(void) {
  return wrap(qlx::fabric::SyndromeType::getTypeID());
}

bool fabricTypeIsAResourceState(MlirType type) {
  return llvm::isa<qlx::fabric::ResourceStateType>(unwrap(type));
}

MlirType fabricResourceStateTypeGet(MlirContext ctx, MlirStringRef resource) {
  auto rt = qlx::fabric::symbolizeResourceType(unwrap(resource));
  if (!rt)
    return MlirType{nullptr};
  return wrap(qlx::fabric::ResourceStateType::get(unwrap(ctx), *rt));
}

MlirTypeID fabricResourceStateTypeGetTypeID(void) {
  return wrap(qlx::fabric::ResourceStateType::getTypeID());
}

//===----------------------------------------------------------------------===//
// Simple types
//===----------------------------------------------------------------------===//

bool fabricTypeIsABit(MlirType type) {
  return llvm::isa<qlx::fabric::BitType>(unwrap(type));
}
MlirType fabricBitTypeGet(MlirContext ctx) {
  return wrap(qlx::fabric::BitType::get(unwrap(ctx)));
}
MlirTypeID fabricBitTypeGetTypeID(void) {
  return wrap(qlx::fabric::BitType::getTypeID());
}

bool fabricTypeIsAFrame(MlirType type) {
  return llvm::isa<qlx::fabric::FrameType>(unwrap(type));
}
MlirType fabricFrameTypeGet(MlirContext ctx) {
  return wrap(qlx::fabric::FrameType::get(unwrap(ctx)));
}
MlirTypeID fabricFrameTypeGetTypeID(void) {
  return wrap(qlx::fabric::FrameType::getTypeID());
}

bool fabricTypeIsASlot(MlirType type) {
  return llvm::isa<qlx::fabric::SlotType>(unwrap(type));
}
MlirType fabricSlotTypeGet(MlirContext ctx) {
  return wrap(qlx::fabric::SlotType::get(unwrap(ctx)));
}
MlirTypeID fabricSlotTypeGetTypeID(void) {
  return wrap(qlx::fabric::SlotType::getTypeID());
}

bool fabricTypeIsAMachine(MlirType type) {
  return llvm::isa<qlx::fabric::MachineType>(unwrap(type));
}
MlirType fabricMachineTypeGet(MlirContext ctx) {
  return wrap(qlx::fabric::MachineType::get(unwrap(ctx)));
}
MlirTypeID fabricMachineTypeGetTypeID(void) {
  return wrap(qlx::fabric::MachineType::getTypeID());
}

//===----------------------------------------------------------------------===//
// Enum attributes
//===----------------------------------------------------------------------===//

#define DEFINE_FABRIC_ENUM_ATTR(NAME, MlirSymCheck, AttrClass, EnumSym,        \
                                EnumStringify)                                 \
  bool fabricAttributeIsA##NAME(MlirAttribute attr) {                          \
    return llvm::isa<qlx::fabric::AttrClass>(unwrap(attr));                    \
  }                                                                            \
  MlirAttribute fabric##NAME##AttrGet(MlirContext ctx, MlirStringRef value) {  \
    auto sym = qlx::fabric::EnumSym(unwrap(value));                            \
    if (!sym)                                                                  \
      return MlirAttribute{nullptr};                                           \
    return wrap(qlx::fabric::AttrClass::get(unwrap(ctx), *sym));               \
  }                                                                            \
  MlirStringRef fabric##NAME##AttrGetValue(MlirAttribute attr) {               \
    auto a = llvm::cast<qlx::fabric::AttrClass>(unwrap(attr));                 \
    return wrap(qlx::fabric::EnumStringify(a.getValue()));                     \
  }                                                                            \
  MlirTypeID fabric##NAME##AttrGetTypeID(void) {                               \
    return wrap(qlx::fabric::AttrClass::getTypeID());                          \
  }

DEFINE_FABRIC_ENUM_ATTR(Partition, _, PartitionAttr, symbolizePartition,
                        stringifyPartition)
DEFINE_FABRIC_ENUM_ATTR(Role, _, RoleAttr, symbolizeRole, stringifyRole)
DEFINE_FABRIC_ENUM_ATTR(Prep, _, PrepAttr, symbolizePrep, stringifyPrep)
DEFINE_FABRIC_ENUM_ATTR(MergeBasis, _, MergeBasisAttr, symbolizeMergeBasis,
                        stringifyMergeBasis)
DEFINE_FABRIC_ENUM_ATTR(Boundary, _, BoundaryAttr, symbolizeBoundary,
                        stringifyBoundary)
DEFINE_FABRIC_ENUM_ATTR(Resource, _, ResourceTypeAttr, symbolizeResourceType,
                        stringifyResourceType)

#undef DEFINE_FABRIC_ENUM_ATTR

//===----------------------------------------------------------------------===//
// Composite attributes
//===----------------------------------------------------------------------===//

bool fabricAttributeIsAFloorplan(MlirAttribute attr) {
  return llvm::isa<qlx::fabric::FloorplanAttr>(unwrap(attr));
}

MlirAttribute fabricFloorplanAttrGet(MlirContext ctx, MlirStringRef layout,
                                     intptr_t numParams,
                                     const int64_t *params) {
  auto sym = qlx::fabric::symbolizeLayout(unwrap(layout));
  if (!sym)
    return MlirAttribute{nullptr};
  llvm::ArrayRef<int64_t> ref(params, static_cast<size_t>(numParams));
  return wrap(qlx::fabric::FloorplanAttr::get(unwrap(ctx), *sym, ref));
}

MlirTypeID fabricFloorplanAttrGetTypeID(void) {
  return wrap(qlx::fabric::FloorplanAttr::getTypeID());
}

bool fabricAttributeIsAFlow(MlirAttribute attr) {
  return llvm::isa<qlx::fabric::FlowAttr>(unwrap(attr));
}

MlirAttribute fabricFlowAttrGet(MlirContext ctx, MlirStringRef xTo,
                                MlirStringRef zTo) {
  auto *mlirCtx = unwrap(ctx);
  auto x = mlir::StringAttr::get(mlirCtx, unwrap(xTo));
  auto z = mlir::StringAttr::get(mlirCtx, unwrap(zTo));
  return wrap(qlx::fabric::FlowAttr::get(mlirCtx, x, z));
}

MlirTypeID fabricFlowAttrGetTypeID(void) {
  return wrap(qlx::fabric::FlowAttr::getTypeID());
}

bool fabricAttributeIsASpecOnly(MlirAttribute attr) {
  return llvm::isa<qlx::fabric::SpecOnlyAttr>(unwrap(attr));
}

MlirAttribute fabricSpecOnlyAttrGet(MlirContext ctx, MlirStringRef name) {
  return wrap(qlx::fabric::SpecOnlyAttr::get(unwrap(ctx), unwrap(name)));
}

MlirStringRef fabricSpecOnlyAttrGetName(MlirAttribute attr) {
  auto a = llvm::cast<qlx::fabric::SpecOnlyAttr>(unwrap(attr));
  return wrap(a.getName());
}

MlirTypeID fabricSpecOnlyAttrGetTypeID(void) {
  return wrap(qlx::fabric::SpecOnlyAttr::getTypeID());
}
