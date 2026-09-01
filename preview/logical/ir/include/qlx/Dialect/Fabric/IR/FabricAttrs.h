//===- FabricAttrs.h - Fabric attribute declarations -------------*- C++
//-*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef FABRIC_FABRICATTRS_H
#define FABRIC_FABRICATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

// Enum definitions (Partition, MergeBasis, Boundary, Prep, Layout, Role, Route)
// Note: ResourceType enum is in FabricEnums.h.inc (generated from
// FabricTypes.td)
#include "qlx/Dialect/Fabric/IR/FabricAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "qlx/Dialect/Fabric/IR/FabricAttrs.h.inc"

#endif // FABRIC_FABRICATTRS_H
