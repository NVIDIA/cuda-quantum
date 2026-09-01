//===- PopulateSubmodules.h - Internal binding helpers ---------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// Internal header: declares the per-dialect "populate" functions used by
// the single-module `_qlx_ext` extension.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_BINDINGS_POPULATE_SUBMODULES_H
#define QLX_BINDINGS_POPULATE_SUBMODULES_H

#include <nanobind/nanobind.h>

namespace qlx::python {

/// Populate the "qlx" submodule with QLX dialect types, enumerated attributes,
/// and product helpers (PauliAttr and set_inherent_attr).
void populateQLXSubmodule(nanobind::module_ &m);

/// Populate the "fabric" submodule with Fabric dialect types and attributes
/// (PatchType, PartitionAttr, FloorplanAttr, ...).
void populateFabricSubmodule(nanobind::module_ &m);

} // namespace qlx::python

#endif // QLX_BINDINGS_POPULATE_SUBMODULES_H
