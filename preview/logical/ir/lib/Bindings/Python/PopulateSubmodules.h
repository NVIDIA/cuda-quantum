/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
//
// Internal header: declares the per-dialect "populate" functions used by
// the single-module `_qlx_ext` extension.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>

namespace qlx::python {

/// Populate the "qlx" submodule with QLX dialect enum attrs and helpers
/// (PauliAttr, set_inherent_attr, ...).
void populateQLXSubmodule(nanobind::module_ &m);

/// Populate the "fabric" submodule with Fabric dialect types and attrs
/// (PatchType, PartitionAttr, FloorplanAttr, ...).
void populateFabricSubmodule(nanobind::module_ &m);

/// Populate the "event" submodule with the Event dialect's shared handle
/// type (HandleType).
void populateEventSubmodule(nanobind::module_ &m);

} // namespace qlx::python
