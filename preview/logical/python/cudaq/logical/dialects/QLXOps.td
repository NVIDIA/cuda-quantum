//===- QLXOps.td - thin include shim for python bindings ---*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// declare_mlir_dialect_python_bindings invokes mlir-tblgen with this file
// as the entry point.  The canonical QLX op definitions live in
// ir/include/qlx/Dialect/QLX/IR/QLXOps.td -- this file just re-includes them
// so the Python op-binding generator runs against the same source of
// truth as the C++ side.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_PYTHON_QLXOPS_TD
#define QLX_PYTHON_QLXOPS_TD

include "qlx/Dialect/QLX/IR/QLXOps.td"

#endif // QLX_PYTHON_QLXOPS_TD
