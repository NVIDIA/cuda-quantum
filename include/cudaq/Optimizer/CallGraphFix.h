/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/Analysis/CallGraph.h"

namespace llvm {
// FIXME: `GraphTraits` specialization for `const mlir::CallGraphNode *` in
// "mlir/Analysis/CallGraph.h" has a bug.
// In particular, `GraphTraits<const mlir::CallGraphNode *>` typedef'ed `NodeRef
// -> mlir::CallGraphNode *`, (without `const`), causing problems when using
// `mlir::CallGraphNode` with graph iterator (e.g., `llvm::df_iterator`). The
// entry node getter has the signature `NodeRef getEntryNode(NodeRef node)`,
// i.e., `mlir::CallGraphNode * getEntryNode(mlir::CallGraphNode * node)`; but a
// graph iterator for `const mlir::CallGraphNode *` will pass a `const
// mlir::CallGraphNode *` to that `getEntryNode` function => compile error.
// Here, we define a non-const overload, which hasn't been defined, to work
// around that issue.
//
// Note: this isn't an issue for the whole `mlir::CallGraph` graph, i.e.,
// `GraphTraits<const mlir::CallGraph *>`. `getEntryNode` is defined as
// `getExternalCallerNode`, which is a const method of `mlir::CallGraph`.

template <>
struct GraphTraits<mlir::CallGraphNode *> {
  using NodeRef = mlir::CallGraphNode *;
  static NodeRef getEntryNode(NodeRef node) { return node; }

  static NodeRef unwrap(const mlir::CallGraphNode::Edge &edge) {
    return edge.getTarget();
  }
  using ChildIteratorType =
      mapped_iterator<mlir::CallGraphNode::iterator, decltype(&unwrap)>;
  static ChildIteratorType child_begin(NodeRef node) {
    return {node->begin(), &unwrap};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node->end(), &unwrap};
  }
};
} // namespace llvm
