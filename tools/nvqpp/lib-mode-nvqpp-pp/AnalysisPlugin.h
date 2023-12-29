/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "KernelCharacteristics.h"
#include "common/Registry.h"
#include "clang/AST/Decl.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace cudaq {

/// @brief The AnalysisPlugin provides a hook for developers to
/// inject a Clang AST visitor on a quantum kernel function when compiling
/// in library-mode. By implementing the `traverseTree` method, sub-types have
/// an opportunity to analyze the tree, rewrite the source code, or emit
/// compiler diagnostics.
class AnalysisPlugin : public registry::RegisteredType<AnalysisPlugin> {
public:
  /// Define a type to describe a `FunctionDecl` and its
  /// `KernelCharacteristics`.
  using Function = std::pair<KernelCharacteristics, clang::FunctionDecl *>;

  /// @brief Traverse the given `FunctionDecl` tree and apply any
  /// sub-type specific rewrites or throw errors / warnings. Implementations
  /// have access to the pre-computed `KernelCharacteristics` for the kernel
  /// represented by the given `FunctionDecl`.
  virtual void traverseTree(const Function &function,
                            clang::DiagnosticsEngine &de,
                            clang::Rewriter &rewriter) = 0;
};

} // namespace cudaq