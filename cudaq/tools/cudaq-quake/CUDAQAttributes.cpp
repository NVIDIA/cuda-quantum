/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/ParsedAttrInfo.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

// The attributes that CUDA-Q adds to the C++ attribute syntax, so that clang
// knows them: `[[cudaq::atomic_region]]`. Clang has no attribute that it
// can apply to a compound statement besides `annotate`, so each of these is
// an alias for the `annotate` attribute with the annotation that the bridge
// looks for. They are registered with clang's plugin mechanism.

namespace {

struct AtomicRegionAttrInfo : public clang::ParsedAttrInfo {
  AtomicRegionAttrInfo() {
    static constexpr Spelling spellings[] = {
        {clang::ParsedAttr::AS_CXX11, "cudaq::atomic_region"}};
    Spellings = spellings;
    IsStmt = 1;
  }

  /// On a declaration, it applies to functions.
  bool diagAppertainsToDecl(clang::Sema &S, const clang::ParsedAttr &attr,
                            const clang::Decl *decl) const override {
    if (llvm::isa<clang::FunctionDecl>(decl))
      return true;
    S.Diag(attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type)
        << attr << attr.isRegularKeywordAttribute()
        << clang::ExpectedFunctionOrMethod;
    return false;
  }

  /// On a statement, it applies to a compound statement. (The bridge reports
  /// an error for the other statements, to say that it is not supported.)
  bool diagAppertainsToStmt(clang::Sema &S, const clang::ParsedAttr &attr,
                            const clang::Stmt *stmt) const override {
    return true;
  }

  AttrHandling
  handleDeclAttribute(clang::Sema &S, clang::Decl *decl,
                      const clang::ParsedAttr &attr) const override {
    decl->addAttr(clang::AnnotateAttr::Create(
        S.Context, cudaq::atomicQuantumRegionAnnotation, nullptr, 0, attr));
    return AttributeApplied;
  }

  AttrHandling handleStmtAttribute(clang::Sema &S, clang::Stmt *stmt,
                                   const clang::ParsedAttr &attr,
                                   clang::Attr *&result) const override {
    result = clang::AnnotateAttr::Create(
        S.Context, cudaq::atomicQuantumRegionAnnotation, nullptr, 0, attr);
    return AttributeApplied;
  }
};

} // namespace

static clang::ParsedAttrInfoRegistry::Add<AtomicRegionAttrInfo>
    registerAtomicRegion("cudaq-atomic-region",
                         "the atomic quantum region attribute of CUDA-Q");
