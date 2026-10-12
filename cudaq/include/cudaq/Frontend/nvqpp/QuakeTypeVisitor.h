/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Frontend/nvqpp/ASTResultVisitor.h"
#include "clang/AST/Mangle.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Builders.h"
#include <cstdint>
#include <utility>

namespace cudaq::detail {

/// Find the `operator()` of a class, if it has one.
clang::FunctionDecl *findCallOperator(const clang::CXXRecordDecl *decl);

/// Is this a class that will be ignored by the bridge?
/// FIXME: This is a bit of a hack to skip over certain AST nodes.
bool ignoredClass(clang::RecordDecl *x);

/// Converts clang types to MLIR types (Quake and CC dialects, etc.). The result
/// of traversing a `clang::QualType` is the corresponding `mlir::Type`. A type
/// that has no MLIR equivalent (such as an ignored class) has no result.
class QuakeTypeVisitor : public ASTResultVisitor<QuakeTypeVisitor, mlir::Type> {
public:
  QuakeTypeVisitor(clang::ASTContext *astCtx, mlir::OpBuilder &bldr,
                   clang::ItaniumMangleContext *mangler, bool tuplesAreReversed)
      : astContext(astCtx), builder(bldr), mangler(mangler),
        tuplesAreReversed(tuplesAreReversed) {}

  /// Convert \p t, a builtin type, to the corresponding MLIR type.
  mlir::Type builtinTypeToType(const clang::BuiltinType *t);

  // Handlers. Only the types that need non-default behavior have one. Sugar
  // is looked through by default.
  Result visit(clang::BuiltinType *t);
  Result visit(clang::PointerType *t);
  Result visit(clang::LValueReferenceType *t);
  Result visit(clang::RValueReferenceType *t);
  Result visit(clang::ConstantArrayType *t);
  Result visit(clang::FunctionProtoType *t);
  Result visit(clang::RecordType *t);
  /// A kind of type that is not sugar, has no handler, and has no children that
  /// the base knows is not supported (yet), whether it never was or is new to
  /// clang. It is an error. It is not ignored.
  void unsupported(clang::Type *t);

  /// When determining a kernel's signature, an unknown record type is converted
  /// to `none` instead of being an error. The caller is expected to diagnose
  /// the signature afterwards.
  bool allowUnknownRecordType = false;

private:
  /// Convert a record, \p x, that is not (yet) in the cache.
  Result convertRecord(clang::RecordDecl *x);
  /// Some records are replaced with high-level types in Quake. Sets \p
  /// intercepted if \p x was one of these, in which case the result is the
  /// replacement (if it has one).
  Result interceptRecordDecl(clang::RecordDecl *x, bool &intercepted);
  /// Convert the product type \p x, whose fields' types are \p fieldTys.
  Result convertProductType(clang::RecordDecl *x,
                            llvm::ArrayRef<mlir::Type> fieldTys);
  /// Traverse \p qt. It is an error if there is no result.
  Result requireType(clang::SourceRange loc, clang::QualType qt);

  std::pair<std::uint64_t, unsigned> getWidthAndAlignment(clang::RecordDecl *x);
  mlir::Location toLocation(const clang::SourceRange &range);

  clang::ASTContext *astContext;
  mlir::OpBuilder &builder;
  clang::ItaniumMangleContext *mangler;
  const bool tuplesAreReversed;
  /// Cache of converted record types.
  llvm::DenseMap<const clang::RecordDecl *, mlir::Type> records;
  /// The records that are being converted. A record that refers to itself,
  /// directly or through other records, is recursive. That cannot be a type of
  /// a kernel.
  llvm::DenseSet<const clang::RecordDecl *> converting;
};

} // namespace cudaq::detail
