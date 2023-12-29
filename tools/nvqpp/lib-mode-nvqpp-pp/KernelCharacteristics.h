/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace cudaq {

/// @brief `KernelCharacteristics` is a simple utility
/// struct that provides extra information about a CUDA Quantum kernel
/// and how it is called from host code.
struct KernelCharacteristics {
  /// @brief Does the kernel return `void`?
  bool voidReturn = false;
  /// @brief Does the kernel have measurements specified?
  bool hasMeasurements = false;
  /// @brief Is the kernel used with `cudaq::observe()`?
  bool isObserved = false;
  /// @brief What is the kernel function name?
  std::string name;
  /// can add more in the futute ...
};

using namespace clang;

/// @brief Visit the AST for the given kernel `FunctionDecl` and
/// set any pertinent flags on the given `KernelCharacteristics` instance.
class ApplyQuantumKernelCharacteristics
    : public RecursiveASTVisitor<ApplyQuantumKernelCharacteristics> {
public:
  /// @brief This Kernel's characteristics reference.
  cudaq::KernelCharacteristics &characteristics;

  ApplyQuantumKernelCharacteristics(
      cudaq::KernelCharacteristics &characteristics)
      : characteristics(characteristics) {}

  /// @brief Visit kernels and get their name and if they return `void`.
  bool VisitFunctionDecl(FunctionDecl *decl) {
    if (auto attr = decl->getAttr<clang::AnnotateAttr>()) {
      if (attr->getAnnotation().str() == "quantum") {
        characteristics.name = [&]() {
          auto func = decl->getDefinition();
          if (func) {
            if (isa<CXXMethodDecl>(func))
              return cast<CXXRecordDecl>(func->getParent())->getName();
          }
          return decl->getName();
        }();
      }

      if (decl->getReturnType()->isVoidType())
        characteristics.voidReturn = true;
    }
    return true;
  }

  /// @brief Visit `DecleRefExprs` and see if any of them are
  /// user-specified measurements.
  bool VisitDeclRefExpr(DeclRefExpr *expr) {
    auto funcDecl = dyn_cast<FunctionDecl>(expr->getDecl());
    if (!funcDecl)
      return true;

    if (!funcDecl->getDeclName().isIdentifier())
      return true;

    auto name = funcDecl->getName();
    std::vector<std::string> allowed{"mx", "my", "mz"};
    if (std::find(allowed.begin(), allowed.end(), name.str()) != allowed.end())
      characteristics.hasMeasurements = true;

    return true;
  }
};

/// @brief Visit the classical `FunctionDecl` pointers in the main source file
/// to get extra information about how CUDA Quantum kernels are invoked.
class ApplyHostFunctionCharacteristics
    : public RecursiveASTVisitor<ApplyHostFunctionCharacteristics> {
protected:
  /// Map kernel names to their characteristics.
  std::map<FunctionDecl *, cudaq::KernelCharacteristics> &characteristics;

public:
  ApplyHostFunctionCharacteristics(
      std::map<FunctionDecl *, cudaq::KernelCharacteristics> &characteristics)
      : characteristics(characteristics) {}

  /// @brief Visit classical call expressions, see if they call
  /// a kernel and from what classical function.
  bool VisitCallExpr(CallExpr *expr) {
    struct FindObserveDeclRef
        : public clang::RecursiveASTVisitor<FindObserveDeclRef> {
      bool isObserve = false;
      bool VisitDeclRefExpr(DeclRefExpr *expr) {
        auto funcDecl = dyn_cast<FunctionDecl>(expr->getDecl());
        if (!funcDecl)
          return true;

        if (!funcDecl->getDeclName().isIdentifier())
          return true;
        auto name = funcDecl->getName();
        if (name.equals("observe")) {
          isObserve = true;
          return false;
        }
        return true;
      }
    };

    FindObserveDeclRef vis;
    vis.TraverseCallExpr(expr);
    if (vis.isObserve) {
      // check the first arg
      auto rawDecl = dyn_cast<DeclRefExpr>(expr->getArg(0))->getDecl();
      auto fdecl = dyn_cast<FunctionDecl>(rawDecl);
      for (auto &[unusedDecl, c] : characteristics)
        if (fdecl->getName().equals(c.name))
          c.isObserved = true;
    }
    return true;
  }
};

} // namespace cudaq