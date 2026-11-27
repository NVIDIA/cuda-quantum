/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Every node class must be complete for the dispatch switches below.
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclOpenACC.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenACC.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtSYCL.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Stack.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <variant>

/// \file
/// An AST visitor that threads a result value through the traversal.

// clang's `RecursiveASTVisitor` only threads a `bool` (continue / stop)
// through a traversal, so a visitor that builds something (IR, a type, a
// string, ...) has to smuggle its results around on side stacks. This
// visitor instead has every node visit return a `std::optional<R>`.
//
//  - `std::nullopt` means the node produced no value. This is *not* an error.
//    Statements, for example, typically produce no value.
//  - Errors are state in the visitor: call `fail()` and test `hasFailed()`.
//    Once the visitor has failed, traversal of sibling nodes stops.
//
// Handlers are overloads of `visit` on the clang node class. There is no
// `VisitIfStmt`; there is `visit(clang::IfStmt *)`. A handler written for a
// base class (`visit(clang::CastExpr *)`) is used for all of its subclasses
// that have no handler of their own. There is no need to "walk up".
//
//   Result visit(X *x);
//      The handler is in full control and decides which children to traverse
//      and in what order, by calling `traverse` (or `traverseAll`) itself.
//
//   Result visit(X *x, ChildResults<R> &kids);
//      The children have already been traversed (in order) and their results
//      are in `kids`. (A null child, like the absent `init` of a `for` loop,
//      is a `nullopt` entry so that positions stay stable.)
//
// If both shapes are applicable to a node, the first (full control) is used.
// A node with neither is traversed by default: the children are traversed and
// the result is `std::nullopt`; sugared types are replaced by the type they
// desugar to. Handlers must be public members of `Derived`.
//
// If `Derived` defines `Result unhandled(X *)`, that is called for each node
// that has no handler of either shape, instead of the default. This lets a
// visitor be introduced a node class at a time, where the rest of the nodes
// are visited some other way.
//
// The kinds of node that can be traversed are `clang::Stmt` (including all
// expressions), `clang::Decl`, `clang::QualType`, `clang::CXXCtorInitializer`
// and `clang::TemplateArgument`.
//
// Children
// --------
//
// The children of a node are the sub-nodes that `RecursiveASTVisitor` would
// visit, in the same order, with these differences. Types are \e semantic
// `QualType`s, so that `auto fn(auto p)` has a type to look at. The type of a
// declarator or a function return type is a child; the type of an arbitrary
// expression is not, but a type that is written as a part of an expression
// (the target of an explicit cast, `sizeof(T)`, `new T`, ...) is. A function's
// parameters are `ParmVarDecl` children, and their types and default arguments
// are the children of those.
//
// Whether implicit code, template instantiations, and lambda bodies are
// visited is controlled by the same optional hooks that `RecursiveASTVisitor`
// uses. Define any of these in `Derived`:
//
//   bool shouldVisitImplicitCode();            // default: false
//   bool shouldVisitTemplateInstantiations();  // default: false
//   bool shouldVisitLambdaBody();              // default: true
//
// Implicit code includes implicit declarations (such as an implicitly
// defined copy constructor, its member initializers and its body), implicit
// constructor initializers, default arguments, and the semantic form of an
// `InitListExpr` (the syntactic form is the child otherwise).
//
// A type that the base has no knowledge about (it is not sugar, has no
// handler and has no known children) is reported by calling the optional
// hook `void unsupported(clang::Type *)`, if `Derived` has one.
//
// To use, derive via CRTP, pick `R` (a `std::variant` if a node kind produces
// different things), and call `traverse(node)` to start.
//
//   struct V : cudaq::detail::ASTResultVisitor<V, mlir::Type> {
//     std::optional<mlir::Type> visit(clang::PointerType *t) {
//       auto pointee = traverse(t->getPointeeType());
//       if (!pointee) { fail(); return std::nullopt; }
//       return cc::PointerType::get(*pointee);
//     }
//   };
//
// Differences from `RecursiveASTVisitor`
// --------------------------------------
//
// These are deliberate, and are checked by the unit tests that compare the
// two visitors over a corpus.
//  - The semantic form of an `InitListExpr` (with its array filler) is
//    visited once, if implicit code is visited. `RecursiveASTVisitor` visits
//    both forms, and visits their common children twice.
//  - The parameters of a function are its `ParmVarDecl`s, even when the
//    function was declared with a type that is not written as a function
//    (`__typeof(f)`, or a typedef of a function type).
//  - Syntax-only parts are not visited: expressions inside types (array
//    bounds, `decltype`, `noexcept`), attributes, names and qualifiers, and
//    the parameter declarations of a function type that is written inside
//    another type.
//
// Termination
// -----------
//
// The children of a node are the nodes it contains. A reference (the callee of
// a call, the type of a record) is never followed by the base, so traversing a
// node terminates. A handler that follows a reference can revisit a node that
// it is already inside of (a recursive function, a record that has a pointer
// to itself) and must break that cycle itself, as `QuakeTypeVisitor` does with
// its cache of records. The base only stops a runaway traversal: nesting
// deeper than `setMaxDepth` fails the visitor (and if assertions are enabled,
// so does a node that is its own ancestor).
//
// Clang's data recursion queue is not used. A very deeply nested expression
// (such as `a + b + c + ...` with thousands of terms) recurses, and uses about
// 2 KB of native stack for each level. Like clang's own recursive code, the
// traversal checks how much stack is left at each node and continues on a new
// stack when it is nearly exhausted (`clang::runWithSufficientStackSpace`).

namespace cudaq::detail {

template <typename T>
struct IsVariant : std::false_type {};
template <typename... Ts>
struct IsVariant<std::variant<Ts...>> : std::true_type {};

/// The results of traversing the children of a node, in child order.
template <typename R>
class ChildResults {
public:
  using Entry = std::optional<R>;

  std::size_t size() const { return entries.size(); }
  bool empty() const { return entries.empty(); }
  const Entry &operator[](std::size_t i) const { return entries[i]; }
  auto begin() const { return entries.begin(); }
  auto end() const { return entries.end(); }
  void push_back(Entry e) { entries.push_back(std::move(e)); }

  // Get the i-th child's result as a `T`. Returns `nullopt` if that child
  // produced no value, or (if `R` is a variant) produced something else.
  template <typename T>
  std::optional<T> get(std::size_t i) const {
    if (i >= entries.size() || !entries[i])
      return std::nullopt;
    if (auto *p = as<T>(*entries[i]))
      return *p;
    return std::nullopt;
  }

  /// All of the children's results that are a `T`, in order, skipping the first
  /// \p skip children. Children with no value are omitted. This is how, for
  /// example, the arguments of a call are packaged as a list.
  template <typename T>
  llvm::SmallVector<T> values(std::size_t skip = 0) const {
    llvm::SmallVector<T> result;
    for (std::size_t i = skip; i < entries.size(); ++i)
      if (entries[i])
        if (auto *p = as<T>(*entries[i]))
          result.push_back(*p);
    return result;
  }

private:
  template <typename T>
  static const T *as(const R &r) {
    if constexpr (std::is_same_v<R, T>) {
      return &r;
    } else if constexpr (IsVariant<R>::value) {
      return std::get_if<T>(&r);
    } else {
      static_assert(sizeof(T) == 0, "T is not a possible result type");
    }
  }

  llvm::SmallVector<Entry, 4> entries;
};

/// Which optional parts of the AST are visited.
struct ChildPolicy {
  bool implicitCode = false;
  bool templateInstantiations = false;
  bool lambdaBody = true;
};

// How to enumerate the children of a node.
//
// `enumerate(node, policy, fn)` calls `fn` with each child, in order, until
// `fn` returns false. A child is a `clang::Stmt *`, `clang::Decl *`,
// `clang::QualType`, `clang::CXXCtorInitializer *` or a `const
// clang::TemplateArgument &`; any of these may be null. The return value is
// false only if the kind of node is not known (see
// `ASTResultVisitor::unsupported`). Overload resolution selects the most
// derived overload, so a node class without its own overload inherits the
// children of its base class.
namespace ast_children {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Calls to the helpers return false iff the enumeration was stopped.

template <typename F>
bool stmtChildren(clang::Stmt *x, F &fn) {
  for (clang::Stmt *c : x->children())
    if (!fn(c))
      return false;
  return true;
}

template <typename F>
bool templateParameters(clang::TemplateParameterList *tpl, F &fn) {
  if (!tpl)
    return true;
  for (clang::NamedDecl *d : *tpl)
    if (!fn(static_cast<clang::Decl *>(d)))
      return false;
  return fn(static_cast<clang::Stmt *>(tpl->getRequiresClause()));
}

// Children of a `DeclContext`. Lambda classes (visited through their
// `LambdaExpr`), blocks and captured decls (visited through their statements)
// are not visited here.
template <typename F>
bool declContext(clang::DeclContext *dc, F &fn) {
  if (!dc)
    return true;
  for (clang::Decl *child : dc->decls()) {
    if (llvm::isa<clang::BlockDecl>(child) ||
        llvm::isa<clang::CapturedDecl>(child))
      continue;
    if (auto *cls = llvm::dyn_cast<clang::CXXRecordDecl>(child))
      if (cls->isLambda())
        continue;
    if (!fn(child))
      return false;
  }
  return true;
}

template <typename F>
bool templateArguments(const clang::TemplateArgumentLoc *args, unsigned count,
                       F &fn) {
  for (unsigned i = 0; i < count; ++i)
    if (!fn(args[i].getArgument()))
      return false;
  return true;
}

/// The children of a type constraint (`template <Concept T>`): the expression
/// that clang synthesizes for it if implicit code is visited, and otherwise the
/// explicit arguments of the concept.
template <typename F>
bool typeConstraint(const clang::TypeConstraint *tc, const ChildPolicy &p,
                    F &fn) {
  if (!tc)
    return true;
  if (auto *constraint = tc->getImmediatelyDeclaredConstraint();
      constraint && p.implicitCode)
    return fn(static_cast<clang::Stmt *>(constraint));
  if (auto *written = tc->getTemplateArgsAsWritten())
    return templateArguments(written->getTemplateArgs(),
                             written->NumTemplateArgs, fn);
  return true;
}

// The template parameter lists from outer templates of a declarator or tag.
template <typename T, typename F>
bool outerTemplateParameters(T *x, F &fn) {
  for (unsigned i = 0; i < x->getNumTemplateParameterLists(); ++i)
    if (!templateParameters(x->getTemplateParameterList(i), fn))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Stmt
//===----------------------------------------------------------------------===//

template <typename F>
bool enumerate(clang::Stmt *x, const ChildPolicy &, F &&fn) {
  stmtChildren(x, fn);
  return true;
}

// The decls contain the initializers; do not also visit those.
template <typename F>
bool enumerate(clang::DeclStmt *x, const ChildPolicy &, F &&fn) {
  for (clang::Decl *d : x->decls())
    if (!fn(d))
      break;
  return true;
}

template <typename F>
bool enumerate(clang::CXXForRangeStmt *x, const ChildPolicy &p, F &&fn) {
  if (p.implicitCode) {
    stmtChildren(x, fn);
    return true;
  }
  // The source order of what is written.
  [[maybe_unused]] auto unused =
      ((!x->getInit() || fn(static_cast<clang::Stmt *>(x->getInit()))) &&
       fn(static_cast<clang::Stmt *>(x->getLoopVarStmt())) &&
       fn(static_cast<clang::Stmt *>(x->getRangeInit())) &&
       fn(static_cast<clang::Stmt *>(x->getBody())));
  return true;
}

template <typename F>
bool enumerate(clang::CXXCatchStmt *x, const ChildPolicy &, F &&fn) {
  if (fn(static_cast<clang::Decl *>(x->getExceptionDecl())))
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::CXXDefaultArgExpr *x, const ChildPolicy &p, F &&fn) {
  if (p.implicitCode)
    fn(static_cast<clang::Stmt *>(x->getExpr()));
  return true;
}

template <typename F>
bool enumerate(clang::CXXDefaultInitExpr *x, const ChildPolicy &p, F &&fn) {
  if (p.implicitCode)
    fn(static_cast<clang::Stmt *>(x->getExpr()));
  return true;
}

/// An `InitListExpr` has a syntactic form (what was written) and a semantic
/// form (what is initialized, with implicit conversions and any array filler).
/// The child is the syntactic form unless implicit code is visited, and then it
/// is the semantic form.
template <typename F>
bool enumerate(clang::InitListExpr *x, const ChildPolicy &p, F &&fn) {
  clang::InitListExpr *form = x;
  if (p.implicitCode) {
    if (!x->isSemanticForm())
      if (auto *sem = x->getSemanticForm())
        form = sem;
  } else {
    if (x->isSemanticForm())
      if (auto *syn = x->getSyntacticForm())
        form = syn;
  }
  if (!stmtChildren(form, fn))
    return true;
  if (p.implicitCode && form->hasArrayFiller())
    fn(static_cast<clang::Stmt *>(form->getArrayFiller()));
  return true;
}

template <typename F>
bool enumerate(clang::LambdaExpr *x, const ChildPolicy &p, F &&fn) {
  // The captures. An init capture is declared; others are initialized.
  for (unsigned i = 0, n = x->capture_size(); i != n; ++i) {
    const clang::LambdaCapture *c = x->capture_begin() + i;
    if (!c->isExplicit() && !p.implicitCode)
      continue;
    bool more =
        x->isInitCapture(c)
            ? fn(static_cast<clang::Decl *>(c->getCapturedVar()))
            : fn(static_cast<clang::Stmt *>(x->capture_init_begin()[i]));
    if (!more)
      return true;
  }
  if (p.implicitCode) {
    // Everything else is in the lambda class.
    fn(static_cast<clang::Decl *>(x->getLambdaClass()));
    return true;
  }
  if (!templateParameters(x->getTemplateParameterList(), fn))
    return true;
  clang::CXXMethodDecl *callOp = x->getCallOperator();
  if (x->hasExplicitParameters())
    for (clang::ParmVarDecl *parm : callOp->parameters())
      if (!fn(static_cast<clang::Decl *>(parm)))
        return true;
  if (x->hasExplicitResultType())
    if (!fn(callOp->getReturnType()))
      return true;
  if (!fn(static_cast<clang::Stmt *>(const_cast<clang::Expr *>(
          x->getTrailingRequiresClause().ConstraintExpr))))
    return true;
  fn(static_cast<clang::Stmt *>(x->getBody()));
  return true;
}

// Explicit template arguments.
template <typename F>
bool enumerate(clang::DeclRefExpr *x, const ChildPolicy &, F &&fn) {
  templateArguments(x->getTemplateArgs(), x->getNumTemplateArgs(), fn);
  return true;
}

template <typename F>
bool enumerate(clang::MemberExpr *x, const ChildPolicy &, F &&fn) {
  if (!templateArguments(x->getTemplateArgs(), x->getNumTemplateArgs(), fn))
    return true;
  stmtChildren(x, fn);
  return true;
}

// The asm string, the constraints and the clobbers are expressions, then the
// operands.
template <typename F>
bool enumerate(clang::GCCAsmStmt *x, const ChildPolicy &, F &&fn) {
  if (!fn(static_cast<clang::Stmt *>(x->getAsmStringExpr())))
    return true;
  for (unsigned i = 0, n = x->getNumInputs(); i < n; ++i)
    if (!fn(static_cast<clang::Stmt *>(x->getInputConstraintExpr(i))))
      return true;
  for (unsigned i = 0, n = x->getNumOutputs(); i < n; ++i)
    if (!fn(static_cast<clang::Stmt *>(x->getOutputConstraintExpr(i))))
      return true;
  for (unsigned i = 0, n = x->getNumClobbers(); i < n; ++i)
    if (!fn(static_cast<clang::Stmt *>(x->getClobberExpr(i))))
      return true;
  stmtChildren(x, fn);
  return true;
}

// The explicit arguments of the concept.
template <typename F>
bool enumerate(clang::ConceptSpecializationExpr *x, const ChildPolicy &,
               F &&fn) {
  if (auto *written = x->getTemplateArgsAsWritten())
    templateArguments(written->getTemplateArgs(), written->NumTemplateArgs, fn);
  return true;
}

// A temporary whose lifetime is extended is declared. The declaration has the
// expression that is the temporary.
template <typename F>
bool enumerate(clang::MaterializeTemporaryExpr *x, const ChildPolicy &,
               F &&fn) {
  if (auto *decl = x->getLifetimeExtendedTemporaryDecl())
    fn(static_cast<clang::Decl *>(decl));
  else
    stmtChildren(x, fn);
  return true;
}

// Explicit template arguments of expressions that are not resolved (because
// they are dependent), then the other children.
template <typename F>
bool enumerate(clang::UnresolvedLookupExpr *x, const ChildPolicy &, F &&fn) {
  if (x->hasExplicitTemplateArgs())
    templateArguments(x->getTemplateArgs(), x->getNumTemplateArgs(), fn);
  return true;
}

template <typename F>
bool enumerate(clang::DependentScopeDeclRefExpr *x, const ChildPolicy &,
               F &&fn) {
  if (x->hasExplicitTemplateArgs())
    templateArguments(x->getTemplateArgs(), x->getNumTemplateArgs(), fn);
  return true;
}

template <typename F>
bool enumerate(clang::UnresolvedMemberExpr *x, const ChildPolicy &, F &&fn) {
  if (x->hasExplicitTemplateArgs())
    if (!templateArguments(x->getTemplateArgs(), x->getNumTemplateArgs(), fn))
      return true;
  stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::CXXDependentScopeMemberExpr *x, const ChildPolicy &,
               F &&fn) {
  if (x->hasExplicitTemplateArgs())
    if (!templateArguments(x->getTemplateArgs(), x->getNumTemplateArgs(), fn))
      return true;
  stmtChildren(x, fn);
  return true;
}

// The type written as the target of an explicit cast, then the operand.
template <typename F>
bool enumerate(clang::ExplicitCastExpr *x, const ChildPolicy &, F &&fn) {
  if (fn(x->getTypeAsWritten()))
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::CXXNewExpr *x, const ChildPolicy &, F &&fn) {
  if (fn(x->getAllocatedType()))
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::CXXScalarValueInitExpr *x, const ChildPolicy &, F &&fn) {
  fn(x->getTypeSourceInfo()->getType());
  return true;
}

template <typename F>
bool enumerate(clang::CompoundLiteralExpr *x, const ChildPolicy &, F &&fn) {
  if (fn(x->getTypeSourceInfo()->getType()))
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::OffsetOfExpr *x, const ChildPolicy &, F &&fn) {
  if (fn(x->getTypeSourceInfo()->getType()))
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::UnaryExprOrTypeTraitExpr *x, const ChildPolicy &,
               F &&fn) {
  if (x->isArgumentType())
    fn(x->getArgumentType());
  else
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::CXXTypeidExpr *x, const ChildPolicy &, F &&fn) {
  if (x->isTypeOperand())
    fn(x->getTypeOperandSourceInfo()->getType());
  else
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::VAArgExpr *x, const ChildPolicy &, F &&fn) {
  if (fn(x->getWrittenTypeInfo()->getType()))
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::CXXUnresolvedConstructExpr *x, const ChildPolicy &,
               F &&fn) {
  if (fn(x->getTypeAsWritten()))
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::TypeTraitExpr *x, const ChildPolicy &, F &&fn) {
  for (unsigned i = 0, n = x->getNumArgs(); i != n; ++i)
    if (!fn(x->getArg(i)->getType()))
      break;
  return true;
}

template <typename F>
bool enumerate(clang::ArrayTypeTraitExpr *x, const ChildPolicy &, F &&fn) {
  if (fn(x->getQueriedType()))
    stmtChildren(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::BlockExpr *x, const ChildPolicy &, F &&fn) {
  fn(static_cast<clang::Decl *>(x->getBlockDecl()));
  return true;
}

template <typename F>
bool enumerate(clang::CapturedStmt *x, const ChildPolicy &, F &&fn) {
  if (fn(static_cast<clang::Decl *>(x->getCapturedDecl())))
    stmtChildren(x, fn);
  return true;
}

// The syntactic form and the sources of the semantic expressions.
template <typename F>
bool enumerate(clang::PseudoObjectExpr *x, const ChildPolicy &, F &&fn) {
  if (!fn(static_cast<clang::Stmt *>(x->getSyntacticForm())))
    return true;
  for (clang::Expr *sub : x->semantics()) {
    if (auto *ove = llvm::dyn_cast<clang::OpaqueValueExpr>(sub))
      sub = ove->getSourceExpr();
    if (!fn(static_cast<clang::Stmt *>(sub)))
      break;
  }
  return true;
}

template <typename F>
bool enumerate(clang::ArrayInitLoopExpr *x, const ChildPolicy &, F &&fn) {
  if (auto *common = x->getCommonExpr())
    if (!fn(static_cast<clang::Stmt *>(common->getSourceExpr())))
      return true;
  stmtChildren(x, fn);
  return true;
}

// Visit the operands as written unless implicit code is visited.
template <typename F>
bool enumerate(clang::CXXRewrittenBinaryOperator *x, const ChildPolicy &p,
               F &&fn) {
  if (p.implicitCode) {
    stmtChildren(x, fn);
    return true;
  }
  auto form = x->getDecomposedForm();
  [[maybe_unused]] auto unused =
      (fn(static_cast<clang::Stmt *>(const_cast<clang::Expr *>(form.LHS))) &&
       fn(static_cast<clang::Stmt *>(const_cast<clang::Expr *>(form.RHS))));
  return true;
}

// Coroutines: only what was written unless implicit code is visited.
template <typename F>
bool enumerate(clang::CoroutineBodyStmt *x, const ChildPolicy &p, F &&fn) {
  if (p.implicitCode)
    stmtChildren(x, fn);
  else
    fn(static_cast<clang::Stmt *>(x->getBody()));
  return true;
}

template <typename F>
bool enumerate(clang::CoreturnStmt *x, const ChildPolicy &p, F &&fn) {
  if (p.implicitCode)
    stmtChildren(x, fn);
  else
    fn(static_cast<clang::Stmt *>(x->getOperand()));
  return true;
}

template <typename F>
bool enumerate(clang::CoroutineSuspendExpr *x, const ChildPolicy &p, F &&fn) {
  if (p.implicitCode)
    stmtChildren(x, fn);
  else
    fn(static_cast<clang::Stmt *>(x->getOperand()));
  return true;
}

template <typename F>
bool enumerate(clang::DependentCoawaitExpr *x, const ChildPolicy &p, F &&fn) {
  if (p.implicitCode)
    stmtChildren(x, fn);
  else
    fn(static_cast<clang::Stmt *>(x->getOperand()));
  return true;
}

/// A requires expression: the body, the local parameters, and the requirements.
template <typename F>
bool enumerate(clang::RequiresExpr *x, const ChildPolicy &p, F &&fn) {
  if (!fn(static_cast<clang::Decl *>(x->getBody())))
    return true;
  for (clang::ParmVarDecl *parm : x->getLocalParameters())
    if (!fn(static_cast<clang::Decl *>(parm)))
      return true;
  for (clang::concepts::Requirement *req : x->getRequirements()) {
    switch (req->getKind()) {
    case clang::concepts::Requirement::RK_Type: {
      auto *typeReq = llvm::cast<clang::concepts::TypeRequirement>(req);
      if (!typeReq->isSubstitutionFailure())
        if (!fn(typeReq->getType()->getType()))
          return true;
      break;
    }
    case clang::concepts::Requirement::RK_Simple:
    case clang::concepts::Requirement::RK_Compound: {
      auto *exprReq = llvm::cast<clang::concepts::ExprRequirement>(req);
      if (!exprReq->isExprSubstitutionFailure())
        if (!fn(static_cast<clang::Stmt *>(exprReq->getExpr())))
          return true;
      auto &ret = exprReq->getReturnTypeRequirement();
      if (ret.isTypeConstraint()) {
        // The template parameter list is implicit.
        if (p.implicitCode) {
          if (!templateParameters(ret.getTypeConstraintTemplateParameterList(),
                                  fn))
            return true;
        } else if (!typeConstraint(ret.getTypeConstraint(), p, fn)) {
          return true;
        }
      }
      break;
    }
    case clang::concepts::Requirement::RK_Nested: {
      auto *nested = llvm::cast<clang::concepts::NestedRequirement>(req);
      if (!nested->hasInvalidConstraint())
        if (!fn(static_cast<clang::Stmt *>(nested->getConstraintExpr())))
          return true;
      break;
    }
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Decl
//===----------------------------------------------------------------------===//

// Declarations that are `DeclContext`s but have no more specific overload have
// the decls they contain as children.
template <typename F>
bool enumerate(clang::Decl *x, const ChildPolicy &, F &&fn) {
  declContext(llvm::dyn_cast<clang::DeclContext>(x), fn);
  return true;
}

template <typename F>
bool enumerate(clang::LifetimeExtendedTemporaryDecl *x, const ChildPolicy &,
               F &&fn) {
  fn(static_cast<clang::Stmt *>(x->getTemporaryExpr()));
  return true;
}

template <typename F>
bool enumerate(clang::FileScopeAsmDecl *x, const ChildPolicy &, F &&fn) {
  fn(static_cast<clang::Stmt *>(x->getAsmStringExpr()));
  return true;
}

template <typename F>
bool enumerate(clang::TopLevelStmtDecl *x, const ChildPolicy &, F &&fn) {
  fn(static_cast<clang::Stmt *>(x->getStmt()));
  return true;
}

template <typename F>
bool enumerate(clang::TypedefNameDecl *x, const ChildPolicy &, F &&fn) {
  fn(x->getUnderlyingType());
  return true;
}

template <typename F>
bool enumerate(clang::EnumConstantDecl *x, const ChildPolicy &, F &&fn) {
  fn(static_cast<clang::Stmt *>(x->getInitExpr()));
  return true;
}

template <typename F>
bool enumerate(clang::StaticAssertDecl *x, const ChildPolicy &, F &&fn) {
  if (fn(static_cast<clang::Stmt *>(x->getAssertExpr())))
    fn(static_cast<clang::Stmt *>(x->getMessage()));
  return true;
}

/// A friend is a type or a declaration. A class that is declared by the friend
/// type is not in the parent context, so it is also a child.
template <typename F>
bool enumerate(clang::FriendDecl *x, const ChildPolicy &, F &&fn) {
  if (auto *tsi = x->getFriendType()) {
    if (!fn(tsi->getType()))
      return true;
    if (auto *tt = tsi->getType()->getAs<clang::TagType>();
        tt && tt->isTagOwned())
      fn(static_cast<clang::Decl *>(tt->getDecl()));
  } else {
    fn(static_cast<clang::Decl *>(x->getFriendDecl()));
  }
  return true;
}

template <typename F>
bool enumerate(clang::BlockDecl *x, const ChildPolicy &, F &&fn) {
  if (!fn(static_cast<clang::Stmt *>(x->getBody())))
    return true;
  for (const auto &capture : x->captures())
    if (capture.hasCopyExpr())
      if (!fn(static_cast<clang::Stmt *>(capture.getCopyExpr())))
        break;
  return true;
}

template <typename F>
bool enumerate(clang::CapturedDecl *x, const ChildPolicy &, F &&fn) {
  fn(static_cast<clang::Stmt *>(x->getBody()));
  return true;
}

/// Declarators: the template parameter lists of outer templates, then the type.
template <typename F>
bool declarator(clang::DeclaratorDecl *x, F &fn) {
  return outerTemplateParameters(x, fn) && fn(x->getType());
}

template <typename F>
bool enumerate(clang::DeclaratorDecl *x, const ChildPolicy &, F &&fn) {
  declarator(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::FieldDecl *x, const ChildPolicy &, F &&fn) {
  [[maybe_unused]] auto unused =
      (declarator(x, fn) &&
       (!x->isBitField() || fn(static_cast<clang::Stmt *>(x->getBitWidth()))) &&
       (!x->hasInClassInitializer() ||
        fn(static_cast<clang::Stmt *>(x->getInClassInitializer()))));
  return true;
}

template <typename F>
bool enumerate(clang::VarDecl *x, const ChildPolicy &p, F &&fn) {
  if (!declarator(x, fn))
    return true;
  // Default arguments of a parameter are children of the ParmVarDecl.
  if (!llvm::isa<clang::ParmVarDecl>(x) &&
      (!x->isCXXForRangeDecl() || p.implicitCode))
    fn(static_cast<clang::Stmt *>(x->getInit()));
  return true;
}

template <typename F>
bool enumerate(clang::ParmVarDecl *x, const ChildPolicy &p, F &&fn) {
  if (!declarator(x, fn))
    return true;
  if (x->hasDefaultArg() && !x->hasUnparsedDefaultArg()) {
    if (x->hasUninstantiatedDefaultArg())
      fn(static_cast<clang::Stmt *>(x->getUninstantiatedDefaultArg()));
    else
      fn(static_cast<clang::Stmt *>(x->getDefaultArg()));
  }
  return true;
}

template <typename F>
bool enumerate(clang::DecompositionDecl *x, const ChildPolicy &p, F &&fn) {
  if (!enumerate(static_cast<clang::VarDecl *>(x), p, fn))
    return false;
  for (clang::BindingDecl *binding : x->bindings())
    if (!fn(static_cast<clang::Decl *>(binding)))
      break;
  return true;
}

/// A structured binding. Its binding expression (and holding variable, for a
/// tuple-like binding) is implicit code.
template <typename F>
bool enumerate(clang::BindingDecl *x, const ChildPolicy &p, F &&fn) {
  if (p.implicitCode)
    if (fn(static_cast<clang::Stmt *>(x->getBinding())))
      if (auto *holding = x->getHoldingVar())
        fn(static_cast<clang::Decl *>(holding));
  return true;
}

template <typename F>
bool enumerate(clang::NonTypeTemplateParmDecl *x, const ChildPolicy &, F &&fn) {
  if (!declarator(x, fn))
    return true;
  if (x->hasDefaultArgument() && !x->defaultArgumentWasInherited())
    fn(x->getDefaultArgument().getArgument());
  return true;
}

// A type parameter: the type constraint (`template <Concept T>`), then the
// default argument.
template <typename F>
bool enumerate(clang::TemplateTypeParmDecl *x, const ChildPolicy &p, F &&fn) {
  if (!typeConstraint(x->getTypeConstraint(), p, fn))
    return true;
  if (x->hasDefaultArgument() && !x->defaultArgumentWasInherited())
    fn(x->getDefaultArgument().getArgument());
  return true;
}

// A function: the template parameter lists of outer templates, explicit
// template arguments, the return type, the parameters, the trailing requires
// clause, constructor initializers, and the body. The decls contained by the
// function (its parameters) are not visited again.
template <typename F>
bool enumerate(clang::FunctionDecl *x, const ChildPolicy &p, F &&fn) {
  if (!outerTemplateParameters(x, fn))
    return true;
  // The arguments of an explicit specialization.
  if (auto *info = x->getTemplateSpecializationInfo()) {
    auto kind = info->getTemplateSpecializationKind();
    if (kind != clang::TSK_Undeclared &&
        kind != clang::TSK_ImplicitInstantiation)
      if (auto *written = info->TemplateArgumentsAsWritten)
        if (!templateArguments(written->getTemplateArgs(),
                               written->NumTemplateArgs, fn))
          return true;
  } else if (auto *dep = x->getDependentSpecializationInfo()) {
    if (auto *written = dep->TemplateArgumentsAsWritten)
      if (!templateArguments(written->getTemplateArgs(),
                             written->NumTemplateArgs, fn))
        return true;
  }
  if (!fn(x->getReturnType()))
    return true;
  for (clang::ParmVarDecl *parm : x->parameters())
    if (!fn(static_cast<clang::Decl *>(parm)))
      return true;
  if (!fn(static_cast<clang::Stmt *>(const_cast<clang::Expr *>(
          x->getTrailingRequiresClause().ConstraintExpr))))
    return true;
  if (auto *ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(x))
    for (clang::CXXCtorInitializer *init : ctor->inits())
      if (init->isWritten() || p.implicitCode)
        if (!fn(init))
          return true;
  // Do not visit the body of a function that clang generated unless implicit
  // code is requested.
  bool visitBody = x->isThisDeclarationADefinition() &&
                   (!x->isDefaulted() || p.implicitCode);
  if (auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(x))
    if (const clang::CXXRecordDecl *cls = method->getParent())
      if (cls->isLambda() &&
          declaresSameEntity(cls->getLambdaCallOperator(), method))
        visitBody = visitBody && p.lambdaBody;
  if (visitBody) {
    if (!fn(static_cast<clang::Stmt *>(x->getBody())))
      return true;
    // The body may contain using declarations whose shadows are parented to the
    // function.
    for (clang::Decl *child : x->decls())
      if (llvm::isa<clang::UsingShadowDecl>(child))
        if (!fn(child))
          return true;
  }
  return true;
}

/// A record: the template parameter lists of outer templates, the bases (if
/// this is a definition), and the contained declarations.
template <typename F>
bool enumerate(clang::RecordDecl *x, const ChildPolicy &, F &&fn) {
  if (outerTemplateParameters(x, fn))
    declContext(x, fn);
  return true;
}

template <typename F>
bool cxxRecord(clang::CXXRecordDecl *x, F &fn) {
  if (!outerTemplateParameters(x, fn))
    return false;
  if (x->isCompleteDefinition())
    for (const auto &base : x->bases())
      if (!fn(base.getType()))
        return false;
  return declContext(x, fn);
}

template <typename F>
bool enumerate(clang::CXXRecordDecl *x, const ChildPolicy &, F &&fn) {
  cxxRecord(x, fn);
  return true;
}

/// An implicit instantiation is not written anywhere in the source, so unless
/// instantiations are visited (or this is an explicit specialization), it has
/// no children.
template <typename F>
bool enumerate(clang::ClassTemplateSpecializationDecl *x, const ChildPolicy &p,
               F &&fn) {
  if (auto *written = x->getTemplateArgsAsWritten())
    if (!templateArguments(written->getTemplateArgs(), written->NumTemplateArgs,
                           fn))
      return true;
  if (p.templateInstantiations ||
      x->getSpecializationKind() == clang::TSK_ExplicitSpecialization)
    cxxRecord(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::ClassTemplatePartialSpecializationDecl *x,
               const ChildPolicy &, F &&fn) {
  if (!templateParameters(x->getTemplateParameters(), fn))
    return true;
  if (auto *written = x->getTemplateArgsAsWritten())
    if (!templateArguments(written->getTemplateArgs(), written->NumTemplateArgs,
                           fn))
      return true;
  cxxRecord(x, fn);
  return true;
}

template <typename F>
bool enumerate(clang::VarTemplateSpecializationDecl *x, const ChildPolicy &p,
               F &&fn) {
  if (auto *written = x->getTemplateArgsAsWritten())
    if (!templateArguments(written->getTemplateArgs(), written->NumTemplateArgs,
                           fn))
      return true;
  if (p.templateInstantiations ||
      x->getSpecializationKind() == clang::TSK_ExplicitSpecialization)
    enumerate(static_cast<clang::VarDecl *>(x), p, fn);
  return true;
}

template <typename F>
bool enumerate(clang::VarTemplatePartialSpecializationDecl *x,
               const ChildPolicy &p, F &&fn) {
  if (!templateParameters(x->getTemplateParameters(), fn))
    return true;
  if (auto *written = x->getTemplateArgsAsWritten())
    if (!templateArguments(written->getTemplateArgs(), written->NumTemplateArgs,
                           fn))
      return true;
  enumerate(static_cast<clang::VarDecl *>(x), p, fn);
  return true;
}

/// A template: the parameters, the templated declaration, and, optionally, the
/// instantiations that were not written in the source.
template <typename F>
bool enumerate(clang::TemplateDecl *x, const ChildPolicy &, F &&fn) {
  if (templateParameters(x->getTemplateParameters(), fn))
    fn(static_cast<clang::Decl *>(x->getTemplatedDecl()));
  return true;
}

// For an alias template, the aliased type is visited first.
template <typename F>
bool enumerate(clang::TypeAliasTemplateDecl *x, const ChildPolicy &, F &&fn) {
  if (fn(static_cast<clang::Decl *>(x->getTemplatedDecl())))
    templateParameters(x->getTemplateParameters(), fn);
  return true;
}

template <typename F>
bool enumerate(clang::ConceptDecl *x, const ChildPolicy &, F &&fn) {
  if (templateParameters(x->getTemplateParameters(), fn))
    fn(static_cast<clang::Stmt *>(x->getConstraintExpr()));
  return true;
}

template <typename F>
bool enumerate(clang::ClassTemplateDecl *x, const ChildPolicy &p, F &&fn) {
  if (!enumerate(static_cast<clang::TemplateDecl *>(x), p, fn))
    return false;
  if (p.templateInstantiations && x == x->getCanonicalDecl())
    for (auto *spec : x->specializations())
      for (auto *redecl : spec->redecls()) {
        auto kind = llvm::cast<clang::ClassTemplateSpecializationDecl>(redecl)
                        ->getSpecializationKind();
        if (kind == clang::TSK_Undeclared ||
            kind == clang::TSK_ImplicitInstantiation)
          if (!fn(static_cast<clang::Decl *>(redecl)))
            return true;
      }
  return true;
}

template <typename F>
bool enumerate(clang::VarTemplateDecl *x, const ChildPolicy &p, F &&fn) {
  if (!enumerate(static_cast<clang::TemplateDecl *>(x), p, fn))
    return false;
  if (p.templateInstantiations && x == x->getCanonicalDecl())
    for (auto *spec : x->specializations())
      for (auto *redecl : spec->redecls()) {
        auto kind = llvm::cast<clang::VarTemplateSpecializationDecl>(redecl)
                        ->getSpecializationKind();
        if (kind == clang::TSK_Undeclared ||
            kind == clang::TSK_ImplicitInstantiation)
          if (!fn(static_cast<clang::Decl *>(redecl)))
            return true;
      }
  return true;
}

template <typename F>
bool enumerate(clang::FunctionTemplateDecl *x, const ChildPolicy &p, F &&fn) {
  if (!enumerate(static_cast<clang::TemplateDecl *>(x), p, fn))
    return false;
  if (p.templateInstantiations && x == x->getCanonicalDecl())
    for (auto *spec : x->specializations())
      for (auto *redecl : spec->redecls()) {
        if (redecl->getTemplateSpecializationKind() ==
            clang::TSK_ExplicitSpecialization)
          continue;
        if (!fn(static_cast<clang::Decl *>(redecl)))
          return true;
      }
  return true;
}

//===----------------------------------------------------------------------===//
// Other nodes
//===----------------------------------------------------------------------===//

/// The base type of a base initializer, and the initializing expression (if it
/// was written or implicit code is visited).
template <typename F>
bool enumerate(clang::CXXCtorInitializer *x, const ChildPolicy &p, F &&fn) {
  if (auto *tsi = x->getTypeSourceInfo())
    if (!fn(tsi->getType()))
      return true;
  if (x->isWritten() || p.implicitCode)
    fn(static_cast<clang::Stmt *>(x->getInit()));
  return true;
}

template <typename F>
bool enumerate(clang::TemplateArgument *x, const ChildPolicy &, F &&fn) {
  switch (x->getKind()) {
  case clang::TemplateArgument::Type:
    fn(x->getAsType());
    break;
  case clang::TemplateArgument::Expression:
    fn(static_cast<clang::Stmt *>(x->getAsExpr()));
    break;
  case clang::TemplateArgument::Pack:
    for (const clang::TemplateArgument &element : x->pack_elements())
      if (!fn(element))
        break;
    break;
  default:
    break;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

/// The catch-all for a type that is not known: returns false.
template <typename F>
bool enumerate(clang::Type *, const ChildPolicy &, F &&) {
  return false;
}

// Types that have no children.
template <typename F>
bool enumerate(clang::BuiltinType *, const ChildPolicy &, F &&) {
  return true;
}
template <typename F>
bool enumerate(clang::TagType *, const ChildPolicy &, F &&) {
  return true;
}
template <typename F>
bool enumerate(clang::TemplateTypeParmType *, const ChildPolicy &, F &&) {
  return true;
}
template <typename F>
bool enumerate(clang::InjectedClassNameType *, const ChildPolicy &, F &&) {
  return true;
}

template <typename F>
bool enumerate(clang::PointerType *x, const ChildPolicy &, F &&fn) {
  fn(x->getPointeeType());
  return true;
}
template <typename F>
bool enumerate(clang::BlockPointerType *x, const ChildPolicy &, F &&fn) {
  fn(x->getPointeeType());
  return true;
}
template <typename F>
bool enumerate(clang::ReferenceType *x, const ChildPolicy &, F &&fn) {
  fn(x->getPointeeType());
  return true;
}
template <typename F>
bool enumerate(clang::MemberPointerType *x, const ChildPolicy &, F &&fn) {
  fn(x->getPointeeType());
  return true;
}
template <typename F>
bool enumerate(clang::ArrayType *x, const ChildPolicy &, F &&fn) {
  fn(x->getElementType());
  return true;
}
template <typename F>
bool enumerate(clang::ComplexType *x, const ChildPolicy &, F &&fn) {
  fn(x->getElementType());
  return true;
}
template <typename F>
bool enumerate(clang::VectorType *x, const ChildPolicy &, F &&fn) {
  fn(x->getElementType());
  return true;
}
template <typename F>
bool enumerate(clang::AtomicType *x, const ChildPolicy &, F &&fn) {
  fn(x->getValueType());
  return true;
}
template <typename F>
bool enumerate(clang::PackExpansionType *x, const ChildPolicy &, F &&fn) {
  fn(x->getPattern());
  return true;
}
template <typename F>
bool enumerate(clang::FunctionNoProtoType *x, const ChildPolicy &, F &&fn) {
  fn(x->getReturnType());
  return true;
}

/// Result type first, then the parameter types in order.
template <typename F>
bool enumerate(clang::FunctionProtoType *x, const ChildPolicy &, F &&fn) {
  if (!fn(x->getReturnType()))
    return true;
  for (auto t : x->param_types())
    if (!fn(t))
      break;
  return true;
}

} // namespace ast_children

/// Closed-world checks for a class that has a handler that is also used for
/// the classes that derive from it.
///
/// A handler for a base class (`visit(clang::CastExpr *)`) is used for every
/// class that derives from it. That is the point, but a class that clang adds
/// later (or moves) under that base class would be given the handler without
/// anyone having looked at it. `derivedStmtsAreKnown<Base, Known...>()` and
/// `derivedDeclsAreKnown<Base, Known...>()` fail to compile, and name the class
/// in the template arguments of `UnacknowledgedNode`, if there is a class
/// derived from `Base` that is not one of `Known`. This only looks at the
/// classes below the bases that have such handlers, which is a few classes and
/// is a part of the AST that does not change often, not at the whole AST.
///
///   static_assert(derivedStmtsAreKnown<clang::CallExpr,
///                                      clang::CXXMemberCallExpr, ...>());
template <typename Node, typename Base>
struct UnacknowledgedNode {
  static_assert(!std::is_same_v<Node, Node>,
                "A class of clang's AST derives from a class that has a "
                "handler, which is also the handler of the derived class. "
                "Decide if that is right for the class that is named in the "
                "template arguments of UnacknowledgedNode, and add it to the "
                "list of the classes that were looked at.");
};

template <typename Base, typename... Known>
constexpr bool derivedStmtsAreKnown() {
#define ABSTRACT_STMT(STMT) STMT
#define STMT(CLASS, PARENT)                                                    \
  if constexpr (std::is_base_of_v<Base, clang::CLASS> &&                       \
                !std::is_same_v<Base, clang::CLASS> &&                         \
                !(std::is_same_v<clang::CLASS, Known> || ...))                 \
      [[maybe_unused]]                                                         \
    constexpr auto unused = sizeof(UnacknowledgedNode<clang::CLASS, Base>);
#include "clang/AST/StmtNodes.inc"
  return true;
}

template <typename Base, typename... Known>
constexpr bool derivedDeclsAreKnown() {
#define ABSTRACT_DECL(DECL) DECL
#define DECL(CLASS, BASE)                                                      \
  if constexpr (std::is_base_of_v<Base, clang::CLASS##Decl> &&                 \
                !std::is_same_v<Base, clang::CLASS##Decl> &&                   \
                !(std::is_same_v<clang::CLASS##Decl, Known> || ...))           \
      [[maybe_unused]]                                                         \
    constexpr auto unused =                                                    \
        sizeof(UnacknowledgedNode<clang::CLASS##Decl, Base>);
#include "clang/AST/DeclNodes.inc"
  return true;
}

template <typename Derived, typename R>
class ASTResultVisitor {
public:
  using Result = std::optional<R>;
  using Children = ChildResults<R>;

  //===--------------------------------------------------------------------===//
  // Entry points
  //===--------------------------------------------------------------------===//

  /// Traverse a statement or expression. A null \p s yields `nullopt`.
  Result traverse(clang::Stmt *s) {
    if (!s || hasFailed())
      return std::nullopt;
    Guard guard(*this, s);
    if (!guard)
      return std::nullopt;
    Result result;
    clang::runWithSufficientStackSpace([] {},
                                       [&] { result = dispatchStmt(s); });
    return result;
  }

  /// Traverse a declaration. A null \p d yields `nullopt`. Implicit
  /// declarations are not traversed unless implicit code is visited.
  Result traverse(clang::Decl *d) {
    if (!d || hasFailed())
      return std::nullopt;
    if (!visitsImplicitCode() && d->isImplicit())
      return std::nullopt;
    Guard guard(*this, d);
    if (!guard)
      return std::nullopt;
    Result result;
    clang::runWithSufficientStackSpace([] {},
                                       [&] { result = dispatchDecl(d); });
    return result;
  }

  /// Traverse a type. Qualifiers are ignored. A null \p qt yields `nullopt`.
  Result traverse(clang::QualType qt) {
    if (qt.isNull() || hasFailed())
      return std::nullopt;
    Guard guard(*this, nullptr);
    if (!guard)
      return std::nullopt;
    Result result;
    clang::runWithSufficientStackSpace(
        [] {},
        [&] {
          result = dispatchType(const_cast<clang::Type *>(qt.getTypePtr()));
        });
    return result;
  }

  /// Traverse a constructor initializer. Null yields `nullopt`.
  Result traverse(clang::CXXCtorInitializer *init) {
    if (!init || hasFailed())
      return std::nullopt;
    Guard guard(*this, init);
    if (!guard)
      return std::nullopt;
    Result result;
    clang::runWithSufficientStackSpace([] {}, [&] { result = dispatch(init); });
    return result;
  }

  /// Traverse a template argument.
  Result traverse(const clang::TemplateArgument &arg) {
    if (hasFailed())
      return std::nullopt;
    Guard guard(*this, nullptr);
    if (!guard)
      return std::nullopt;
    Result result;
    clang::runWithSufficientStackSpace(
        [] {},
        [&] {
          result = dispatch(const_cast<clang::TemplateArgument *>(&arg));
        });
    return result;
  }

  /// Traverse each node in \p nodes in order. Returns the results that have a
  /// value. Stops at the first failure.
  template <typename Range>
  llvm::SmallVector<R> traverseAll(Range &&nodes) {
    llvm::SmallVector<R> result;
    for (auto node : nodes) {
      if (auto r = traverse(node))
        result.push_back(std::move(*r));
      if (hasFailed())
        break;
    }
    return result;
  }

  //===--------------------------------------------------------------------===//
  // Error state
  //===--------------------------------------------------------------------===//

  /// Has an error been signaled? Traversal stops after an error.
  bool hasFailed() const { return failureFlag; }
  /// Signal an error. The result is "no value", so that a handler that fails
  /// can `return fail();`.
  Result fail() {
    failureFlag = true;
    return std::nullopt;
  }
  void clearFailure() { failureFlag = false; }

  /// The deepest nesting of nodes that is traversed. A traversal that goes
  /// deeper fails (see `depthExceeded`). The native stack is not what limits
  /// the depth: when it is nearly exhausted, the traversal continues on a new
  /// stack, as clang does. Real ASTs are trees, or are walked as trees, but a
  /// handler that follows references (such as the callee of a call) can reach a
  /// node that it is already inside of, and then this is what stops it.
  void setMaxDepth(unsigned depth) { maxDepth = depth; }
  bool depthExceeded() const { return hitMaxDepth; }

protected:
  Derived &self() { return static_cast<Derived &>(*this); }

  bool visitsImplicitCode() {
    if constexpr (requires { self().shouldVisitImplicitCode(); })
      return self().shouldVisitImplicitCode();
    else
      return false;
  }

  ChildPolicy childPolicy() {
    ChildPolicy policy;
    policy.implicitCode = visitsImplicitCode();
    if constexpr (requires { self().shouldVisitTemplateInstantiations(); })
      policy.templateInstantiations =
          self().shouldVisitTemplateInstantiations();
    if constexpr (requires { self().shouldVisitLambdaBody(); })
      policy.lambdaBody = self().shouldVisitLambdaBody();
    return policy;
  }

  /// Traverse the children of \p x, collecting their results in \p kids.
  /// Returns false if the base does not know what the children of \p x are.
  template <typename X>
  bool traverseChildren(X *x, Children &kids) {
    return ast_children::enumerate(x, childPolicy(), [&](auto &&child) {
      if (hasFailed())
        return false;
      kids.push_back(traverse(child));
      return !hasFailed();
    });
  }

private:
  /// Counts the nesting of traversals, and (if assertions are enabled) detects
  /// a node that is traversed while it is already being traversed.
  class Guard {
  public:
    Guard(ASTResultVisitor &v, const void *node) : visitor(v), node(node) {
      // The traversal can nest more deeply than the stack is big, and then it
      // goes on (see `runWithSufficientStackSpace`). This is where the stack
      // starts.
      if (visitor.depth == 0)
        clang::noteBottomOfStack();
      if (visitor.depth >= visitor.maxDepth) {
        visitor.hitMaxDepth = true;
        visitor.fail();
        return;
      }
#ifndef NDEBUG
      if (node && !visitor.active.insert(node).second) {
        // A cycle: this node is its own ancestor.
        visitor.hitMaxDepth = true;
        visitor.fail();
        return;
      }
#endif
      ++visitor.depth;
      entered = true;
    }
    ~Guard() {
      if (!entered)
        return;
      --visitor.depth;
#ifndef NDEBUG
      if (node)
        visitor.active.erase(node);
#endif
    }
    explicit operator bool() const { return entered; }

  private:
    ASTResultVisitor &visitor;
    const void *node;
    bool entered = false;
  };

  Result dispatchStmt(clang::Stmt *s) {
    switch (s->getStmtClass()) {
    case clang::Stmt::NoStmtClass:
      return std::nullopt;
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                                    \
  case clang::Stmt::CLASS##Class:                                              \
    return dispatch(static_cast<clang::CLASS *>(s));
#include "clang/AST/StmtNodes.inc"
    }
    return std::nullopt;
  }

  Result dispatchDecl(clang::Decl *d) {
    switch (d->getKind()) {
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE)                                                      \
  case clang::Decl::CLASS:                                                     \
    return dispatch(static_cast<clang::CLASS##Decl *>(d));
#include "clang/AST/DeclNodes.inc"
    }
    return std::nullopt;
  }

  Result dispatchType(clang::Type *t) {
    switch (t->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, BASE)
#define TYPE(CLASS, BASE)                                                      \
  case clang::Type::CLASS:                                                     \
    return dispatch(static_cast<clang::CLASS##Type *>(t));
#include "clang/AST/TypeNodes.inc"
    }
    return std::nullopt;
  }

  template <typename X>
  Result dispatch(X *x) {
    Derived &d = self();
    if constexpr (requires {
                    { d.visit(x) } -> std::convertible_to<Result>;
                  }) {
      return d.visit(x);
    } else if constexpr (requires(Children &kids) {
                           { d.visit(x, kids) } -> std::convertible_to<Result>;
                         }) {
      Children kids;
      traverseChildren(x, kids);
      if (hasFailed())
        return std::nullopt;
      return d.visit(x, kids);
    } else if constexpr (requires {
                           { d.unhandled(x) } -> std::convertible_to<Result>;
                         }) {
      return d.unhandled(x);
    } else {
      return defaultVisit(x);
    }
  }

public:
  /// The default visit of a node: visit its children, and then there is no
  /// result. Sugar is looked through for a type. A handler can use this for a
  /// node that has a handler that it should not use, such as a subclass.
  template <typename X>
  Result defaultVisit(X *x) {
    [[maybe_unused]] Derived &d = self();
    if constexpr (std::is_base_of_v<clang::Type, X>) {
      // Look through sugar, otherwise there is nothing to do for a type that
      // has no handler.
      clang::QualType next = x->getLocallyUnqualifiedSingleStepDesugaredType();
      if (next.getTypePtr() != static_cast<const clang::Type *>(x))
        return traverse(next);
      Children kids;
      if (!traverseChildren(x, kids)) {
        if constexpr (requires { d.unsupported(x); })
          d.unsupported(x);
      }
      return std::nullopt;
    } else {
      Children kids;
      traverseChildren(x, kids);
      return std::nullopt;
    }
  }

private:
  bool failureFlag = false;
  bool hitMaxDepth = false;
  unsigned depth = 0;
  unsigned maxDepth = 1 << 16;
#ifndef NDEBUG
  llvm::SmallPtrSet<const void *, 32> active;
#endif
};

} // namespace cudaq::detail
