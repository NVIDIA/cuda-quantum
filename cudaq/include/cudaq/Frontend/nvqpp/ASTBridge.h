/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Frontend/nvqpp/QuakeTypeVisitor.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Todo.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/InitAllDialects.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cudaq::detail {
/// Report a clang error diagnostic. Note that the message must be a string
/// literal. \p astNode is a node from the clang AST with source location
/// information.
template <typename T, unsigned N>
void reportClangError(T *astNode, clang::DiagnosticsEngine &de,
                      const char (&msg)[N]) {
  auto id = de.getCustomDiagID(clang::DiagnosticsEngine::Error, msg);
  de.Report(astNode->getBeginLoc(), id);
}
template <typename T, unsigned N>
void reportClangError(T *astNode, clang::ItaniumMangleContext *mangler,
                      const char (&msg)[N]) {
  reportClangError(astNode, mangler->getASTContext().getDiagnostics(), msg);
}

/// `measure_handle` arrives at the bridge as either an SSA `!cc.measure_handle`
/// or as the pointer form `!cc.ptr<!cc.measure_handle>` left by lvalue access
/// (named variable read, `operator=` LHS, struct-member-of-handle, ...). Most
/// consumers want the value form, so funnel that normalization through one
/// helper rather than open-coding the `dyn_cast` chain at every call site.
inline mlir::Value loadHandleIfPointer(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value v) {
  if (auto ptrTy = mlir::dyn_cast<cudaq::cc::PointerType>(v.getType()))
    if (mlir::isa<cudaq::cc::MeasureHandleType>(ptrTy.getElementType()))
      return cudaq::cc::LoadOp::create(builder, loc, v);
  return v;
}

/// Same intent as `loadHandleIfPointer`, but for the bulk-discriminate /
/// `to_integer` / range-for / qec.{detector,observable,pair_detectors} paths
/// where the lvalue carries a `std::vector<measure_handle>` and `ConvertDecl`
/// has stack-allocated a descriptor slot for it.
inline mlir::Value loadHandleVectorIfPointer(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value v) {
  if (auto ptrTy = mlir::dyn_cast<cudaq::cc::PointerType>(v.getType()))
    if (auto sv =
            mlir::dyn_cast<cudaq::cc::SequenceType>(ptrTy.getElementType());
        sv && mlir::isa<cudaq::cc::MeasureHandleType>(sv.getElementType()))
      return cudaq::cc::LoadOp::create(builder, loc, v);
  return v;
}
} // namespace cudaq::detail

#undef TODO_BRIDGE
#undef TODO_x

#if defined(NDEBUG) || defined(CUDAQ_NOTRACEBACKS)
#define TODO_BRIDGE(MlirLoc, ToDoXPtr, ToDoMangler, ToDoMsg, ToDoFile,         \
                    ToDoLine)                                                  \
  cudaq::detail::reportClangError(ToDoXPtr, ToDoMangler,                       \
                                  ToDoMsg " is not yet supported");
#else
#define TODO_BRIDGE(MlirLoc, ToDoXPtr, ToDoMangler, ToDoMsg, ToDoFile,         \
                    ToDoLine)                                                  \
  do {                                                                         \
    mlir::emitError(MlirLoc, llvm::Twine(ToDoFile ":" TODOQUOTE(               \
                                 ToDoLine) ": not yet implemented: ") +        \
                                 ToDoMsg);                                     \
    cudaq::detail::reportClangError(ToDoXPtr, ToDoMangler,                     \
                                    ToDoMsg " is not yet supported");          \
  } while (false);
#endif

// TODO for inside the bridge. This TODO will always expand to a clang error
// message. If assertions are enabled this will add source location information
// from the compiler as well. This TODO does not support tracebacks for the
// developer (these errors are not catastrophic, they are user errors) so is
// more suited to the user.
#define TODO_x(MlirLoc, ToDoXPtr, ToDoMangler, ToDoMsg)                        \
  TODO_BRIDGE(MlirLoc, ToDoXPtr, ToDoMangler, ToDoMsg, __FILE__, __LINE__)

// TODO: Enable lowering the transitive closure of the call graph reachable from
// kernel entry points.
#define LOWERING_TRANSITIVE_CLOSURE false

namespace cudaq {

using EmittedFunctionPair = std::pair<std::string, const clang::FunctionDecl *>;
using EmittedFunctionsCollection = std::deque<EmittedFunctionPair>;
using MangledKernelNamesMap = std::map<std::string, std::string>;
using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, mlir::Value>;
using SymbolTableScope =
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;

/// Convert a clang::SourceRange to an mlir::Location.
mlir::Location toSourceLocation(mlir::MLIRContext *ctx,
                                clang::ASTContext *astCtx,
                                const clang::SourceRange &srcRange);

namespace detail {

/// Use the name mangler to create a unique name for this declaration. This
/// unique name can be used to unique the MLIR name of a quantum kernel.
std::string getCxxMangledDeclName(clang::GlobalDecl decl,
                                  clang::ItaniumMangleContext *mangler);

/// Use the name mangler to create a unique name for this type. Used with lambda
/// expressions. The unique name will also be available to the programmer
/// through the use of `typeid(lambda).name()` for introspection, looking up the
/// kernel code.
std::string getCxxMangledTypeName(clang::QualType ty,
                                  clang::ItaniumMangleContext *mangler);

/// Use this helper to convert a tag name to an `nvq++` mangled name.
inline std::string getCudaqKernelName(const std::string &tag) {
  return runtime::cudaqGenPrefixName + tag;
}

/// Creates the tag name for a quantum kernel. The tag name is a name by which
/// one can lookup a kernel at runtime. This name does not include the `nvq++`
/// prefix nor the unique (C++ mangled) suffix.
std::string getTagNameOfFunctionDecl(const clang::FunctionDecl *func,
                                     clang::ItaniumMangleContext *mangler);

//===----------------------------------------------------------------------===//
// QuakeBridgeVisitor
//===----------------------------------------------------------------------===//

/// The values of the operands of a node of the AST that is being lowered. The
/// last operand is on the top. Lowering a node takes its operands from the
/// stack and pushes the value that the node computes, if it computes one. The
/// value of the node is what is left on top.
class OperandStack {
public:
  OperandStack() = default;
  explicit OperandStack(llvm::SmallVector<mlir::Value> operands)
      : values(std::move(operands)) {}

  bool push(mlir::Value v) {
    values.push_back(v);
    return true;
  }
  mlir::Value pop() {
    assert(!values.empty() && "no operands left");
    mlir::Value result = values.back();
    values.pop_back();
    return result;
  }
  mlir::Value peek() const {
    assert(!values.empty() && "no operands left");
    return values.back();
  }
  /// Remove the last \p n operands and return them in left-to-right (natural)
  /// order. For a call, `foo(a, b, c)` this can be used to return a list
  /// `[value_a value_b value_c]`.
  llvm::SmallVector<mlir::Value> last(unsigned n) {
    assert(n <= values.size() && "stack has fewer values than requested");
    llvm::SmallVector<mlir::Value> result(values.end() - n, values.end());
    values.pop_back_n(n);
    return result;
  }
  std::size_t size() const { return values.size(); }
  bool empty() const { return values.empty(); }
  mlir::Value operator[](std::size_t i) const { return values[i]; }

private:
  llvm::SmallVector<mlir::Value> values;
};

/// The nodes that do nothing but wrap a statement or an expression, and are
/// visited by visiting what they wrap. A kind of node that is not here and does
/// not have a handler is not supported in a kernel.
template <typename X>
inline constexpr bool isTransparentNode =
    std::is_same_v<X, clang::ParenExpr> ||
    std::is_same_v<X, clang::ExprWithCleanups> ||
    std::is_same_v<X, clang::CXXBindTemporaryExpr> ||
    std::is_same_v<X, clang::CXXStdInitializerListExpr> ||
    std::is_same_v<X, clang::SubstNonTypeTemplateParmExpr> ||
    std::is_same_v<X, clang::ConstantExpr> ||
    std::is_same_v<X, clang::NullStmt> || std::is_same_v<X, clang::LabelStmt>;

/// What a node of the AST produces when it is visited. At present, only
/// expressions produce a result: the value that is computed.
using BridgeResult = std::variant<mlir::Value, mlir::Type>;

/// QuakeBridgeVisitor is a visitor pattern for crawling over the AST and
/// generating Quake, CC, and other MLIR dialects.
///
/// It is an `ASTResultVisitor`. Each node of the AST is visited by the handler
/// for its class (`visit(clang::IfStmt *)`, ...), and a handler for a base
/// class is used for the classes that derive from it. An expression produces a
/// result: the value that it computes, which its parent gets from the visit of
/// the operand (`traverseValue`). Types are converted by the `QuakeTypeVisitor`
/// (`convertType`). A handler that fails calls `fail()`.
class QuakeBridgeVisitor
    : public ASTResultVisitor<QuakeBridgeVisitor, BridgeResult> {

public:
  explicit QuakeBridgeVisitor(
      clang::ASTContext *astCtx, mlir::MLIRContext *mlirCtx,
      mlir::OpBuilder &bldr, mlir::ModuleOp module, SymbolTable &symTab,
      EmittedFunctionsCollection &funcsToEmit,
      llvm::ArrayRef<clang::Decl *> reachableFuncs,
      MangledKernelNamesMap &namesMap, clang::CompilerInstance &ci,
      clang::ItaniumMangleContext *mangler,
      std::unordered_map<std::string, std::string> &customOperations,
      llvm::BumpPtrAllocator &alloc, bool tuplesAreReversed)
      : astContext(astCtx), mlirContext(mlirCtx), builder(bldr), module(module),
        symbolTable(symTab), functionsToEmit(funcsToEmit),
        reachableFunctions(reachableFuncs), namesMap(namesMap),
        compilerInstance(ci), mangler(mangler),
        customOperationNames(customOperations), allocator(alloc),
        typeVisitor(astCtx, bldr, mangler, tuplesAreReversed),
        tuplesAreReversed(tuplesAreReversed) {}

  /// `nvq++` renames quantum kernels to differentiate them from classical C++
  /// code. This renaming is done on function names. \p tag makes it easier
  /// to identify the kernel class from which the function was extracted.
  std::string generateCudaqKernelName(const clang::FunctionDecl *func) {
    return getCudaqKernelName(
        cudaq::detail::getTagNameOfFunctionDecl(func, mangler));
  }
  std::string generateCudaqKernelName(const EmittedFunctionPair &emittedFunc) {
    if (emittedFunc.first.starts_with(runtime::cudaqGenPrefixName))
      return emittedFunc.first;
    return generateCudaqKernelName(emittedFunc.second);
  }

  //===--------------------------------------------------------------------===//
  // Decl nodes to lower to Quake.
  //===--------------------------------------------------------------------===//

  /// Declarations are visited by the handlers below, which are found by the
  /// `ASTResultVisitor`. The members of a declaration that does not have a
  /// handler are visited, and that is all.
  bool traverseDecl(clang::Decl *x) {
    traverse(x);
    return !hasFailed();
  }

  /// Traverse a declaration that has a value, and get it. Fails if there is no
  /// value.
  std::optional<mlir::Value> traverseValue(clang::Decl *x) {
    return valueOf(traverse(x));
  }

  /// FunctionDecl: use a custom traversal for function declarations. This
  /// is also the traversal of every subclass of FunctionDecl (methods,
  /// constructors, ...).
  Result visit(clang::FunctionDecl *x);
  /// Create a constant that is a reference to the function \p x.
  Result referenceFunction(clang::FunctionDecl *x);

  // Do not traverse unresolved template declarations.
  Result visit(clang::FunctionTemplateDecl *) { return std::nullopt; }

  // VarDecl: the type of the variable is visited, and so is the initializer, if
  // there is one.
  Result visit(clang::VarDecl *x);
  /// Declare the variable \p x, which has the type \p type. If the variable has
  /// an initializer, \p init is the value of it.
  Result declareVariable(clang::VarDecl *x, mlir::Type type,
                         std::optional<mlir::Value> init);
  /// Lower the declaration of \p x. \p declaredType is the type of the
  /// variable, if it was converted.
  Result lowerVariable(clang::VarDecl *x,
                       std::optional<mlir::Type> &declaredType);
  /// A variable that could not be declared is entered in the symbol table
  /// anyway, as a poison value. An error was reported for the declaration, and
  /// the references to the variable are not errors too.
  void poisonVariable(clang::VarDecl *x,
                      std::optional<mlir::Type> declaredType);
  // The subclasses of VarDecl are not visited by the VarDecl handler.
  Result visit(clang::ParmVarDecl *x);
  Result visit(clang::ImplicitParamDecl *x) { return defaultVisit(x); }
  Result visit(clang::DecompositionDecl *x) { return defaultVisit(x); }
  Result visit(clang::VarTemplateSpecializationDecl *x) {
    return defaultVisit(x);
  }
  Result visit(clang::OMPCapturedExprDecl *x) { return defaultVisit(x); }
  /// A named declaration that is not otherwise handled is a reference to the
  /// symbol of that name. Its members are visited first.
  template <typename X>
    requires(std::is_base_of_v<clang::NamedDecl, X> &&
             !std::is_base_of_v<clang::FunctionDecl, X> &&
             !std::is_base_of_v<clang::VarDecl, X> &&
             !std::is_same_v<X, clang::FunctionTemplateDecl>)
  Result visit(X *x) {
    defaultVisit(x);
    if (hasFailed())
      return std::nullopt;
    return referenceSymbol(x);
  }
  Result referenceSymbol(clang::NamedDecl *x);

  //===--------------------------------------------------------------------===//
  // Stmt nodes to lower to Quake.
  //===--------------------------------------------------------------------===//

  /// Statements are visited by the handlers below, which are found by the
  /// `ASTResultVisitor`.
  /// Traverse a statement, or an expression whose value is not needed. Returns
  /// false if that failed.
  bool traverseStmt(clang::Stmt *x) {
    traverse(x);
    return !hasFailed();
  }

  /// Visit a statement that does not have a handler. The nodes that only wrap
  /// an expression or a statement (parentheses, temporaries, cleanups, ...) are
  /// transparent: the operands are visited, and the value of the last operand
  /// that has one is passed along. Every other node is not supported (yet),
  /// whether it never was or is a new node of clang's AST, and that is an
  /// error. It is not ignored, which would give a kernel that is not the one
  /// that was written.
  template <typename X>
    requires(std::is_base_of_v<clang::Stmt, X>)
  Result unhandled(X *x) {
    if constexpr (!isTransparentNode<X>) {
      reportUnsupportedNode(x);
      return std::nullopt;
    } else {
      Children kids;
      traverseChildren(x, kids);
      if (hasFailed())
        return std::nullopt;
      for (auto i = kids.size(); i > 0; --i)
        if (auto &kid = kids[i - 1])
          if (auto *v = std::get_if<mlir::Value>(&*kid))
            return value(*v);
      return std::nullopt;
    }
  }

  /// Report that the kind of node \p x is not supported in a kernel (yet).
  void reportUnsupportedNode(clang::Stmt *x);

  /// A result that is a value.
  static Result value(mlir::Value v) { return BridgeResult{v}; }

  /// Traverse an expression that must have a value, and get it. Fails if there
  /// is no value.
  std::optional<mlir::Value> traverseValue(clang::Stmt *x) {
    return valueOf(traverse(x));
  }

  /// The value that is the result of a visit. Fails if there is no value.
  std::optional<mlir::Value> valueOf(const Result &result) {
    if (hasFailed())
      return std::nullopt;
    if (result)
      if (auto *v = std::get_if<mlir::Value>(&*result))
        return *v;
    fail();
    return std::nullopt;
  }

  /// The values of all of the children. Fails if a child has no value.
  std::optional<llvm::SmallVector<mlir::Value>>
  childValues(const Children &kids) {
    llvm::SmallVector<mlir::Value> values;
    for (auto &kid : kids) {
      auto *v = kid ? std::get_if<mlir::Value>(&*kid) : nullptr;
      if (!v) {
        fail();
        return std::nullopt;
      }
      values.push_back(*v);
    }
    return values;
  }

  /// Statements have no value (and so no result), but visiting one can fail.
  Result finish(bool ok) {
    if (!ok)
      return fail();
    return std::nullopt;
  }

  Result visit(clang::BreakStmt *x);
  Result visit(clang::ContinueStmt *x);
  Result visit(clang::DeclStmt *x);
  Result visit(clang::CompoundStmt *x);
  Result lowerCompound(clang::CompoundStmt *x, bool atomicRegion);
  Result visit(clang::AttributedStmt *x);
  Result visit(clang::CompoundAssignOperator *x);
  Result visit(clang::ReturnStmt *x);

  template <bool postCondition, typename S>
  bool traverseDoOrWhileStmt(S *x);
  Result visit(clang::DoStmt *x);
  Result visit(clang::WhileStmt *x);
  Result visit(clang::ForStmt *x);
  Result visit(clang::IfStmt *x);

  Result visit(clang::ConditionalOperator *x);

  // These misc. statements are not (yet) handled by lowering.
  Result visit(clang::AsmStmt *x);
  Result visit(clang::CXXCatchStmt *x);
  Result visit(clang::CXXForRangeStmt *x);
  Result visit(clang::CXXTryStmt *x);
  Result visit(clang::CapturedStmt *x);
  Result visit(clang::CoreturnStmt *x);
  Result visit(clang::CoroutineBodyStmt *x);
  Result visit(clang::GotoStmt *x);
  Result visit(clang::IndirectGotoStmt *x);
  Result visit(clang::SwitchStmt *x);

  //===--------------------------------------------------------------------===//
  // Expr nodes to lower to Quake.
  //===--------------------------------------------------------------------===//

  Result visit(clang::ArraySubscriptExpr *x, Children &kids);
  Result visit(clang::BinaryOperator *x);
  /// Visit the operands of a node, and collect the values that they compute.
  template <typename Range>
  bool traverseOperands(Range &&operands, OperandStack &stack) {
    llvm::SmallVector<mlir::Value> values;
    for (auto *operand : operands) {
      Result result = traverse(operand);
      if (hasFailed())
        return false;
      // An operand that has no value (such as a default argument that is not
      // visited) contributes nothing.
      if (result)
        if (auto *v = std::get_if<mlir::Value>(&*result))
          values.push_back(*v);
    }
    stack = OperandStack(std::move(values));
    return true;
  }

  /// The result of lowering a node: the value that is on top of \p stack, if
  /// there is one. \p ok is false if lowering failed.
  Result finishOperands(bool ok, OperandStack &stack) {
    if (!ok)
      return fail();
    if (stack.empty())
      return std::nullopt;
    return value(stack.peek());
  }

  // Calls: of functions, member functions, and operators. The nodes are lowered
  // with their operands, the callee and the arguments (the values of the
  // visits of the children), in an `OperandStack`.
  Result visit(clang::CallExpr *x);
  bool lowerCall(clang::CallExpr *x, OperandStack &stack);
  bool visitMathLibFunc(clang::CallExpr *x, clang::FunctionDecl *func,
                        mlir::Location loc, llvm::StringRef funcName,
                        OperandStack &stack);
  Result visit(clang::CXXOperatorCallExpr *x);
  bool lowerOperatorCall(clang::CXXOperatorCallExpr *x, OperandStack &stack);
  /// Check that the value on the top of the stack is an entry-point kernel.
  bool hasTOSEntryKernel(OperandStack &stack);

  // Constructors, including the constructors of temporary objects.
  Result visit(clang::CXXConstructExpr *x);
  bool lowerConstruct(clang::CXXConstructExpr *x, OperandStack &stack,
                      mlir::Type ctorTy);
  Result visit(clang::CXXParenListInitExpr *x);
  bool lowerParenListInit(clang::CXXParenListInitExpr *x, OperandStack &stack,
                          mlir::Type ty);
  Result visit(clang::DeclRefExpr *x);
  Result visit(clang::FloatingLiteral *x);
  Result visit(clang::ImaginaryLiteral *x, Children &kids);

  // Cast operations. All of the casts: implicit and explicit, of every kind.
  Result visit(clang::CastExpr *x);
  /// Lower the cast \p x, whose operand was visited, to the type \p castToTy.
  Result lowerCast(clang::CastExpr *x, Children &kids, mlir::Type castToTy);

  Result visit(clang::InitListExpr *x);
  bool lowerInitList(clang::InitListExpr *x, OperandStack &stack,
                     mlir::Type initListTy);
  Result visit(clang::IntegerLiteral *x);
  Result visit(clang::CharacterLiteral *x);
  Result visit(clang::CXXBoolLiteralExpr *x);
  Result visit(clang::MaterializeTemporaryExpr *x);
  Result visit(clang::UnaryOperator *x, Children &kids);
  Result visit(clang::StringLiteral *x);
  Result visit(clang::CXXScalarValueInitExpr *x);
  Result visit(clang::UnaryExprOrTypeTraitExpr *x);

  Result visit(clang::CXXDefaultArgExpr *x);

  Result visit(clang::MemberExpr *x);
  Result visit(clang::LambdaExpr *x);

  //===--------------------------------------------------------------------===//
  // Type nodes to lower to Quake.
  //===--------------------------------------------------------------------===//

  /// Convert the type \p t to an MLIR type with the `QuakeTypeVisitor`. There
  /// is no result if there was an error (and then this visitor has failed), or
  /// if the type has no MLIR equivalent.
  std::optional<mlir::Type> convertType(clang::QualType t) {
    auto result = typeVisitor.traverse(t);
    if (typeVisitor.hasFailed()) {
      typeVisitor.clearFailure();
      fail();
      return std::nullopt;
    }
    return result;
  }

  /// Convert \p t, a builtin type, to the corresponding MLIR type.
  mlir::Type builtinTypeToType(const clang::BuiltinType *t) {
    return typeVisitor.builtinTypeToType(t);
  }

  bool shouldVisitImplicitCode() { return visitImplicitCode; }

  //===--------------------------------------------------------------------===//
  // Misc.
  //===--------------------------------------------------------------------===//

  void maybeAddCallOperationSignature(clang::Decl *x);

  /// Coerce an integer value, \p srcVal, to be the same width as \p dstTy.
  mlir::Value integerCoercion(mlir::Location loc,
                              const clang::QualType &clangTy, mlir::Type dstTy,
                              mlir::Value srcVal);

  /// Coerce an float value, \p value, to be the same width as \p toTypey.
  mlir::Value floatingPointCoercion(mlir::Location loc, mlir::Type toType,
                                    mlir::Value value);

  mlir::SmallVector<mlir::Value>
  convertKernelArgs(mlir::Location loc, std::size_t dropFrontNum,
                    const mlir::SmallVector<mlir::Value> &args,
                    mlir::ArrayRef<mlir::Type> kernelArgTys,
                    clang::CallExpr *x);

  /// Load the value referenced by an addressable value, if \p val is an address
  /// type. Otherwise, just returns \p val.
  mlir::Value loadLValue(mlir::Value val) {
    auto valTy = val.getType();
    if (isa<cudaq::cc::PointerType>(valTy))
      return cudaq::cc::LoadOp::create(builder, val.getLoc(), val);
    if (isa<mlir::LLVM::LLVMPointerType>(valTy))
      return mlir::LLVM::LoadOp::create(builder, val.getLoc(),
                                        builder.getI8Type(), val);
    return val;
  }

  // Does the block have a proper terminator?
  static bool hasTerminator(mlir::Block &block);
  static bool hasTerminator(mlir::Block *block) {
    return hasTerminator(*block);
  }

  /// Used to set the name of a kernel entry function.
  void setEntryName(llvm::StringRef name) {
    loweredFuncName = name.str();
    isEntry = true;
  }

  /// Used to set the name of any function that is not a kernel entry.
  void setCurrentFunctionName(llvm::StringRef name) {
    loweredFuncName = name.str();
    isEntry = false;
  }

  /// Generate the C++ mangled name for declaration, \p decl.
  std::string cxxMangledDeclName(clang::GlobalDecl decl) {
    return getCxxMangledDeclName(decl, mangler);
  }

  /// Generate the C++ mangled name for a type, \p ty.
  std::string cxxMangledTypeName(clang::QualType ty) {
    return getCxxMangledTypeName(ty, mangler);
  }

  /// Generate a function declaration in the module.
  bool generateFunctionDeclaration(mlir::StringRef funcName,
                                   const clang::FunctionDecl *x);
  bool doSyntaxChecks(const clang::FunctionDecl *x, mlir::FunctionType funcTy);

  bool isItaniumCXXABI();

private:
  /// Map the block arguments to the names of the function parameters.
  void addArgumentSymbols(mlir::Block *entryBlock,
                          mlir::ArrayRef<clang::ParmVarDecl *> parameters);

  /// Get the current function's name.
  std::string getCurrentFunctionName() { return loweredFuncName; }

  /// Clear the current function name.
  void resetCurrentFunctionName() { loweredFuncName.clear(); }

  /// Returns true if \p decl is a kernel entry point.
  bool isKernelEntryPoint(const clang::FunctionDecl *decl);

  /// Returns true if \p decl is a function to lower to Quake.
  bool needToLowerFunction(const clang::FunctionDecl *decl);

  /// Helpers to convert an AST node's clang source range to an MLIR Location.
  template <typename A>
  mlir::Location toLocation(const A *x) {
    return toLocation(x->getSourceRange());
  }

  mlir::Location toLocation(const clang::SourceRange &srcRange) {
    return toSourceLocation(getMLIRContext(), getContext(), srcRange);
  }

  /// Add an entry block to FuncOp \p func corresponding to the AST FunctionDecl
  /// \p x.
  void createEntryBlock(mlir::func::FuncOp func, const clang::FunctionDecl *x);

  /// Returns the type name of an intercepted `operator[]` to the caller. If the
  /// `operator[]` is not being intercepted, then returns `std::nullopt`.
  std::optional<std::string>
  isInterceptedSubscriptOperator(clang::CXXOperatorCallExpr *x);

  static mlir::FunctionType peelPointerFromFunction(mlir::Type ty);

  mlir::MLIRContext *getMLIRContext() { return mlirContext; }

  /// Get the ASTContext.
  clang::ASTContext *getContext() const { return astContext; }

  /// Calls should be to C++ mangled names unless this is a known entry point.
  /// In the latter case, use the entry point name.
  std::string genLoweredName(clang::FunctionDecl *x, mlir::FunctionType funcTy);

  /// Return a FuncOp for the specified function, given a name and signature. If
  /// the function already exists and is defined (has a body), then the the
  /// second member of the returned pair will be `true`.
  std::pair<mlir::func::FuncOp, bool> getOrAddFunc(mlir::Location loc,
                                                   mlir::StringRef funcName,
                                                   mlir::FunctionType funcTy);

  /// Definite-assignment check for `for (bool b : v)` where `v` is a
  /// `std::vector<measure_handle>`. Returns false only when it can prove the
  /// vector was never bound to a measurement, and true for every shape it
  /// cannot disprove. See: https://github.com/NVIDIA/cuda-quantum/issues/4479.
  bool isBoundHandleVector(mlir::Value, llvm::SmallPtrSetImpl<mlir::Value> &);

  /// Stack of the innermost enclosing loop's loop-carried arguments. `break`/
  /// `continue` need these current values to build a `cc.unwind_break`/
  /// `cc.unwind_continue` with the arity the enclosing `cc.loop` requires.
  llvm::SmallVector<mlir::ValueRange, 4> loopArgsStack;

  /// RAII helper to push/pop `loopArgsStack` around the construction of a
  /// loop body.
  struct LoopArgsScope {
    LoopArgsScope(QuakeBridgeVisitor &visitor, mlir::ValueRange args)
        : visitor(visitor) {
      visitor.loopArgsStack.push_back(args);
    }
    ~LoopArgsScope() { visitor.loopArgsStack.pop_back(); }
    QuakeBridgeVisitor &visitor;
  };

  /// The current loop-carried arguments for the nearest enclosing loop, to be
  /// forwarded as operands to a `cc.unwind_break`/`cc.unwind_continue`.
  mlir::ValueRange currentLoopArgs() {
    return loopArgsStack.empty() ? mlir::ValueRange{} : loopArgsStack.back();
  }

  clang::ASTContext *astContext;
  mlir::MLIRContext *mlirContext;
  mlir::OpBuilder &builder;
  mlir::ModuleOp module;
  SymbolTable &symbolTable;
  EmittedFunctionsCollection &functionsToEmit;
  llvm::ArrayRef<clang::Decl *> reachableFunctions;
  MangledKernelNamesMap &namesMap;
  clang::CompilerInstance &compilerInstance;
  /// Lowered name of the function. Entry points have their names changed.
  clang::ItaniumMangleContext *mangler;
  std::string loweredFuncName;
  llvm::SmallVector<mlir::Value> negations;
  std::unordered_map<std::string, std::string> &customOperationNames;
  /// Allocator for dynamically generated symbol names, referenced by the symbol
  /// table.
  llvm::BumpPtrAllocator &allocator;

  //===--------------------------------------------------------------------===//
  // Type conversion
  //===--------------------------------------------------------------------===//

  /// Converts clang types to MLIR types.
  QuakeTypeVisitor typeVisitor;

  // State Flags
  const bool tuplesAreReversed : 1;
  bool skipCompoundScope : 1 = false;
  bool isEntry : 1 = false;
  /// If there is a catastrophic error in the bridge (there is no rational way
  /// to proceed to emit correct code), emit an error using the diagnostic
  /// engine, set this flag, and return false.
  bool raisedError : 1 = false;
  bool visitImplicitCode : 1 = false;
  bool initializerIsGlobal : 1 = false;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// ASTBridgeAction
//===----------------------------------------------------------------------===//

/// The ASTBridgeAction enables the insertion of a custom ASTConsumer to the
/// Clang AST analysis / processing workflow. The nested ASTBridgeConsumer
/// drives the process of walking the Clang AST and translate pertinent nodes to
/// an MLIR Op tree containing Quake, CC, and other MLIR dialect operations.
/// In short, this Action generates the MLIR Module.
class ASTBridgeAction : public clang::ASTFrontendAction {
public:
  using MangledKernelNamesMap = cudaq::MangledKernelNamesMap;

  /// Options controlling emission of a Makefile-syntax dependency file (the
  /// GNU-style -MD/-MMD/-MT/-MF information). nvq++ lowers __qpu__ kernels
  /// through cudaq-quake instead of a plain clang -c, so this action -- the one
  /// place that runs a preprocessor over the input for both the MLIR-only and
  /// the LLVM-IR (CudaQAction-composed) paths -- is where header dependencies
  /// are recorded. See attachDependencyFileGenerator().
  struct DependencyFileOptions {
    /// Path to write the dependency file to (the -MF value). When empty,
    /// dependency-file generation is disabled.
    std::string outputFile;
    /// Target name(s) for the emitted rule (the -MT values).
    std::vector<std::string> targets;
    /// Whether to include system headers (-MD) or omit them (-MMD).
    bool includeSystemHeaders = false;
    /// Canonical on-disk path of the main input file. clang::tooling maps the
    /// source to a virtual file named after the (bare) input spelling, so the
    /// dependency generator would otherwise record the main file under a
    /// non-existent relative name. When set, that entry is rewritten to this
    /// path so the emitted rule's prerequisite matches the real source.
    std::string mainFileRealPath;
  };

  /// Constructor. \p depOpts is stored by reference and so must outlive this
  /// action (in cudaq-quake it is a local in main() that outlives the
  /// synchronous tool run); pass a default-constructed value to disable
  /// dependency-file generation.
  ASTBridgeAction(mlir::OwningOpRef<mlir::ModuleOp> &_module,
                  MangledKernelNamesMap &cxx_mangled,
                  const DependencyFileOptions &depOpts)
      : dependencyFileOptions(depOpts), module(_module),
        cxx_mangled_kernel_names(cxx_mangled) {}

  /// Instantiate the ASTBridgeConsumer for this ASTFrontendAction.
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    llvm::StringRef inFile) override {
    // Collect header dependencies during the preprocessor pass this consumer
    // is about to drive. This runs for both the standalone MLIR path and the
    // LLVM-IR path (where CudaQAction forwards to this same CreateASTConsumer).
    attachDependencyFileGenerator(compiler);
    return std::make_unique<ASTBridgeConsumer>(compiler, module,
                                               cxx_mangled_kernel_names);
  }

  //===--------------------------------------------------------------------===//
  // ASTBridgeConsumer - inner class
  //===--------------------------------------------------------------------===//
  class ASTBridgeConsumer : public clang::ASTConsumer {
    using MangledKernelNamesMap = ASTBridgeAction::MangledKernelNamesMap;

  protected:
    // The Clang AST Context
    clang::ASTContext &astContext;
    clang::CompilerInstance &ci;
    MangledKernelNamesMap &cxx_mangled_kernel_names;

    // The MLIR Module we are building up
    mlir::OwningOpRef<mlir::ModuleOp> &module;

    // Observed quantum functions, we will iterate through these in buildMLIR()
    EmittedFunctionsCollection functionsToEmit;

    // The functions that are not known to be kernels (or intrinsics) when they
    // are found, and that take a quantum type. A function can be declared
    // before the declaration that makes it a kernel is seen, so these are only
    // checked at the end of the translation unit.
    std::vector<const clang::FunctionDecl *> deferredParameterChecks;
    clang::CallGraph callGraphBuilder;

    // The builder instance used to create MLIR nodes
    mlir::OpBuilder builder;

    // The symbol table, holding MLIR values keyed on variable name.
    SymbolTable symbol_table;

    /// Allocator for dynamically generated symbol names, referenced by the
    /// symbol table.
    llvm::BumpPtrAllocator allocator;

    // The mangler is constructed and owned by `this`.
    clang::ItaniumMangleContext *mangler;

    // Keep track of user custom operation names.
    std::unordered_map<std::string, std::string> customOperationNames;

    bool tuplesAreReversed = false;

    /// Add a placeholder definition to the module in \p visitor for the
    /// function, \p funcDecl. This is used for adding the host-side function
    /// corresponding to the kernel. The code for this function will be
    /// automatically generated by the GenKernelExecution pass. \p funcTy is the
    /// type of \p funcDecl. \p devFuncName is the name of the device-side
    /// kernel. The placeholder definition lets any argument attributes be
    /// properly communicated through the pass pipeline and prevents lossy
    /// pipelines which erase private declarations.
    void addFunctionDecl(const clang::FunctionDecl *funcDecl,
                         detail::QuakeBridgeVisitor &visitor,
                         mlir::FunctionType funcTy, mlir::StringRef devFuncName,
                         bool isDecl);

  public:
    ASTBridgeConsumer(clang::CompilerInstance &compiler,
                      mlir::OwningOpRef<mlir::ModuleOp> &_module,
                      MangledKernelNamesMap &cxx_mangled);

    // This gets called after HandleTopLevelDecl, we have the quantum kernel
    // FunctionDecls, emit the MLIR code for each
    void HandleTranslationUnit(clang::ASTContext &Context) override;

    // Find all FunctionDecls that are quantum kernels
    bool HandleTopLevelDecl(clang::DeclGroupRef dg) override;

    // Clean up the symbol name pointers.
    virtual ~ASTBridgeConsumer() { delete mangler; }

    // Return true if this FunctionDecl is a quantum kernel.
    static bool isQuantum(const clang::FunctionDecl *decl);

    // Return true if this FunctionDecl is a generator function for custom
    // operation
    static bool isCustomOpGenerator(const clang::FunctionDecl *decl);
  };

private:
  /// Attach clang's DependencyFileGenerator to \p ci's preprocessor when
  /// dependencyFileOptions is non-empty. The generator writes the dependency
  /// file when the preprocessor reaches the end of the main file; no extra
  /// object is emitted.
  void attachDependencyFileGenerator(clang::CompilerInstance &ci);

  const DependencyFileOptions &dependencyFileOptions;

protected:
  // The MLIR Module we are building up
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  MangledKernelNamesMap &cxx_mangled_kernel_names;
};

/// Return true if and only if \p x was declared at the top-level.
inline bool isNotInANamespace(const clang::Decl *x) {
  assert(x && "decl is null");
  auto *declCtx = x->getDeclContext();
  do {
    if (isa<clang::NamespaceDecl>(declCtx))
      return false;
    declCtx = declCtx->getParent();
  } while (declCtx);
  return true;
}

/// Return true if and only if \p x was declared in the namespace \p nsName.
/// This test will "drill through" any nested namespaces in search of a match.
inline bool isInNamespace(const clang::Decl *x, mlir::StringRef nsName) {
  assert(x && "decl is null");
  auto *declCtx = x->getDeclContext();
  do {
    if (const auto *nsd = dyn_cast<clang::NamespaceDecl>(declCtx))
      if (const auto *nsi = nsd->getIdentifier())
        if (nsi->getName() == nsName)
          return true;
    declCtx = declCtx->getParent();
  } while (declCtx);
  return false;
}

/// Return true if and only if \p x was declared in the class \p className and
/// that class was furthermore declared in the namespace \p nsName.
inline bool isInClassInNamespace(const clang::Decl *x,
                                 mlir::StringRef className,
                                 mlir::StringRef nsName) {
  assert(x && "decl is null");
  if (const auto *cld = dyn_cast<clang::RecordDecl>(x->getDeclContext()))
    if (const auto *cli = cld->getIdentifier())
      return (cli->getName() == className) && isInNamespace(cld, nsName);
  return false;
}

bool isInExternC(const clang::GlobalDecl &x);

/// Is \p kindValue the `operator()` function?
inline bool isCallOperator(clang::OverloadedOperatorKind kindValue) {
  return kindValue == clang::OverloadedOperatorKind::OO_Call;
}

/// Is \p t of type `char *`?
inline bool isCharPointerType(mlir::Type t) {
  if (auto ptrTy = dyn_cast<cc::PointerType>(t)) {
    mlir::Type eleTy = ptrTy.getElementType();
    if (auto arrTy = dyn_cast<cc::ArrayType>(eleTy))
      eleTy = arrTy.getElementType();
    if (auto intTy = dyn_cast<mlir::IntegerType>(eleTy))
      return intTy.getWidth() == 8;
  }
  return false;
}

/// Is \p t a `char` span type? The type `pauli_word` maps to a span of `char`.
inline bool isCharspanPointerType(mlir::Type t) {
  if (auto ptrTy = dyn_cast<cc::PointerType>(t)) {
    mlir::Type eleTy = ptrTy.getElementType();
    return isa<cc::CharspanType>(eleTy);
  }
  return false;
}

} // namespace cudaq
