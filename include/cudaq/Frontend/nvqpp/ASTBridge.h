/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/InitAllDialects.h"

namespace cudaq::details {
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
} // namespace cudaq::details

#undef TODO_BRIDGE
#undef TODO_x

#if defined(NDEBUG) || defined(CUDAQ_NOTRACEBACKS)
#define TODO_BRIDGE(MlirLoc, ToDoXPtr, ToDoMangler, ToDoMsg, ToDoFile,         \
                    ToDoLine)                                                  \
  cudaq::details::reportClangError(ToDoXPtr, ToDoMangler,                      \
                                   ToDoMsg " is not yet supported");
#else
#define TODO_BRIDGE(MlirLoc, ToDoXPtr, ToDoMangler, ToDoMsg, ToDoFile,         \
                    ToDoLine)                                                  \
  do {                                                                         \
    mlir::emitError(MlirLoc, llvm::Twine(ToDoFile ":" TODOQUOTE(               \
                                 ToDoLine) ": not yet implemented: ") +        \
                                 ToDoMsg);                                     \
    cudaq::details::reportClangError(ToDoXPtr, ToDoMangler,                    \
                                     ToDoMsg " is not yet supported");         \
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

namespace details {

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

/// Is this a class that will be ignored by the bridge?
/// FIXME: This is a bit of a hack to skip over certain AST nodes.
bool ignoredClass(clang::RecordDecl *x);

//===----------------------------------------------------------------------===//
// QuakeBridgeVisitor
//===----------------------------------------------------------------------===//

/// QuakeBridgeVisitor is a visitor pattern for crawling over the AST and
/// generating Quake, CC, and other MLIR dialects.
///
/// The general design is to walk the tree in a post-order traversal and
/// assemble the IR from the leaves back down the tree. Traversals over types
/// should push Type values to the type stack. Traversals over expressions
/// should create IR in the ModuleOp as well as push subexpressions on the
/// stack for parent nodes. A parent node always knows how many children it
/// needs to be constructed correctly. The types of expressions are carried
/// along with the expressions in the IR and need not be duplicated on the type
/// stack.
///
/// Unfortunately, clang's RecursiveASTVisitor doesn't always visit nodes in the
/// AST and can skip visiting types or even some expressions.
class QuakeBridgeVisitor
    : public clang::RecursiveASTVisitor<QuakeBridgeVisitor> {
  using Base = clang::RecursiveASTVisitor<QuakeBridgeVisitor>;

public:
  explicit QuakeBridgeVisitor(clang::ASTContext *astCtx,
                              mlir::MLIRContext *mlirCtx, mlir::OpBuilder &bldr,
                              mlir::ModuleOp module, SymbolTable &symTab,
                              EmittedFunctionsCollection &funcsToEmit,
                              llvm::ArrayRef<clang::Decl *> reachableFuncs,
                              MangledKernelNamesMap &namesMap,
                              clang::CompilerInstance &ci,
                              clang::ItaniumMangleContext *mangler)
      : astContext(astCtx), mlirContext(mlirCtx), builder(bldr), module(module),
        symbolTable(symTab), functionsToEmit(funcsToEmit),
        reachableFunctions(reachableFuncs), namesMap(namesMap),
        compilerInstance(ci), mangler(mangler) {}

  /// `nvq++` renames quantum kernels to differentiate them from classical C++
  /// code. This renaming is done on function names. \p tag makes it easier
  /// to identify the kernel class from which the function was extracted.
  std::string generateCudaqKernelName(const clang::FunctionDecl *func) {
    return getCudaqKernelName(
        cudaq::details::getTagNameOfFunctionDecl(func, mangler));
  }
  std::string generateCudaqKernelName(const EmittedFunctionPair &emittedFunc) {
    if (emittedFunc.first.starts_with(runtime::cudaqGenPrefixName))
      return emittedFunc.first;
    return generateCudaqKernelName(emittedFunc.second);
  }

  //===--------------------------------------------------------------------===//
  // Decl nodes to lower to Quake.
  //===--------------------------------------------------------------------===//

  // FunctionDecl: use a custom traversal for function declarations and all
  // subtypes of FunctionDecl.
  bool TraverseFunctionDecl(clang::FunctionDecl *x);
  bool WalkUpFromFunctionDecl(clang::FunctionDecl *x) {
    // Do not walk up to the super classes of FunctionDecl.
    return VisitFunctionDecl(x);
  }
  bool VisitFunctionDecl(clang::FunctionDecl *x);
  bool TraverseCXXDeductionGuideDecl(clang::CXXDeductionGuideDecl *x) {
    if (inRecType)
      return true;
    return TraverseFunctionDecl(x);
  }
  bool TraverseCXXMethodDecl(clang::CXXMethodDecl *x) {
    if (inRecType)
      return true;
    return TraverseFunctionDecl(x);
  }
  bool TraverseCXXConstructorDecl(clang::CXXConstructorDecl *x) {
    return TraverseCXXMethodDecl(x);
  }
  bool TraverseCXXConversionDecl(clang::CXXConversionDecl *x) {
    return TraverseCXXMethodDecl(x);
  }
  bool TraverseCXXDestructorDecl(clang::CXXDestructorDecl *x) {
    return TraverseCXXMethodDecl(x);
  }

  bool TraverseFunctionTemplateDecl(clang::FunctionTemplateDecl *x) {
    // Do not traverse unresolved template declarations.
    return true;
  }

  bool VisitNamedDecl(clang::NamedDecl *x);

  // ParmVarDecl
  bool WalkUpFromParmVarDecl(clang::ParmVarDecl *x) {
    // Prevent walking up to base classes.
    return VisitParmVarDecl(x);
  }
  bool VisitParmVarDecl(clang::ParmVarDecl *x);

  // VarDecl
  bool TraverseVarDecl(clang::VarDecl *x);
  bool WalkUpFromVarDecl(clang::VarDecl *x) {
    // Prevent walking up to base classes.
    return VisitVarDecl(x);
  }
  bool VisitVarDecl(clang::VarDecl *x);

  //===--------------------------------------------------------------------===//
  // Stmt nodes to lower to Quake.
  //===--------------------------------------------------------------------===//

  bool VisitBreakStmt(clang::BreakStmt *x);
  bool TraverseCompoundStmt(clang::CompoundStmt *x,
                            DataRecursionQueue *q = nullptr);

  bool WalkUpFromCompoundAssignOperator(clang::CompoundAssignOperator *x) {
    return VisitCompoundAssignOperator(x);
  }

  bool VisitCompoundAssignOperator(clang::CompoundAssignOperator *x);
  bool VisitContinueStmt(clang::ContinueStmt *x);

  template <bool postCondition, typename S>
  bool traverseDoOrWhileStmt(S *x);
  bool TraverseDoStmt(clang::DoStmt *x, DataRecursionQueue *q = nullptr);
  bool TraverseWhileStmt(clang::WhileStmt *x, DataRecursionQueue *q = nullptr);

  bool TraverseForStmt(clang::ForStmt *x, DataRecursionQueue *q = nullptr);
  bool TraverseIfStmt(clang::IfStmt *x, DataRecursionQueue *q = nullptr);
  bool TraverseConditionalOperator(clang::ConditionalOperator *x,
                                   DataRecursionQueue *q = nullptr);
  bool VisitReturnStmt(clang::ReturnStmt *x);
  bool TraverseInitListExpr(clang::InitListExpr *x,
                            DataRecursionQueue *q = nullptr);

  // These misc. statements are not (yet) handled by lowering.
  bool TraverseAsmStmt(clang::AsmStmt *x, DataRecursionQueue *q = nullptr);
  bool TraverseCXXCatchStmt(clang::CXXCatchStmt *x,
                            DataRecursionQueue *q = nullptr);
  bool TraverseCXXForRangeStmt(clang::CXXForRangeStmt *x,
                               DataRecursionQueue *q = nullptr);
  bool TraverseCXXTryStmt(clang::CXXTryStmt *x,
                          DataRecursionQueue *q = nullptr);
  bool TraverseCapturedStmt(clang::CapturedStmt *x,
                            DataRecursionQueue *q = nullptr);
  bool TraverseCoreturnStmt(clang::CoreturnStmt *x,
                            DataRecursionQueue *q = nullptr);
  bool TraverseCoroutineBodyStmt(clang::CoroutineBodyStmt *x,
                                 DataRecursionQueue *q = nullptr);
  bool TraverseGotoStmt(clang::GotoStmt *x, DataRecursionQueue *q = nullptr);
  bool TraverseIndirectGotoStmt(clang::IndirectGotoStmt *x,
                                DataRecursionQueue *q = nullptr);
  bool TraverseSwitchStmt(clang::SwitchStmt *x,
                          DataRecursionQueue *q = nullptr);

  //===--------------------------------------------------------------------===//
  // Expr nodes to lower to Quake.
  //===--------------------------------------------------------------------===//

  bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *x);
  bool VisitBinaryOperator(clang::BinaryOperator *x);
  bool VisitCallExpr(clang::CallExpr *x);
  bool TraverseCXXConstructExpr(clang::CXXConstructExpr *x,
                                DataRecursionQueue *q = nullptr);
  bool VisitCXXConstructExpr(clang::CXXConstructExpr *x);
  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *x);
  bool WalkUpFromCXXOperatorCallExpr(clang::CXXOperatorCallExpr *x);
  bool TraverseDeclRefExpr(clang::DeclRefExpr *x,
                           DataRecursionQueue *q = nullptr);
  bool VisitDeclRefExpr(clang::DeclRefExpr *x);
  bool VisitFloatingLiteral(clang::FloatingLiteral *x);

  // Cast operations.
  bool TraverseCastExpr(clang::CastExpr *x, DataRecursionQueue *q = nullptr);
  bool VisitCastExpr(clang::CastExpr *x);

  bool TraverseImplicitCastExpr(clang::ImplicitCastExpr *x,
                                DataRecursionQueue *q = nullptr) {
    return TraverseCastExpr(x, q);
  }
  bool TraverseExplicitCastExpr(clang::ExplicitCastExpr *x,
                                DataRecursionQueue *q = nullptr) {
    return TraverseCastExpr(x, q);
  }
  bool TraverseCStyleCastExpr(clang::CStyleCastExpr *x,
                              DataRecursionQueue *q = nullptr) {
    return TraverseExplicitCastExpr(x, q);
  }
  bool TraverseCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *x,
                                     DataRecursionQueue *q = nullptr) {
    return TraverseExplicitCastExpr(x, q);
  }
  bool TraverseCXXAddrspaceCastExpr(clang::CXXAddrspaceCastExpr *x,
                                    DataRecursionQueue *q = nullptr) {
    return TraverseExplicitCastExpr(x, q);
  }
  bool TraverseCXXConstCastExpr(clang::CXXConstCastExpr *x,
                                DataRecursionQueue *q = nullptr) {
    return TraverseExplicitCastExpr(x, q);
  }
  bool TraverseCXXDynamicCastExpr(clang::CXXDynamicCastExpr *x,
                                  DataRecursionQueue *q = nullptr) {
    return TraverseExplicitCastExpr(x, q);
  }
  bool TraverseCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *x,
                                      DataRecursionQueue *q = nullptr) {
    return TraverseExplicitCastExpr(x, q);
  }
  bool TraverseCXXStaticCastExpr(clang::CXXStaticCastExpr *x,
                                 DataRecursionQueue *q = nullptr) {
    return TraverseExplicitCastExpr(x, q);
  }
  bool TraverseBuiltinBitCastExpr(clang::BuiltinBitCastExpr *x,
                                  DataRecursionQueue *q = nullptr) {
    return TraverseExplicitCastExpr(x, q);
  }

  bool VisitInitListExpr(clang::InitListExpr *x);
  bool VisitIntegerLiteral(clang::IntegerLiteral *x);
  bool VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *x);
  bool VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *x);
  bool VisitUnaryOperator(clang::UnaryOperator *x);
  bool VisitStringLiteral(clang::StringLiteral *x);
  bool VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *x);

  bool TraverseMemberExpr(clang::MemberExpr *x,
                          DataRecursionQueue *q = nullptr);
  bool VisitMemberExpr(clang::MemberExpr *x);
  bool TraverseBinaryOperator(clang::BinaryOperator *x,
                              DataRecursionQueue *q = nullptr);
  bool TraverseLambdaExpr(clang::LambdaExpr *x,
                          DataRecursionQueue *q = nullptr);

  //===--------------------------------------------------------------------===//
  // Type nodes to lower to Quake.
  //===--------------------------------------------------------------------===//

  bool TraverseTypedefType(clang::TypedefType *t) {
    return TraverseType(t->desugar());
  }
  bool TraverseTypedefTypeLoc(clang::TypedefTypeLoc tl) {
    return TraverseType(tl.getType());
  }
  bool TraverseUsingType(clang::UsingType *t) {
    return TraverseType(t->desugar());
  }
  bool TraverseUsingTypeLoc(clang::UsingTypeLoc tl) {
    return TraverseType(tl.getType());
  }
  bool
  TraverseTemplateSpecializationType(clang::TemplateSpecializationType *t) {
    return TraverseType(t->desugar());
  }
  bool TraverseTypeOfExprType(clang::TypeOfExprType *t) {
    // Do not visit the expression as it is has no semantics other than for
    // inferring a type.
    return TraverseType(t->desugar());
  }
  bool TraverseNestedNameSpecifier(clang::NestedNameSpecifier *) {
    return true;
  }
  bool TraverseDecltypeType(clang::DecltypeType *t) {
    return TraverseType(t->desugar());
  }

  // When processing a record type, visit the type of all the field decls. This
  // will push 1 new type on the stack for each field. These types will be the
  // member types of the StructType.
  bool TraverseFieldDecl(clang::FieldDecl *x) {
    if (inRecType)
      return TraverseType(x->getType());
    return Base::TraverseFieldDecl(x);
  }
  bool WalkUpFromFieldDecl(clang::FieldDecl *x) {
    if (inRecType)
      return true;
    return Base::WalkUpFromFieldDecl(x);
  }

  bool TraverseRecordType(clang::RecordType *t);
  bool interceptRecordDecl(clang::RecordDecl *x);
  bool VisitRecordDecl(clang::RecordDecl *x);

  // Type declarations to be converted to high-level Quake and CC types are
  // either classified as Record, CXXRecord, or ClassTemplateSpecialization
  // declarations. Other decls will be converted to cc.struct types.
  // These traversals are explicitly called, so we must overload them.
  template <typename D>
  bool traverseAnyRecordDecl(D *x);
  bool TraverseRecordDecl(clang::RecordDecl *x);
  bool TraverseCXXRecordDecl(clang::CXXRecordDecl *x);
  bool TraverseClassTemplateSpecializationDecl(
      clang::ClassTemplateSpecializationDecl *x);

  bool VisitFunctionProtoType(clang::FunctionProtoType *t);
  bool VisitBuiltinType(clang::BuiltinType *t);
  bool VisitPointerType(clang::PointerType *t);
  bool VisitLValueReferenceType(clang::LValueReferenceType *t);
  bool VisitRValueReferenceType(clang::RValueReferenceType *t);
  bool VisitConstantArrayType(clang::ConstantArrayType *t);

  /// Convert \p t, a builtin type, to the corresponding MLIR type.
  mlir::Type builtinTypeToType(const clang::BuiltinType *t);

  bool shouldVisitImplicitCode() { return visitImplicitCode; }
  bool shouldVisitTemplateInstantiations() { return inRecType; }

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

  /// Load the value referenced by an addressable value, if \p val is an address
  /// type. Otherwise, just returns \p val.
  mlir::Value loadLValue(mlir::Value val) {
    auto valTy = val.getType();
    if (valTy.isa<cudaq::cc::PointerType>())
      return builder.create<cudaq::cc::LoadOp>(val.getLoc(), val);
    if (valTy.isa<mlir::LLVM::LLVMPointerType>())
      return builder.create<mlir::LLVM::LoadOp>(val.getLoc(), val);
    return val;
  }

  // The AST is visited in postorder.
  bool shouldTraversePostOrder() { return true; }

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

  /// Reset the visitor for the next top-level function to convert.
  void resetNextTopLevelFunction() {
    valueStack.clear();
    typeStack.clear();
  }

  /// Generate a function declaration in the module.
  bool generateFunctionDeclaration(mlir::StringRef funcName,
                                   const clang::FunctionDecl *x);
  bool doSyntaxChecks(const clang::FunctionDecl *x);

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

  static clang::FunctionDecl *
  findCallOperator(const clang::CXXRecordDecl *decl);

#ifndef NDEBUG
  // Debug versions have to be in the .cpp file which pulls in the LLVM debug
  // support code header, etc.
  bool pushValue(mlir::Value v);
  mlir::Value popValue();
  mlir::SmallVector<mlir::Value> lastValues(unsigned n);
#else
  // If not a debug build, inline these methods for efficiency.
  bool pushValue(mlir::Value v) {
    valueStack.push_back(v);
    return true;
  }
  mlir::Value popValue() {
    auto result = peekValue();
    valueStack.pop_back();
    return result;
  }
  /// Return the last `n` values from the stack in left-to-right (natural)
  /// order. For a call, `foo(a, b, c)` this can be used to return a list
  /// `[value_a value_b value_c]`.
  mlir::SmallVector<mlir::Value> lastValues(unsigned n) {
    assert(n <= valueStack.size() && "stack has fewer values than requested");
    mlir::SmallVector<mlir::Value> result(valueStack.end() - n,
                                          valueStack.end());
    valueStack.pop_back_n(n);
    return result;
  }
#endif
  // Return a copy of the Value on the top of the value stack.
  mlir::Value peekValue() { return valueStack.back(); }

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

  /// Stack of Values built by the visitor. (right-to-left ordering)
  mlir::SmallVector<mlir::Value> valueStack;
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

  //===--------------------------------------------------------------------===//
  // Type traversals
  //===--------------------------------------------------------------------===//

  bool pushType(mlir::Type t);
  mlir::Type popType();
  mlir::Type peekType() { return typeStack.back(); }

  /// Return the last `n` types from the stack in left-to-right (natural)
  /// order. For a signature, `f(T, U, V)` this can be used to return a list
  /// `[type_T type_U type_V]`.
  llvm::SmallVector<mlir::Type> lastTypes(unsigned n);

  /// Stack of Types built by the visitor. (right-to-left ordering)
  llvm::SmallVector<mlir::Type> typeStack;
  llvm::DenseMap<clang::RecordType *, mlir::Type> records;

  // State Flags
  bool skipCompoundScope : 1 = false;
  bool isEntry : 1 = false;
  /// If there is a catastrophic error in the bridge (there is no rational way
  /// to proceed to emit correct code), emit an error using the diagnostic
  /// engine, set this flag, and return false.
  bool raisedError : 1 = false;
  bool visitImplicitCode : 1 = false;
  bool inRecType : 1 = false;
  bool allowUnknownRecordType : 1 = false;
  bool initializerIsGlobal : 1 = false;
};
} // namespace details

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

  /// Constructor.
  ASTBridgeAction(mlir::OwningOpRef<mlir::ModuleOp> &_module,
                  MangledKernelNamesMap &cxx_mangled)
      : module(_module), cxx_mangled_kernel_names(cxx_mangled) {}

  /// Instantiate the ASTBridgeConsumer for this ASTFrontendAction.
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    llvm::StringRef inFile) override {
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
    clang::CallGraph callGraphBuilder;

    // The builder instance used to create MLIR nodes
    mlir::OpBuilder builder;

    // The symbol table, holding MLIR values keyed on variable name.
    SymbolTable symbol_table;

    // The mangler is constructed and owned by `this`.
    clang::ItaniumMangleContext *mangler;

    /// Add a placeholder definition to the module in \p visitor for the
    /// function, \p funcDecl. This is used for adding the host-side function
    /// corresponding to the kernel. The code for this function will be
    /// automatically generated by the GenKernelExecution pass. \p funcTy is the
    /// type of \p funcDecl. \p devFuncName is the name of the device-side
    /// kernel. The placeholder definition lets any argument attributes be
    /// properly communicated through the pass pipeline and prevents lossy
    /// pipelines which erase private declarations.
    void addFunctionDecl(const clang::FunctionDecl *funcDecl,
                         details::QuakeBridgeVisitor &visitor,
                         mlir::FunctionType funcTy,
                         mlir::StringRef devFuncName);

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
  };

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
        if (nsi->getName().equals(nsName))
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
      return cli->getName().equals(className) && isInNamespace(cld, nsName);
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

inline bool isCudaqStateType(mlir::Type t) {
  if (auto strTy = dyn_cast<cc::StructType>(t))
    return strTy.getName().getValue().equals("cudaq.state");
  return false;
}

} // namespace cudaq
