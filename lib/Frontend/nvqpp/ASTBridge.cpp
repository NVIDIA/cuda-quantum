/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Frontend/nvqpp/QisBuilder.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include <cxxabi.h>
#include <fstream>
#include <regex>

#define DEBUG_TYPE "lower-ast"

using namespace mlir;

// Generate a list (as a vector) of all the reachable functions recorded in the
// call graph, \p cgn.
static llvm::SmallVector<clang::Decl *>
listReachableFunctions(clang::CallGraphNode *cgn) {
  llvm::SmallVector<clang::Decl *> result;
  for (auto call : *cgn) {
    auto *decl = call.Callee->getDecl();
    if (auto *nd = dyn_cast<clang::NamedDecl>(decl)) {
      LLVM_DEBUG(llvm::dbgs() << "adding function: "
                              << nd->getQualifiedNameAsString() << '\n');
    }
    result.push_back(decl);
  }
  return result;
}

// Does `ty` refer to a Quake quantum type? This also checks custom recursive
// types. It does not check builtin recursive types; e.g., `!llvm.ptr<T>`.
static bool isQubitType(Type ty) {
  if (ty.isa<quake::RefType, quake::VeqType>())
    return true;
  if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(ty))
    return isQubitType(vecTy.getElementType());
  return false;
}

// Check the builtin type FunctionType to see if it has any references to Quake
// qubit types in its arguments and/or results. It does not check QTX types as
// the bridge should be lowering to Quake.
static bool hasAnyQubitTypes(FunctionType funcTy) {
  for (auto ty : funcTy.getInputs())
    if (isQubitType(ty))
      return true;
  for (auto ty : funcTy.getResults())
    if (isQubitType(ty))
      return true;
  return false;
}

// Remove the Itanium mangling "_ZTS" prefix. This is to match the name returned
// by `typeid(TYPE).name()`.
static std::string
trimmedMangledTypeName(clang::QualType ty,
                       clang::ItaniumMangleContext *mangler) {
  auto s = cudaq::details::getCxxMangledTypeName(ty, mangler);
  // Strip off the prefix.
  constexpr char prefix[] = "_ZTS";
  constexpr auto prefixLength = sizeof(prefix) - 1;
  assert(s.length() > prefixLength);
  if (s.substr(0, prefixLength) == prefix)
    s = s.substr(prefixLength);
  // Strip off any leading length. (Similar to original scheme.)
  while (std::isdigit(s[0]))
    s = s.substr(1);
  return s;
}

static std::string
trimmedMangledTypeName(const clang::Type *ty,
                       clang::ItaniumMangleContext *mangler) {
  return trimmedMangledTypeName(clang::QualType(ty, /*Quals=*/0), mangler);
}

std::string
cudaq::details::getTagNameOfFunctionDecl(const clang::FunctionDecl *func,
                                         clang::ItaniumMangleContext *mangler) {
  if (auto *cxxMethod = dyn_cast<clang::CXXMethodDecl>(func)) {
    auto *cxxCls = cast<clang::CXXRecordDecl>(func->getParent());
    if (cxxMethod->isFunctionTemplateSpecialization()) {
      // Template member function.
      // class C {
      //   template<typename A> T operator()(args...) { ... }
      // };
      // cudaq::get_class_kernel_name<C, As...>();
      auto name = "instance_" +
                  trimmedMangledTypeName(cxxCls->getTypeForDecl(), mangler);
      assert(cxxMethod->getTemplateSpecializationArgs());
      for (auto &templArg :
           cxxMethod->getTemplateSpecializationArgs()->asArray())
        name += trimmedMangledTypeName(templArg.getAsType(), mangler);
      name += '.' + cudaq::details::getCxxMangledDeclName(func, mangler);
      LLVM_DEBUG(llvm::dbgs() << "template member name is: " << name << '\n');
      return name;
    }
    // Member function, but not a template function.
    // cudaq::get_class_kernel_name<C>();
    auto name = trimmedMangledTypeName(cxxCls->getTypeForDecl(), mangler);
    LLVM_DEBUG(llvm::dbgs() << "member name is: " << name << '\n');
    return name;
  }
  if (func->isTemplateInstantiation()) {
    // Template function.
    // template<typename A> __qpu__ T func(args ...);
    // cudaq::get_function_kernel_name<As...>("func");
    auto name = "instance_function_" + func->getName().str();
    for (auto templArg : func->getTemplateSpecializationArgs()->asArray())
      name += trimmedMangledTypeName(templArg.getAsType(), mangler);
    name += '.' + cudaq::details::getCxxMangledDeclName(func, mangler);
    LLVM_DEBUG(llvm::dbgs() << "template function name is: " << name << '\n');
    return name;
  }
  // Plain, old function.
  // __qpu__ T func(args ...);
  // cudaq::get_function_kernel_name("func");
  auto name = "function_" + func->getName().str();
  name += '.' + cudaq::details::getCxxMangledDeclName(func, mangler);
  LLVM_DEBUG(llvm::dbgs() << "function name is: " << name << '\n');
  return name;
}

namespace {
/// The QPU code finder class is used to find all function definitions that the
/// user has annotated with a QPU attribute (`__qpu__`). The finder is a simple
/// visitor that traverses declarations as called from the AST consumer.
class QPUCodeFinder : public clang::RecursiveASTVisitor<QPUCodeFinder> {
public:
  using Base = clang::RecursiveASTVisitor<QPUCodeFinder>;
  explicit QPUCodeFinder(cudaq::EmittedFunctionsCollection &funcsToEmit,
                         clang::CallGraph &cgb,
                         clang::ItaniumMangleContext *mangler)
      : functionsToEmit(funcsToEmit), callGraphBuilder(cgb), mangler(mangler) {}

  /// Add a kernel to the list of kernels to process.
  void processQpu(std::string &&kernelName, const clang::FunctionDecl *f) {
    LLVM_DEBUG(llvm::dbgs()
               << "adding kernel: " << kernelName << ", "
               << reinterpret_cast<void *>(const_cast<clang::FunctionDecl *>(f))
               << '\n');
    functionsToEmit.push_back(std::make_pair(std::move(kernelName), f));
    callGraphBuilder.addToCallGraph(const_cast<clang::FunctionDecl *>(f));
  }

  // Check some of the restrictions and limitations on kernel classes. These
  // restrictions are noted in the CUDA Quantum language specification.
  void check(const clang::CXXRecordDecl *x) {
    if (!checkedClass)
      return;

    auto &de = mangler->getASTContext().getDiagnostics();
    if (!x->bases().empty()) {
      auto id = de.getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "class inheritance is not allowed for CUDA Quantum kernel");
      de.Report(x->getBeginLoc(), id);
    }
    if (!x->fields().empty()) {
      auto id = de.getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "CUDA Quantum kernel class with data members is not yet supported");
      de.Report(x->getBeginLoc(), id);
    }
    unsigned quantumCount = 0;
    for (auto *method : x->methods()) {
      if (cudaq::ASTBridgeAction::ASTBridgeConsumer::isQuantum(method)) {
        if (quantumCount++) {
          auto id =
              de.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                 "CUDA Quantum kernel class with multiple "
                                 "quantum methods not yet supported");
          de.Report(method->getBeginLoc(), id);
          break;
        }
      }
    }
    checkedClass = nullptr;
  }

  bool TraverseCXXRecordDecl(clang::CXXRecordDecl *x) {
    // Keep track of which CXXRecordDecl is being visited for semantics
    // checking.
    auto *savedCheckedClass = checkedClass;
    checkedClass = x;
    bool result = Base::TraverseCXXRecordDecl(x);
    checkedClass = savedCheckedClass;
    return result;
  }

  bool TraverseFunctionDecl(clang::FunctionDecl *x) {
    // Do not allow non-kernel functions to use quantum data types.
    bool saveQuantumTypesNotAllowed = quantumTypesNotAllowed;
    auto result = Base::TraverseFunctionDecl(x);
    quantumTypesNotAllowed = saveQuantumTypesNotAllowed;
    return result;
  }

  bool VisitFunctionDecl(clang::FunctionDecl *func) {
    if (ignoreTemplate)
      return true;
    func = func->getDefinition();
    if (func) {
      if (cudaq::ASTBridgeAction::ASTBridgeConsumer::isQuantum(func)) {
        quantumTypesNotAllowed = false;
        // Run semantics checks on the kernel class.
        if (isa<clang::CXXMethodDecl>(func)) {
          auto *cxxClass = cast<clang::CXXRecordDecl>(func->getParent());
          check(cxxClass);
        }
        processQpu(cudaq::details::getTagNameOfFunctionDecl(func, mangler),
                   func);
      } else {
        quantumTypesNotAllowed = true;
      }
    }
    return true;
  }

  bool TraverseLambdaExpr(clang::LambdaExpr *x) {
    bool saveQuantumTypesNotAllowed = quantumTypesNotAllowed;
    // Rationale: a lambda expression may be passed from classical C++ code into
    // a quantum kernel. It is therefore natural to allow the lambda expression
    // to use quantum types.
    quantumTypesNotAllowed = false;
    auto result = Base::TraverseLambdaExpr(x);
    quantumTypesNotAllowed = saveQuantumTypesNotAllowed;
    return result;
  }

  bool VisitLambdaExpr(clang::LambdaExpr *lambda) {
    if (ignoreTemplate)
      return true;
    if (const auto *cxxMethodDecl = lambda->getCallOperator())
      if (const auto *f = cxxMethodDecl->getAsFunction()->getDefinition();
          f && cudaq::ASTBridgeAction::ASTBridgeConsumer::isQuantum(f))
        processQpu(cudaq::details::getTagNameOfFunctionDecl(f, mangler), f);
    return true;
  }

  // Ignore the body of a template that has not been instantiated.
  bool TraverseClassTemplateDecl(clang::ClassTemplateDecl *x) {
    ignoreTemplate = true;
    auto result = Base::TraverseClassTemplateDecl(x);
    ignoreTemplate = false;
    return result;
  }

  // Process the body of an instantiated template.
  bool TraverseClassTemplateSpecializationDecl(
      clang::ClassTemplateSpecializationDecl *x) {
    bool saveIgnoreTemplate = ignoreTemplate;
    ignoreTemplate = false;
    auto result = Base::TraverseClassTemplateSpecializationDecl(x);
    ignoreTemplate = saveIgnoreTemplate;
    return result;
  }

  // For a function template, skip the template declaration and traverse the
  // instantiations of the template function instead.
  bool TraverseFunctionTemplateDecl(clang::FunctionTemplateDecl *x) {
    if (x == x->getCanonicalDecl())
      if (!TraverseTemplateInstantiations(x))
        return false;
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *x) {
    // The check to make sure that quantum data types are only used in kernels
    // is done here. This checks both variable declarations and parameters.
    if (quantumTypesNotAllowed)
      if (auto *ty = x->getType().getTypePtr())
        if (auto *decl = ty->getAsCXXRecordDecl();
            decl && cudaq::isInNamespace(decl, "cudaq"))
          if (auto *id = decl->getIdentifier()) {
            auto name = id->getName();
            if (name.equals("qubit") || name.equals("qudit") ||
                name.equals("qspan") || name.startswith("qreg"))
              cudaq::details::reportClangError(
                  x, mangler,
                  "may not use quantum types in non-kernel functions");
          }
    return true;
  }

private:
  cudaq::EmittedFunctionsCollection &functionsToEmit;
  clang::CallGraph &callGraphBuilder;
  clang::ItaniumMangleContext *mangler;
  // A class that is being visited. Need to run semantics checks on it if and
  // only if it has a quantum kernel.
  const clang::CXXRecordDecl *checkedClass = nullptr;
  bool ignoreTemplate = false;
  bool quantumTypesNotAllowed = false;
};
} // namespace

namespace cudaq {
#ifndef NDEBUG
bool details::QuakeBridgeVisitor::pushValue(Value v) {
  LLVM_DEBUG(llvm::dbgs() << "push value: " << v << '\n');
  valueStack.push_back(v);
  return true;
}

Value details::QuakeBridgeVisitor::popValue() {
  Value result = peekValue();
  LLVM_DEBUG(llvm::dbgs() << "pop value\n");
  valueStack.pop_back();
  return result;
}
#endif

bool ASTBridgeAction::ASTBridgeConsumer::isQuantum(
    const clang::FunctionDecl *decl) {
  // Quantum kernels are Functions that are annotated with "quantum"
  if (auto attr = decl->getAttr<clang::AnnotateAttr>())
    return attr->getAnnotation().str() == "quantum";
  return false;
}

ASTBridgeAction::ASTBridgeConsumer::ASTBridgeConsumer(
    clang::CompilerInstance &compiler, OwningOpRef<ModuleOp> &_module,
    std::map<std::string, std::string> &cxx_mangled)
    : astContext(compiler.getASTContext()), ci(compiler),
      cxx_mangled_kernel_names(cxx_mangled), module(_module),
      builder(_module->getContext()) {
  mangler =
      clang::ItaniumMangleContext::create(astContext, ci.getDiagnostics());
  assert(mangler && "mangler creation failed");
}

void ASTBridgeAction::ASTBridgeConsumer::HandleTranslationUnit(
    clang::ASTContext &astContext) {
  // First make sure there are no syntax errors, etc.
  auto &de = astContext.getDiagnostics();
  auto errorCount = de.getClient()->getNumErrors();
  if (errorCount != 0) {
    auto id =
        de.getCustomDiagID(clang::DiagnosticsEngine::Error,
                           "C++ source has errors. nvq++ cannot proceed.");
    de.Report(id);
    std::exit(1);
  }

  // If no errors, then proceed to lowering the kernels.
  llvm::SmallVector<clang::Decl *> reachableFuncs =
      listReachableFunctions(callGraphBuilder.getRoot());
  auto *ctx = module->getContext();
  details::QuakeBridgeVisitor visitor(
      &astContext, ctx, builder, module.get(), symbol_table, functionsToEmit,
      reachableFuncs, cxx_mangled_kernel_names, ci, mangler);

  // Lower each kernel entry function.
  for (auto fdPair : functionsToEmit) {
    SymbolTableScope var_scope(symbol_table);
    std::string entryName = visitor.generateCudaqKernelName(fdPair);
    visitor.setEntryName(entryName);
    // Extend the mangled kernel names map.
    auto mangledFuncName = visitor.cxxMangledDeclName(fdPair.second);
    cxx_mangled_kernel_names.insert({entryName, mangledFuncName});
    LLVM_DEBUG(llvm::dbgs() << "lowering function: " << entryName << '\n');
    visitor.TraverseDecl(const_cast<clang::FunctionDecl *>(fdPair.second));
    if (auto func = module->lookupSymbol<func::FuncOp>(entryName)) {
      // Rationale: If a function marked as quantum code takes or returns
      // qubits, then it must be a pure quantum kernel that can only be called
      // from other quantum code and never from classical code. Conversely, a
      // kernel that is called from a quantum kernel entry function must (for
      // now) take or return qubits in order to be stitched into the same
      // circuit with the calling function. CUDA Quantum does not presently
      // support a quantum circuit that invokes a separable quantum circuit.
      // Launching a quantum circuit implies exactly one distinct circuit
      // will be mapped to and execute on the QPU.
      auto unitAttr = UnitAttr::get(ctx);
      // Flag func as a quantum kernel.
      func->setAttr(cudaq::kernelAttrName, unitAttr);
      if (!hasAnyQubitTypes(func.getFunctionType())) {
        // Flag func as an entry point to a quantum kernel.
        func->setAttr(cudaq::entryPointAttrName, unitAttr);
      }
    }
  }

  // Not a kernel or entry point, but part of the call graph.
  if (LOWERING_TRANSITIVE_CLOSURE) {
    for (auto *rfp : reachableFuncs) {
      if (clang::isa<clang::CXXConstructorDecl>(rfp))
        continue;

      if (auto *rf = dyn_cast<clang::FunctionDecl>(rfp)) {
        if (rf->getBody()) {
          SymbolTableScope var_scope(symbol_table);
          // Create the function and set the builder.
          auto mangledFuncName = visitor.cxxMangledDeclName(rf);
          visitor.setCurrentFunctionName(mangledFuncName);
          // FIXME: lower these classical compute functions to CC.
          visitor.TraverseDecl(rf);
        }
      }
    }
  }
}

bool ASTBridgeAction::ASTBridgeConsumer::HandleTopLevelDecl(
    clang::DeclGroupRef dg) {
  QPUCodeFinder finder(functionsToEmit, callGraphBuilder, mangler);
  // Loop over all decls, saving the function decls that are quantum kernels.
  for (const auto *decl : dg)
    finder.TraverseDecl(const_cast<clang::Decl *>(decl));
  return true;
}

Location toSourceLocation(MLIRContext *ctx, clang::ASTContext *astCtx,
                          const clang::SourceRange &srcRange) {
  auto &srcMgr = astCtx->getSourceManager();
  auto srcLoc = srcRange.getBegin();
  auto spellingLoc = srcMgr.getSpellingLoc(srcLoc);
  auto line = srcMgr.getSpellingLineNumber(spellingLoc);
  auto column = srcMgr.getSpellingColumnNumber(spellingLoc);
  auto file = srcMgr.getFilename(spellingLoc);
  return FileLineColLoc::get(ctx, file, line, column);
}
} // namespace cudaq

namespace cudaq::details {

std::string getCxxMangledTypeName(clang::QualType ty,
                                  clang::ItaniumMangleContext *mangler) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mangler->mangleTypeName(ty, os);
  os.flush();
  LLVM_DEBUG(llvm::dbgs() << "type name mangled as '" << s << "'\n");
  return s;
}

std::string getCxxMangledDeclName(clang::GlobalDecl decl,
                                  clang::ItaniumMangleContext *mangler) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mangler->mangleCXXName(decl, os);
  os.flush();
  return s;
}

} // namespace cudaq::details
