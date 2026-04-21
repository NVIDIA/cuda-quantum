/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "clang/Basic/TargetCXXABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include <cxxabi.h>
#include <fstream>
#include <regex>

#define DEBUG_TYPE "lower-ast"

using namespace mlir;

// This flag is useful when debugging and the lower-ast debug output crashes.
// The MLIR printer will, for various reasons, spontaneously crash when printing
// the full Op. Setting this to any value except `1` or `2` will attempt to
// print values. Setting this to `1` prints only the Op's name, avoiding some
// spurious crashes. Setting this to `2` omits trying to print the Values at
// all, which will avoid any/all random crashes in the MLIR printer.
llvm::cl::opt<unsigned> debugOpNameOnly("lower-ast-values", llvm::cl::init(0));

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
  if (quake::isQuakeType(ty))
    return true;
  // FIXME: next if case is a bug.
  if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(ty))
    return isQubitType(vecTy.getElementType());
  return false;
}

// Check the builtin type FunctionType to see if it has any references to Quake
// qubit types in its arguments and/or results.
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
  explicit QPUCodeFinder(
      cudaq::EmittedFunctionsCollection &funcsToEmit, clang::CallGraph &cgb,
      clang::ItaniumMangleContext *mangler, ModuleOp module,
      std::unordered_map<std::string, std::string> &customOperations)
      : functionsToEmit(funcsToEmit), callGraphBuilder(cgb), mangler(mangler),
        module(module), customOperationNames(customOperations) {}

  /// Add a kernel to the list of kernels to process.
  template <bool replace = true>
  void processQpu(const std::string &kernelName, const clang::FunctionDecl *f) {
    LLVM_DEBUG(llvm::dbgs()
               << "adding kernel: " << kernelName << ", "
               << reinterpret_cast<void *>(const_cast<clang::FunctionDecl *>(f))
               << '\n');
    auto iter = std::find_if(functionsToEmit.begin(), functionsToEmit.end(),
                             [&](auto p) { return p.first == kernelName; });
    if constexpr (replace) {
      if (iter == functionsToEmit.end())
        functionsToEmit.push_back(std::make_pair(kernelName, f));
      else
        iter->second = f;
      callGraphBuilder.addToCallGraph(const_cast<clang::FunctionDecl *>(f));
    } else {
      if (iter == functionsToEmit.end())
        functionsToEmit.push_back(std::make_pair(kernelName, f));
    }
  }

  // Check some of the restrictions and limitations on kernel classes. These
  // restrictions are noted in the CUDA-Q language specification.
  void check(const clang::CXXRecordDecl *x) {
    if (!checkedClass)
      return;

    auto &de = mangler->getASTContext().getDiagnostics();
    if (!x->bases().empty()) {
      auto id = de.getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "class inheritance is not allowed for CUDA-Q kernel");
      de.Report(x->getBeginLoc(), id);
    }
    if (!x->fields().empty()) {
      auto id = de.getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "CUDA-Q kernel class with data members is not yet supported");
      de.Report(x->getBeginLoc(), id);
    }
    unsigned quantumCount = 0;
    for (auto *method : x->methods()) {
      if (cudaq::ASTBridgeAction::ASTBridgeConsumer::isQuantum(method)) {
        if (quantumCount++) {
          auto id = de.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                       "CUDA-Q kernel class with multiple "
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

  bool VisitFunctionDecl(clang::FunctionDecl *x) {
    if (ignoreTemplate)
      return true;
    auto *func = x->getDefinition();

    if (func) {
      bool runChecks = false;
      if (cudaq::ASTBridgeAction::ASTBridgeConsumer::isCustomOpGenerator(
              func)) {
        customOperationNames[func->getName().str()] =
            cudaq::details::getTagNameOfFunctionDecl(func, mangler);
        runChecks = true;
      }
      if (cudaq::ASTBridgeAction::ASTBridgeConsumer::isQuantum(func))
        runChecks = true;
      else
        quantumTypesNotAllowed = true;

      if (runChecks) {
        quantumTypesNotAllowed = false;
        // Run semantics checks on the kernel class.
        if (isa<clang::CXXMethodDecl>(func)) {
          auto *cxxClass = cast<clang::CXXRecordDecl>(func->getParent());
          check(cxxClass);
        }
        processQpu(cudaq::details::getTagNameOfFunctionDecl(func, mangler),
                   func);
      }
    } else if (cudaq::ASTBridgeAction::ASTBridgeConsumer::isQuantum(x)) {
      // Add declarations to support separate compilation.
      processQpu</*replace=*/false>(
          cudaq::details::getTagNameOfFunctionDecl(x, mangler), x);
    }
    return true;
  }

  // NB: DataRecursionQueue* argument intentionally omitted.
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
      if (const auto *f = cxxMethodDecl->getAsFunction()->getDefinition())
        if (cudaq::ASTBridgeAction::ASTBridgeConsumer::isQuantum(f))
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

  bool isTupleReverseVar(clang::VarDecl *decl) {
    if (cudaq::isInNamespace(decl, "cudaq"))
      return decl->getName() == "TupleIsReverse";
    return false;
  }

  bool VisitVarDecl(clang::VarDecl *x) {
    if (isTupleReverseVar(x)) {
      auto loc = x->getLocation();
      auto opt = x->getAnyInitializer()->getIntegerConstantExpr(
          x->getASTContext(), &loc, false);
      if (opt) {
        LLVM_DEBUG(llvm::dbgs() << "tuples are reversed: " << *opt << '\n');
        tuplesAreReversed = !opt->isZero();
      }
    }
    if (cudaq::isInNamespace(x, "cudaq") &&
        cudaq::isInNamespace(x, "details") && x->getName() == "_nvqpp_sizeof") {
      // This constexpr is the sizeof a pauli_word and a std::string.
      auto loc = x->getLocation();
      auto opt = x->getAnyInitializer()->getIntegerConstantExpr(
          x->getASTContext(), &loc, false);
      assert(opt && "must compute the sizeof a cudaq::pauli_word");
      auto sizeofString = opt->getZExtValue();
      auto sizeAttr = module->getAttr(cudaq::runtime::sizeofStringAttrName);
      if (sizeAttr) {
        assert(sizeofString == cast<IntegerAttr>(sizeAttr).getUInt());
      } else {
        auto *ctx = module.getContext();
        auto i64Ty = IntegerType::get(ctx, 64);
        module->setAttr(cudaq::runtime::sizeofStringAttrName,
                        IntegerAttr::get(i64Ty, sizeofString));
      }
    }
    // The check to make sure that quantum data types are only used in kernels
    // is done here. This checks both variable declarations and parameters.
    if (quantumTypesNotAllowed)
      if (auto *ty = x->getType().getTypePtr())
        if (auto *decl = ty->getAsCXXRecordDecl();
            decl && cudaq::isInNamespace(decl, "cudaq"))
          if (auto *id = decl->getIdentifier()) {
            auto name = id->getName();
            if (name == "qubit" || name == "qudit" || name == "qspan" ||
                name.startswith("qreg") || name.startswith("qvector") ||
                name.startswith("qarray") || name.startswith("qview"))
              cudaq::details::reportClangError(
                  x, mangler,
                  "may not use quantum types in non-kernel functions");
          }
    return true;
  }

  bool isTupleReversed() const { return tuplesAreReversed; }

private:
  cudaq::EmittedFunctionsCollection &functionsToEmit;
  clang::CallGraph &callGraphBuilder;
  clang::ItaniumMangleContext *mangler;
  ModuleOp module;
  std::unordered_map<std::string, std::string> &customOperationNames;
  // A class that is being visited. Need to run semantics checks on it if and
  // only if it has a quantum kernel.
  const clang::CXXRecordDecl *checkedClass = nullptr;
  bool ignoreTemplate = false;
  bool quantumTypesNotAllowed = false;
  bool tuplesAreReversed = false;
};
} // namespace

#ifndef NDEBUG
namespace cudaq::details {
bool QuakeBridgeVisitor::pushValue(Value v) {
  LLVM_DEBUG(llvm::dbgs() << std::string(valueStack.size(), ' ')
                          << "+push value: ";
             if (debugOpNameOnly != 2) {
               auto *defOp = v.getDefiningOp();
               if (debugOpNameOnly == 1 && defOp)
                 llvm::dbgs() << defOp->getName();
               else
                 llvm::dbgs() << v;
             };
             llvm::dbgs() << '\n');
  valueStack.push_back(v);
  return true;
}

Value QuakeBridgeVisitor::popValue() {
  Value result = peekValue();
  LLVM_DEBUG(
      llvm::dbgs() << std::string(valueStack.size() - 1, ' ') << "-pop value: ";
      auto *defOp = result.getDefiningOp();
      if (debugOpNameOnly && defOp) llvm::dbgs() << defOp->getName() << '\n';
      else llvm::dbgs() << result << '\n';);
  valueStack.pop_back();
  return result;
}

SmallVector<Value> QuakeBridgeVisitor::lastValues(unsigned n) {
  assert(n <= valueStack.size() && "stack has fewer values than requested");
  LLVM_DEBUG(llvm::dbgs() << std::string(valueStack.size() - n, ' ')
                          << "-pop values <" << n << ">\n");
  mlir::SmallVector<mlir::Value> result(valueStack.end() - n, valueStack.end());
  valueStack.pop_back_n(n);
  return result;
}
} // namespace cudaq::details
#endif

namespace cudaq::details {

bool QuakeBridgeVisitor::generateFunctionDeclaration(
    StringRef funcName, const clang::FunctionDecl *x) {
  allowUnknownRecordType = true;
  if (!TraverseType(x->getType())) {
    reportClangError(x, mangler, "failed to generate type for kernel function");
    typeStack.clear();
    return false;
  }
  allowUnknownRecordType = false;
  if (!doSyntaxChecks(x))
    return false;
  auto funcTy = cast<FunctionType>(popType());
  auto loc = toLocation(x);
  [[maybe_unused]] auto fnPair = getOrAddFunc(loc, funcName, funcTy);
  assert(fnPair.first && "expected FuncOp to be created");
  if (!isa<clang::CXXMethodDecl>(x) || x->isStatic())
    fnPair.first->setAttr("no_this", builder.getUnitAttr());
  assert(typeStack.empty() && "expected type stack to be cleared");
  // Retain the attribute for custom operation generator functions
  if (cudaq::ASTBridgeAction::ASTBridgeConsumer::isCustomOpGenerator(x))
    fnPair.first->setAttr(cudaq::generatorAnnotation, builder.getUnitAttr());
  return true;
}

bool QuakeBridgeVisitor::isItaniumCXXABI() {
  return clang::TargetCXXABI{astContext->getCXXABIKind()}.isItaniumFamily();
}

} // namespace cudaq::details

namespace cudaq {
bool ASTBridgeAction::ASTBridgeConsumer::isQuantum(
    const clang::FunctionDecl *decl) {
  // Quantum kernels are Functions that are annotated with "quantum"
  if (auto attr = decl->getAttr<clang::AnnotateAttr>())
    return attr->getAnnotation().str() == cudaq::kernelAnnotation;
  return false;
}

bool ASTBridgeAction::ASTBridgeConsumer::isCustomOpGenerator(
    const clang::FunctionDecl *decl) {
  if (auto attr = decl->getAttr<clang::AnnotateAttr>())
    return attr->getAnnotation().str() == cudaq::generatorAnnotation;
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
  auto triple = astContext.getTargetInfo().getTriple().str();
  (*module)->setAttr(opt::factory::targetTripleAttrName,
                     StringAttr::get(module->getContext(), triple));
  std::string dataLayout{astContext.getTargetInfo().getDataLayoutString()};
  (*module)->setAttr(opt::factory::targetDataLayoutAttrName,
                     StringAttr::get(module->getContext(), dataLayout));
}

static bool isX86_64(clang::ASTContext &astContext) {
  return astContext.getTargetInfo().getTriple().getArch() ==
         llvm::Triple::x86_64;
}

void ASTBridgeAction::ASTBridgeConsumer::addFunctionDecl(
    const clang::FunctionDecl *funcDecl, details::QuakeBridgeVisitor &visitor,
    FunctionType funcTy, StringRef devFuncName, bool isDecl) {
  auto funcName = visitor.cxxMangledDeclName(funcDecl);
  if (module->lookupSymbol(funcName))
    return;
  auto loc = toSourceLocation(module->getContext(), &astContext,
                              funcDecl->getSourceRange());
  auto devFuncTy =
      module->lookupSymbol<func::FuncOp>(devFuncName).getFunctionType();
  OpBuilder build(module->getBodyRegion());
  OpBuilder::InsertionGuard guard(build);
  build.setInsertionPointToEnd(module->getBody());
  bool addThisPtr =
      isa<clang::CXXMethodDecl>(funcDecl) && !funcDecl->isStatic();
  FunctionType hostFuncTy =
      opt::factory::toHostSideFuncType(funcTy, addThisPtr, *module);
  auto func = build.create<func::FuncOp>(loc, funcName, hostFuncTy,
                                         ArrayRef<NamedAttribute>{});
  if (!addThisPtr)
    func->setAttr("no_this", build.getUnitAttr());

  if (isDecl) {
    func.setPrivate();
  } else {
    func.setPublic();
    // Create a dummy implementation for the host-side function. This is so that
    // MLIR's restriction on "public" visibility is met and MLIR does not remove
    // the declaration before we can autogenerate the code in a later pass.
    auto *block = func.addEntryBlock();
    build.setInsertionPointToStart(block);
    SmallVector<Value> results;
    for (auto resTy : hostFuncTy.getResults())
      results.push_back(build.create<cc::UndefOp>(loc, resTy));
    build.create<func::ReturnOp>(loc, results);
  }

  // Walk the arguments and add byval attributes where needed.
  assert(visitor.isItaniumCXXABI() && "Microsoft ABI not implemented");
  auto hiddenSRet = opt::factory::hasHiddenSRet(devFuncTy);
  auto delta = cc::numberOfHiddenArgs(addThisPtr, hiddenSRet);
  if (hiddenSRet) {
    // The first argument should be a pointer type if this function has a hidden
    // sret.
    if (auto ptrTy = dyn_cast<cc::PointerType>(hostFuncTy.getInput(0))) {
      auto eleTy = ptrTy.getElementType();
      assert(isa<cc::StructType>(eleTy) && "must be a struct");
      func.setArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName(),
                      TypeAttr::get(eleTy));
    }
  }

  // Pass over the parameters and add a byval attribute where required.
  for (auto iter : llvm::enumerate(funcDecl->parameters())) {
    auto *paramDecl = iter.value();
    // See clang::CodeGen::TargetInfo::getRecordArgABI()
    auto paramType = paramDecl->getType();
    if (auto *recTy = paramType->getAs<clang::RecordType>()) {
      auto *d = recTy->getDecl();
      if (isa<clang::CXXRecordDecl>(d)) {
        // On x86_64, canPassInRegisters() returning true corresponds to
        // RAA_Default, which means the compiler should use "normal C aggregate
        // rules". Therefore, this struct type value is passed using a pointer
        // and the byval attribute. Otherwise, the compiler will pass by an
        // indirection (pointer), omit the byval attribute, and skip making
        // copies, etc.
        // On aarch64, the convention is to omit the byval attribute.
        if (d->canPassInRegisters() && isX86_64(astContext)) {
          auto idx = iter.index();
          auto argPos = idx + delta;
          if (auto ptrTy =
                  dyn_cast<cc::PointerType>(hostFuncTy.getInput(argPos))) {
            auto strTy = ptrTy.getElementType();
            assert(isa<cc::StructType>(strTy) && "must be a struct");
            func.setArgAttr(argPos, LLVM::LLVMDialect::getByValAttrName(),
                            TypeAttr::get(strTy));
          } else if (opt::factory::structUsesTwoArguments(
                         funcTy.getInput(idx))) {
            // On x86_64, we may pass the struct as 2 distinct arguments. We
            // must adjust the offset on the fly.
            ++delta;
          }
        }
      }
    }
  }
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
      reachableFuncs, cxx_mangled_kernel_names, ci, mangler,
      customOperationNames, allocator, tuplesAreReversed);

  // First generate declarations for all kernels.
  bool ok = true;
  for (auto fdPair : functionsToEmit) {
    std::string entryName = visitor.generateCudaqKernelName(fdPair);
    ok &= visitor.generateFunctionDeclaration(entryName, fdPair.second);
  }
  if (!ok) {
    // Syntax errors: stop processing the kernels.
    return;
  }

  // Now lower each kernel function definition.
  for (auto fdPair : functionsToEmit) {
    SymbolTableScope var_scope(symbol_table);
    std::string entryName = visitor.generateCudaqKernelName(fdPair);
    visitor.setEntryName(entryName);
    // Extend the mangled kernel names map.
    auto mangledFuncName = visitor.cxxMangledDeclName(fdPair.second);
    cxx_mangled_kernel_names.insert({entryName, mangledFuncName});
    LLVM_DEBUG(llvm::dbgs() << "lowering function: " << entryName << '\n');
    visitor.resetNextTopLevelFunction();
    visitor.TraverseDecl(const_cast<clang::FunctionDecl *>(fdPair.second));
    if (auto func = module->lookupSymbol<func::FuncOp>(entryName)) {
      // Rationale: If a function marked as quantum code takes or returns
      // qubits, then it must be a pure quantum kernel that can only be called
      // from other quantum code and never from classical code. Conversely, a
      // kernel that is called from a quantum kernel entry function must (for
      // now) take or return qubits in order to be stitched into the same
      // circuit with the calling function. CUDA-Q does not presently
      // support a quantum circuit that invokes a separable quantum circuit.
      // Launching a quantum circuit implies exactly one distinct circuit
      // will be mapped to and execute on the QPU.
      auto unitAttr = UnitAttr::get(ctx);
      // Flag func as a quantum kernel.
      func->setAttr(kernelAttrName, unitAttr);
      if ((!hasAnyQubitTypes(func.getFunctionType())) &&
          (!cudaq::ASTBridgeAction::ASTBridgeConsumer::isCustomOpGenerator(
              fdPair.second))) {
        // Flag func as an entry point to a quantum kernel.
        func->setAttr(entryPointAttrName, unitAttr);
        // Generate a declaration for the CPU C++ function.
        addFunctionDecl(fdPair.second, visitor, func.getFunctionType(),
                        entryName, func.empty());
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
  QPUCodeFinder finder(functionsToEmit, callGraphBuilder, mangler, module.get(),
                       customOperationNames);
  // Loop over all decls, saving the function decls that are quantum kernels.
  for (const auto *decl : dg)
    finder.TraverseDecl(const_cast<clang::Decl *>(decl));
  tuplesAreReversed |= finder.isTupleReversed();
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

// Determine if this global has `extern "C"` linkage.
bool isInExternC(const clang::GlobalDecl &x) {
  if (auto *funcDecl = dyn_cast<clang::FunctionDecl>(x.getDecl()))
    return funcDecl->isExternC();
  if (auto *varDecl = dyn_cast<clang::VarDecl>(x.getDecl()))
    return varDecl->isExternC();
  return false;
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
  if (isInExternC(decl)) {
    if (auto *funcDecl = dyn_cast<clang::FunctionDecl>(decl.getDecl()))
      return funcDecl->getName().str();
    reportClangError(decl.getDecl(), mangler, "unexpected global");
  }
  std::string s;
  llvm::raw_string_ostream os(s);
  mangler->mangleCXXName(decl, os);
  os.flush();
  return s;
}

} // namespace cudaq::details
