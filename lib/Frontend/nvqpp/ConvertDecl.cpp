/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Todo.h"
#include <span>

#define DEBUG_TYPE "lower-ast-decl"

using namespace mlir;

namespace cudaq::details {

// FIXME: ignoring these allocator classes rather than traversing them. It would
// be better add them to the list of intercepted classes, but that code is
// expected to push a type on the stack.
bool ignoredClass(clang::RecordDecl *x) {
  if (auto *ident = x->getIdentifier()) {
    auto name = ident->getName();
    // Kernels don't support allocators, although they are found in
    // std::vector.
    if (isInNamespace(x, "std"))
      return name.equals("allocator_traits") || name.equals("iterator_traits");
    // Skip non-standard GNU helper classes.
    if (isInNamespace(x, "__gnu_cxx"))
      return name.equals("__alloc_traits");
  }
  return false;
}

bool QuakeBridgeVisitor::isKernelEntryPoint(const clang::FunctionDecl *decl) {
  if (!decl->hasBody())
    return false;
  for (auto fdPair : functionsToEmit) {
    if (decl == fdPair.second) {
      // This is an entry point.
      std::string entryName = generateCudaqKernelName(fdPair);
      setEntryName(entryName);
      // Extend the mangled kernel names map.
      auto mangledFuncName = cxxMangledDeclName(decl);
      namesMap.insert({entryName, mangledFuncName});
      return true;
    }
  }
  return false;
}

bool QuakeBridgeVisitor::needToLowerFunction(const clang::FunctionDecl *decl) {
  if (!decl->hasBody())
    return false;

  // Check if this is a kernel entry point.
  if (isKernelEntryPoint(decl))
    return true;

  if (LOWERING_TRANSITIVE_CLOSURE) {
    // Not a kernel entry point. Test to see if it is some other function we
    // need to lower.
    for (auto *rf : reachableFunctions) {
      if ((decl == rf) && decl->getBody()) {
        // Create the function and set the builder.
        auto mangledFuncName = cxxMangledDeclName(decl);
        setCurrentFunctionName(mangledFuncName);
        return true;
      }
    }
  }

  // Skip this function. It is not part of the call graph of QPU code.
  return false;
}

void QuakeBridgeVisitor::addArgumentSymbols(
    Block *entryBlock, ArrayRef<clang::ParmVarDecl *> parameters) {
  for (auto arg : llvm::enumerate(parameters)) {
    auto index = arg.index();
    auto *argVal = arg.value();
    auto name = argVal->getName();
    if (isa<OpaqueType>(entryBlock->getArgument(index).getType())) {
      // This is a reference type, we want to forward the value.
      symbolTable.insert(name, entryBlock->getArgument(index));
    } else {
      // Transform pass-by-value arguments to stack slots.
      auto loc = toLocation(argVal);
      auto parmTy = entryBlock->getArgument(index).getType();
      if (isa<FunctionType, cc::CallableType, cc::PointerType, cc::SpanLikeType,
              LLVM::LLVMStructType, quake::ControlType, quake::RefType,
              quake::VeqType, quake::WireType>(parmTy)) {
        symbolTable.insert(name, entryBlock->getArgument(index));
      } else {
        auto stackSlot = builder.create<cc::AllocaOp>(loc, parmTy);
        builder.create<cc::StoreOp>(loc, entryBlock->getArgument(index),
                                    stackSlot);
        symbolTable.insert(name, stackSlot);
      }
    }
  }
}

void QuakeBridgeVisitor::createEntryBlock(func::FuncOp func,
                                          const clang::FunctionDecl *x) {
  if (!func.getBlocks().empty())
    return;
  auto *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);
  addArgumentSymbols(entryBlock, x->parameters());
}

std::pair<func::FuncOp, /*alreadyDefined=*/bool>
QuakeBridgeVisitor::getOrAddFunc(Location loc, StringRef funcName,
                                 FunctionType funcTy) {
  auto func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) {
    if (!func.empty()) {
      // Already lowered function func, skip it.
      return {func, /*defined=*/true};
    }
    // Function was declared but not defined.
    return {func, /*defined=*/false};
  }
  // Function not found, so add it to the module.
  OpBuilder build(module.getBodyRegion());
  OpBuilder::InsertionGuard guard(build);
  build.setInsertionPointToEnd(module.getBody());
  SmallVector<NamedAttribute> attrs;
  func = build.create<func::FuncOp>(loc, funcName, funcTy, attrs);
  func.setPrivate();
  return {func, /*defined=*/false};
}

bool QuakeBridgeVisitor::interceptRecordDecl(clang::RecordDecl *x) {
  // Some decls will be intercepted and replaced with high-level types in quake.
  // Do this here to avoid traversing their fields, etc.
  auto *ident = x->getIdentifier();
  if (!ident || x->isLambda())
    return false;
  auto name = ident->getName();
  auto *ctx = builder.getContext();
  if (isInNamespace(x, "cudaq")) {
    // Types from the `cudaq` namespace.
    // A qubit is a qudit<LEVEL=2>.
    if (name.equals("qudit") || name.equals("qubit"))
      return pushType(quake::RefType::get(ctx));
    // qreg<SIZE,LEVEL>, qarray<SIZE,LEVEL>, qspan<SIZE,LEVEL>
    if (name.equals("qspan") || name.equals("qreg") || name.equals("qarray")) {
      // If the first template argument is not `std::dynamic_extent` then we
      // have a constant sized VeqType.
      if (auto *tempSpec =
              dyn_cast<clang::ClassTemplateSpecializationDecl>(x)) {
        auto templArg = tempSpec->getTemplateArgs()[0];
        assert(templArg.getKind() ==
               clang::TemplateArgument::ArgKind::Integral);
        auto getExtValueHelper = [](auto v) -> std::int64_t {
          if (v.isUnsigned())
            return static_cast<std::int64_t>(v.getZExtValue());
          return v.getSExtValue();
        };
        std::int64_t size = getExtValueHelper(templArg.getAsIntegral());
        if (size != static_cast<std::int64_t>(std::dynamic_extent))
          return pushType(quake::VeqType::get(ctx, size));
      }
      return pushType(quake::VeqType::getUnsized(ctx));
    }
    // qvector<LEVEL>, qview<LEVEL>
    if (name.equals("qvector") || name.equals("qview"))
      return pushType(quake::VeqType::getUnsized(ctx));
    if (name.equals("state"))
      return pushType(opt::factory::getCudaqStateType(ctx));
    auto loc = toLocation(x);
    TODO_loc(loc, "unhandled type, " + name + ", in cudaq namespace");
  }
  if (isInNamespace(x, "std")) {
    if (name.equals("vector")) {
      auto *cts = dyn_cast<clang::ClassTemplateSpecializationDecl>(x);
      // Traverse template argument 0 to get the vector's element type.
      if (!cts || !TraverseType(cts->getTemplateArgs()[0].getAsType()))
        return false;
      return pushType(cc::StdvecType::get(ctx, popType()));
    }
    // std::vector<bool>   =>   cc.stdvec<i1>
    if (name.equals("_Bit_reference") || name.equals("__bit_reference")) {
      // Reference to a bit in a std::vector<bool>. Promote to a value.
      return pushType(builder.getI1Type());
    }
    if (name.equals("_Bit_type"))
      return pushType(builder.getI64Type());
    if (name.equals("complex")) {
      auto *cts = dyn_cast<clang::ClassTemplateSpecializationDecl>(x);
      // Traverse template argument 0 to get the complex's element type.
      if (!cts || !TraverseType(cts->getTemplateArgs()[0].getAsType()))
        return false;
      auto memTy = popType();
      return pushType(ComplexType::get(memTy));
    }
    if (name.equals("initializer_list")) {
      auto *cts = dyn_cast<clang::ClassTemplateSpecializationDecl>(x);
      // Traverse template argument 0, the initializer list's element type.
      if (!cts || !TraverseType(cts->getTemplateArgs()[0].getAsType()))
        return false;
      auto memTy = popType();
      return pushType(cc::ArrayType::get(memTy));
    }
    if (name.equals("function")) {
      auto *cts = cast<clang::ClassTemplateSpecializationDecl>(x);
      // Traverse template argument 0 to get the function's signature.
      if (!TraverseType(cts->getTemplateArgs()[0].getAsType()))
        return false;
      auto fnTy = cast<FunctionType>(popType());
      return pushType(cc::CallableType::get(ctx, fnTy));
    }
    if (name.equals("reference_wrapper")) {
      auto *cts = cast<clang::ClassTemplateSpecializationDecl>(x);
      // Traverse template argument 0 to get the function's signature.
      if (!TraverseType(cts->getTemplateArgs()[0].getAsType()))
        return false;
      auto refTy = popType();
      if (isa<quake::RefType, quake::VeqType>(refTy))
        return pushType(refTy);
      return pushType(cc::PointerType::get(ctx, refTy));
    }
    if (name.equals("basic_string")) {
      if (allowUnknownRecordType) {
        // Kernel argument list contains a `std::string` type. Intercept it and
        // generate a clang diagnostic when returning out of determining the
        // kernel's type signature.
        return true;
      }
      TODO_x(toLocation(x), x, mangler, "std::string type");
      return false;
    }
    if (name.equals("pair")) {
      if (allowUnknownRecordType)
        return true;
      TODO_x(toLocation(x), x, mangler, "std::pair type");
      return false;
    }
    if (name.equals("tuple")) {
      if (allowUnknownRecordType)
        return true;
      TODO_x(toLocation(x), x, mangler, "std::tuple type");
      return false;
    }
    if (ignoredClass(x))
      return true;
    if (allowUnknownRecordType) {
      // This is a catch all for other container types (deque, map, set, etc.)
      // that the user may try to pass as arguments to a kernel. Returning true
      // here will cause the kernel's signature to emit a diagnostic.
      return true;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "in std namespace, " << name << " is not matched\n");
  }

  if (isInNamespace(x, "__gnu_cxx")) {
    if (name.equals("__promote") || name.equals("__promote_2")) {
      // Recover the typedef in this class. Then find the canonical type
      // resolved for that typedef and push that as the type.
      [[maybe_unused]] unsigned depth = typeStack.size();
      for (auto *d : x->decls())
        if (auto *tdDecl = dyn_cast<clang::TypedefDecl>(d)) {
          auto qt = tdDecl->getUnderlyingType().getCanonicalType();
          if (!TraverseType(qt))
            return false;
          break;
        }
      assert(typeStack.size() == depth + 1);
      return true;
    }
    if (name.equals("__normal_iterator")) {
      auto *cts = cast<clang::ClassTemplateSpecializationDecl>(x);
      if (!TraverseType(cts->getTemplateArgs()[0].getAsType()))
        return false;
      return true;
    }
  }
  return false; /* not intercepted */
}

template <typename D>
bool QuakeBridgeVisitor::traverseAnyRecordDecl(D *x) {
  if (interceptRecordDecl(x))
    return true;
  if (x->isLambda()) {
    // If this is a lambda, then push the function type on the type stack.
    auto *funcDecl = findCallOperator(cast<clang::CXXRecordDecl>(x));
    if (!TraverseType(funcDecl->getType())) {
      auto loc = toLocation(funcDecl);
      emitFatalError(loc, "expected type for call operator");
    }
    return pushType(cc::CallableType::get(cast<FunctionType>(popType())));
  }
  return false;
}
bool QuakeBridgeVisitor::TraverseRecordDecl(clang::RecordDecl *x) {
  if (traverseAnyRecordDecl(x))
    return true;
  return Base::TraverseRecordDecl(x);
}
bool QuakeBridgeVisitor::TraverseCXXRecordDecl(clang::CXXRecordDecl *x) {
  if (traverseAnyRecordDecl(x))
    return true;
  return Base::TraverseCXXRecordDecl(x);
}
bool QuakeBridgeVisitor::TraverseClassTemplateSpecializationDecl(
    clang::ClassTemplateSpecializationDecl *x) {
  if (traverseAnyRecordDecl(x))
    return true;
  return Base::TraverseClassTemplateSpecializationDecl(x);
}

bool QuakeBridgeVisitor::TraverseFunctionDecl(clang::FunctionDecl *x) {
  // If we're already generating code (this FunctionDecl is nested), we only
  // traverse the type, adding the function type to the type stack.
  if (builder.getBlock()) {
    if (!TraverseType(x->getType()))
      return false;
    return WalkUpFromFunctionDecl(x);
  }

  // If function is not on the list to be lowered, skip it.
  if (!needToLowerFunction(x))
    return true;
  // If this function is a function template and not the specialization of the
  // function template, we skip it. We only want to lower template functions
  // that have their types resolved.
  if (x->getDescribedFunctionTemplate() &&
      !x->isFunctionTemplateSpecialization())
    return true;

  LLVM_DEBUG(llvm::dbgs() << "found function to lower: "
                          << x->getQualifiedNameAsString() << '\n');

  // The following is copied/expanded from RecursiveASTVisitor, especially
  // TraverseFunctionHelper(), since we can't call, override, or customize
  // private methods.
  for (unsigned i = 0; i < x->getNumTemplateParameterLists(); ++i) {
    if (auto *TPL = x->getTemplateParameterList(i)) {
      for (auto *D : *TPL)
        if (!TraverseDecl(D))
          return false;
      if (auto *requiresClause = TPL->getRequiresClause())
        if (!TraverseStmt(requiresClause))
          return false;
    }
  }
  if (!TraverseNestedNameSpecifierLoc(x->getQualifierLoc()))
    return false;
  if (!TraverseDeclarationNameInfo(x->getNameInfo()))
    return false;

  // If we're an explicit template specialization, iterate over the
  // template args that were explicitly specified.  If we were doing
  // this in typing order, we'd do it between the return type and
  // the function args, but both are handled by the FunctionTypeLoc
  // above, so we have to choose one side.  I've decided to do before.
  if (const auto *FTSI = x->getTemplateSpecializationInfo()) {
    if (FTSI->getTemplateSpecializationKind() != clang::TSK_Undeclared &&
        FTSI->getTemplateSpecializationKind() !=
            clang::TSK_ImplicitInstantiation) {
      // A specialization might not have explicit template arguments if it
      // has a templated return type and concrete arguments.
      if (const auto *tali = FTSI->TemplateArgumentsAsWritten) {
        auto *tal = tali->getTemplateArgs();
        for (unsigned i = 0; i != tali->NumTemplateArgs; ++i)
          if (!TraverseTemplateArgumentLoc(tal[i]))
            return false;
      }
    }
  }

  // Traversing the typeloc data structure gives us the unresolved surface
  // syntax, so a decl like `auto fn(auto p)` won't have reified types.
  if (!TraverseType(x->getType()))
    return false;

  // Customization here.
  // After we have the function's type and arguments, create the function and
  // set the builder, if and only if this is a top-level visit to a kernel. If
  // this is just a reference to a kernel, the lowering will happen at some
  // point during the visit to each kernel in the compilation unit. Any
  // referenced kernel should never naively be lowered in the context of the
  // kernel being visited that contains the reference.
  auto funcName = getCurrentFunctionName();
  auto loc = toLocation(x);
  if (funcName.empty())
    return true;

  resetCurrentFunctionName();
  // At present, the bridge only lowers kernels.
  auto funcTy = cast<FunctionType>(popType());
  auto [func, alreadyDefined] = getOrAddFunc(loc, funcName, funcTy);
  if (alreadyDefined)
    return true;

  LLVM_DEBUG(llvm::dbgs() << "created function: " << funcName << " : "
                          << func.getFunctionType() << '\n');
  func.setPublic();
  createEntryBlock(func, x);
  builder.setInsertionPointToEnd(&func.front());
  skipCompoundScope = true;

  // Visit the trailing requires clause, if any.
  if (auto *trailingRequiresClause = x->getTrailingRequiresClause())
    if (!TraverseStmt(trailingRequiresClause))
      return false;

  if (auto *ctor = dyn_cast<clang::CXXConstructorDecl>(x)) {
    // Constructor initializers.
    for (auto *I : ctor->inits())
      if (I->isWritten() || shouldVisitImplicitCode())
        if (!TraverseConstructorInitializer(I))
          return false;
  }

  bool VisitBody = x->isThisDeclarationADefinition() &&
                   (!x->isDefaulted() || shouldVisitImplicitCode());

  if (const auto *MD = dyn_cast<clang::CXXMethodDecl>(x))
    if (const auto *RD = MD->getParent())
      if (RD->isLambda() && declaresSameEntity(RD->getLambdaCallOperator(), MD))
        VisitBody = VisitBody && getDerived().shouldVisitLambdaBody();

  if (VisitBody) {
    if (!TraverseStmt(x->getBody()))
      return false;
    // Body may contain using declarations whose shadows are parented to the
    // FunctionDecl itself.
    for (auto *Child : x->decls())
      if (isa<clang::UsingShadowDecl>(Child))
        if (!TraverseDecl(Child))
          return false;
  }
  // Visit any attributes attached to this declaration.
  for (auto *attr : x->attrs())
    if (!TraverseAttr(attr))
      return false;
  // Do NOT WalkUpFromFunctionDecl(x);
  if (auto *method = dyn_cast<clang::CXXMethodDecl>(x))
    if (raisedError && method->getParent()->isLambda()) {
      auto &de = astContext->getDiagnostics();
      const auto id =
          de.getCustomDiagID(clang::DiagnosticsEngine::Remark,
                             "An inaccessible symbol in a lambda expression "
                             "may be from an implicit capture of a variable "
                             "that is not present in a kernel marked __qpu__.");
      auto db = de.Report(method->getBeginLoc(), id);
      const auto range = method->getSourceRange();
      db.AddSourceRange(clang::CharSourceRange::getCharRange(range));
      raisedError = false;
    }
  if (!hasTerminator(builder.getBlock()))
    builder.create<func::ReturnOp>(toLocation(x));
  builder.clearInsertionPoint();
  return true;
}

bool QuakeBridgeVisitor::VisitCXXScalarValueInitExpr(
    clang::CXXScalarValueInitExpr *x) {
  // This is the basis for a template function.
  Type ty = peekType();
  Value val = peekValue();
  if (val.getType() != ty)
    if (auto ptrTy = dyn_cast<cc::PointerType>(val.getType()))
      if (ptrTy.getElementType() == ty) {
        auto v = popValue();
        auto loc = toLocation(x);
        return pushValue(builder.create<cc::LoadOp>(loc, v));
      }
  return true;
}

bool QuakeBridgeVisitor::VisitFunctionDecl(clang::FunctionDecl *x) {
  assert(builder.getBlock() && "must be generating code");
  auto loc = toLocation(x);
  auto kernName = [&]() {
    if (isKernelEntryPoint(x))
      return generateCudaqKernelName(x);
    return cxxMangledDeclName(x);
  }();
  auto kernSym = SymbolRefAttr::get(builder.getContext(), kernName);
  auto typeFromStack = peelPointerFromFunction(popType());
  if (auto f = module.lookupSymbol<func::FuncOp>(kernSym)) {
    auto fTy = f.getFunctionType();
    assert(typeFromStack == fTy);
    auto fSym = f.getSymNameAttr();
    return pushValue(builder.create<func::ConstantOp>(loc, fTy, fSym));
  }
  auto funcOp = getOrAddFunc(loc, kernName, typeFromStack).first;
  return pushValue(builder.create<func::ConstantOp>(
      loc, funcOp.getFunctionType(), funcOp.getSymNameAttr()));
}

bool QuakeBridgeVisitor::VisitNamedDecl(clang::NamedDecl *x) {
  if (!builder.getBlock() || inRecType) {
    // This decl was reached walking a record type. We don't need to look up
    // the symbol, it's just a member name in the type.
    return true;
  }
  if (x->getIdentifier()) {
    // 1. Look for symbol in the local scope.
    if (!symbolTable.count(x->getName())) {
      // 2. TODO: If the symbol isn't in the local scope, it is a global.
      // Don't look for a global in the module here since we do not allow
      // kernels to access globals at present.
      cudaq::emitFatalError(toLocation(x->getSourceRange()),
                            "Cannot find " + x->getNameAsString() +
                                " in the symbol table.");
    }
    return pushValue(symbolTable.lookup(x->getName()));
  }
  return true;
}

bool QuakeBridgeVisitor::VisitParmVarDecl(clang::ParmVarDecl *x) {
  // If the builder has no insertion point, then this is a prototype.
  if (!builder.getBlock())
    return true;

  if (!x->getIdentifier()) {
    // Parameter has no name, so cannot be referenced. Skip it.
    return true;
  }

  auto name = x->getName();
  if (symbolTable.count(name)) {
    pushValue(symbolTable.lookup(name));
    return true;
  }

  // Something has gone very wrong.
  LLVM_DEBUG(llvm::dbgs() << "parameter was not found\n"; x->dump());
  llvm::report_fatal_error(
      "parameters for the current function must already be entered in the "
      "symbol table, but this parameter wasn't found.");
}

// A variable declaration may or may not have an initializer. This custom
// traversal makes sure that the type of the variable is visited and pushed so
// that VisitVarDecl has the variable's type, whether an initialization
// expression is present or not.
bool QuakeBridgeVisitor::TraverseVarDecl(clang::VarDecl *x) {
  [[maybe_unused]] auto typeStackDepth = typeStack.size();
  for (unsigned i = 0; i < x->getNumTemplateParameterLists(); i++) {
    if (auto *tpl = x->getTemplateParameterList(i)) {
      for (auto *decl : *tpl)
        if (!TraverseDecl(decl))
          return false;
      if (auto *requiresClause = tpl->getRequiresClause())
        if (!TraverseStmt(requiresClause))
          return false;
    }
  }
  if (!TraverseNestedNameSpecifierLoc(x->getQualifierLoc()))
    return false;
  if (!TraverseType(x->getType()))
    return false;
  assert(typeStack.size() == typeStackDepth + 1 &&
         "expected variable to have a type");
  if (!isa<clang::ParmVarDecl>(x) && !x->isCXXForRangeDecl())
    if (auto *init = x->getInit())
      if (!TraverseStmt(init))
        return false;
  if (auto *dc = dyn_cast<clang::DeclContext>(x))
    for (auto *child : dc->decls())
      if (!canIgnoreChildDeclWhileTraversingDeclContext(child))
        if (!TraverseDecl(child))
          return false;
  for (auto *attr : x->attrs())
    if (!TraverseAttr(attr))
      return false;
  auto result = WalkUpFromVarDecl(x);
  assert(typeStack.size() == typeStackDepth &&
         "expected variable's type to be consumed");
  return result;
}

bool QuakeBridgeVisitor::VisitVarDecl(clang::VarDecl *x) {
  if (allowUnknownRecordType) {
    // Processing a kernel's signature. Ignore variable decls.
    return true;
  }
  Type type = popType();
  if (x->hasInit() && !x->isCXXForRangeDecl())
    type = peekValue().getType();
  assert(type && "variable must have a valid type");
  auto loc = toLocation(x->getSourceRange());
  auto name = x->getName();
  if (auto qType = dyn_cast<quake::VeqType>(type)) {
    // Variable is of !quake.veq type.
    mlir::Value qreg;
    std::size_t qregSize = qType.getSize();
    if (qregSize == 0 || (x->hasInit() && !valueStack.empty())) {
      // This is a `qreg q(N);` or `qreg &name = exp;`
      qreg = popValue();
    } else {
      // this is a qreg<N> q;
      auto qregSizeVal = builder.create<mlir::arith::ConstantIntOp>(
          loc, qregSize, builder.getIntegerType(64));
      if (qregSize != 0)
        qreg = builder.create<quake::AllocaOp>(loc, qType);
      else
        qreg = builder.create<quake::AllocaOp>(loc, qType, qregSizeVal);
    }
    symbolTable.insert(name, qreg);
    // allocated_qreg_names.push_back(name);
    return pushValue(qreg);
  }

  if (auto qType = dyn_cast<quake::RefType>(type)) {
    // Variable is of !quake.ref type.
    if (x->hasInit() && !valueStack.empty()) {
      auto val = popValue();
      symbolTable.insert(name, val);
      return pushValue(val);
    }
    auto zero = builder.create<mlir::arith::ConstantIntOp>(
        loc, 0, builder.getIntegerType(64));
    auto qregSizeOne = builder.create<quake::AllocaOp>(
        loc, quake::VeqType::get(builder.getContext(), 1));
    Value addressTheQubit =
        builder.create<quake::ExtractRefOp>(loc, qregSizeOne, zero);
    symbolTable.insert(name, addressTheQubit);
    return pushValue(addressTheQubit);
  }

  // Here we maybe have something like auto var = mz(qreg)
  if (auto vecType = dyn_cast<cc::StdvecType>(type)) {
    // Variable is of !cc.stdvec type.
    if (x->getInit()) {
      // At the very least, its a vector var = vec_init;
      auto initVec = popValue();
      symbolTable.insert(x->getName(), initVec);

      // Let's try to see if this was a auto var = mz(qreg)
      // and if so, find the mz and tag it with the variable name
      auto elementType = vecType.getElementType();

      // Drop out if this is not an i1
      if (!elementType.isIntOrFloat() ||
          elementType.getIntOrFloatBitWidth() != 1)
        return true;

      // Assign registerName
      if (auto descr = initVec.getDefiningOp<quake::DiscriminateOp>())
        if (auto meas = descr.getMeasurement()
                            .getDefiningOp<quake::MeasurementInterface>())
          meas.setRegisterName(builder.getStringAttr(x->getName()));

      // Did this come from a stdvec init op? If not drop out
      auto stdVecInit = initVec.getDefiningOp<cc::StdvecInitOp>();
      if (!stdVecInit)
        return true;

      // Did the first operand come from an LLVM AllocaOp, if not drop out
      auto bitVecAllocation =
          stdVecInit.getOperand(0).getDefiningOp<cc::AllocaOp>();
      if (!bitVecAllocation)
        return true;

      // Search the AllocaOp users, find a potential GEPOp
      for (auto user : bitVecAllocation->getUsers()) {
        auto gepOp = dyn_cast<cc::ComputePtrOp>(user);
        if (!gepOp)
          continue;

        // Must have users
        if (gepOp->getUsers().empty())
          continue;

        // Is the first use a StoreOp, if so, we'll get its operand
        // and see if it came from an MzOp
        auto firstGepUser = *gepOp->getResult(0).getUsers().begin();
        if (auto storeOp = dyn_cast<cc::StoreOp>(firstGepUser)) {
          auto result = storeOp->getOperand(0);
          if (auto discr = result.getDefiningOp<quake::DiscriminateOp>())
            if (auto mzOp =
                    discr.getMeasurement().getDefiningOp<quake::MzOp>()) {
              // Found it, tag it with the name.
              mzOp.setRegisterName(builder.getStringAttr(x->getName()));
              break;
            }
        }
      }

      return true;
    }
  }

  if (auto callableTy = dyn_cast<cc::CallableType>(type)) {
    // Variable is of !cc.callable type. Callables are always in the value
    // domain.
    auto callable = popValue();
    symbolTable.insert(name, callable);
    return pushValue(callable);
  }

  // Variable is of some basic type not already handled. Create a local stack
  // slot in which to save the value. This stack slot is the variable in the
  // memory domain.
  if (!x->getInit() || x->isCXXForRangeDecl()) {
    Value alloca = builder.create<cc::AllocaOp>(loc, type);
    symbolTable.insert(x->getName(), alloca);
    return pushValue(alloca);
  }

  // Initialization expression is present.
  auto initValue = popValue();

  // If this was an auto var = mz(q), then we want to know the
  // var name, as it will serve as the classical bit register name
  if (auto discr = initValue.getDefiningOp<quake::DiscriminateOp>())
    if (auto mz = discr.getMeasurement().getDefiningOp<quake::MzOp>())
      mz.setRegisterName(builder.getStringAttr(x->getName()));

  assert(initValue && "initializer value must be lowered");
  if (isa<IntegerType>(initValue.getType()) && isa<IntegerType>(type)) {
    if (initValue.getType().getIntOrFloatBitWidth() <
        type.getIntOrFloatBitWidth()) {
      // FIXME: Use zero-extend if this is unsigned!
      initValue = builder.create<arith::ExtSIOp>(loc, type, initValue);
    } else if (initValue.getType().getIntOrFloatBitWidth() >
               type.getIntOrFloatBitWidth()) {
      initValue = builder.create<arith::TruncIOp>(loc, type, initValue);
    }
  } else if (isa<IntegerType>(initValue.getType()) && isa<FloatType>(type)) {
    // FIXME: Use UIToFP if this is unsigned!
    initValue = builder.create<arith::SIToFPOp>(loc, type, initValue);
  }

  if (auto initObject = initValue.getDefiningOp<cc::AllocaOp>()) {
    // Initialization expression already left an object in memory. This could be
    // because an object was constructed. TODO: this needs to also handle the
    // case that an object must be cloned instead of casted.
    assert(type == initObject.getType());
    symbolTable.insert(x->getName(), initValue);
    return pushValue(initValue);
  }
  auto qualTy = x->getType().getCanonicalType();
  auto isStdvecBoolReference = [&](clang::QualType &qualTy) {
    if (auto *recTy = dyn_cast<clang::RecordType>(qualTy.getTypePtr())) {
      auto *recDecl = recTy->getDecl();
      if (isInNamespace(recDecl, "std")) {
        auto name = recDecl->getNameAsString();
        return name == "_Bit_reference" || name == "__bit_reference";
      }
    }
    return false;
  };
  if (isStdvecBoolReference(qualTy) || qualTy.getTypePtr()->isReferenceType()) {
    // A similar case is when the C++ variable is a reference to a subobject.
    assert(isa<cc::PointerType>(type));
    Value cast = builder.create<cc::CastOp>(loc, type, initValue);
    symbolTable.insert(x->getName(), cast);
    return pushValue(cast);
  }

  // Initialization expression resulted in a value. Create a variable and save
  // that value to the variable's memory address.
  Value alloca = builder.create<cc::AllocaOp>(loc, type);
  builder.create<cc::StoreOp>(loc, initValue, alloca);
  symbolTable.insert(x->getName(), alloca);
  return pushValue(alloca);
}

} // namespace cudaq::details
