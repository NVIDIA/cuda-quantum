/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include <span>

#define DEBUG_TYPE "lower-ast-decl"

using namespace mlir;

namespace cudaq::detail {

// FIXME: ignoring these allocator classes rather than traversing them. It would
// be better add them to the list of intercepted classes, but that code is
// expected to have a type as the result.
bool ignoredClass(clang::RecordDecl *x) {
  if (auto *ident = x->getIdentifier()) {
    auto name = ident->getName();
    // Kernels don't support allocators, although they are found in
    // std::vector.
    if (isInNamespace(x, "std"))
      return name == "allocator_traits" || name == "iterator_traits";
    // Skip non-standard GNU helper classes.
    if (isInNamespace(x, "__gnu_cxx"))
      return name == "__alloc_traits";
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

llvm::StringRef QuakeBridgeVisitor::getSymbolName(const clang::NamedDecl *x) {
  if (auto *parm = dyn_cast<clang::ParmVarDecl>(x))
    if (auto *func = dyn_cast<clang::FunctionDecl>(parm->getDeclContext()))
      if (auto *pattern = func->getTemplateInstantiationPattern())
        for (auto *patternParm : pattern->parameters())
          if (patternParm->isParameterPack() &&
              patternParm->getName() == parm->getName()) {
            // Every element of the pack is named for the pack.
            std::string name = (parm->getName() + "." +
                                llvm::Twine(parm->getFunctionScopeIndex()))
                                   .str();
            return astContext->Idents.get(name).getName();
          }
  return x->getName();
}

void QuakeBridgeVisitor::addArgumentSymbols(
    Block *entryBlock, ArrayRef<clang::ParmVarDecl *> parameters) {
  for (auto arg : llvm::enumerate(parameters)) {
    auto index = arg.index();
    auto *argVal = arg.value();
    auto name = getSymbolName(argVal);
    if (isa<OpaqueType>(entryBlock->getArgument(index).getType())) {
      // This is a reference type, we want to forward the value.
      symbolTable.insert(name, entryBlock->getArgument(index));
    } else {
      // Transform pass-by-value arguments to stack slots.
      auto loc = toLocation(argVal);
      auto parmTy = entryBlock->getArgument(index).getType();
      if (isa<FunctionType, cc::CallableType, cc::IndirectCallableType,
              cc::PointerType, cc::SpanLikeType, LLVM::LLVMStructType,
              cudaq::quake::ControlType, cudaq::quake::RefType,
              cudaq::quake::StruqType, cudaq::quake::VeqType,
              cudaq::quake::WireType>(parmTy)) {
        symbolTable.insert(name, entryBlock->getArgument(index));
      } else {
        auto stackSlot = cc::AllocaOp::create(builder, loc, parmTy);
        cc::StoreOp::create(builder, loc, entryBlock->getArgument(index),
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

std::pair<func::FuncOp, bool>
QuakeBridgeVisitor::getOrAddFunc(Location loc, StringRef funcName,
                                 FunctionType funcTy) {
  return cudaq::opt::factory::getOrAddFunc(loc, funcName, funcTy, module);
}

// The handlers of these classes are used for the classes that derive from them.
static_assert(derivedDeclsAreKnown<
              clang::FunctionDecl, clang::CXXMethodDecl,
              clang::CXXConstructorDecl, clang::CXXDestructorDecl,
              clang::CXXConversionDecl, clang::CXXDeductionGuideDecl>());
static_assert(derivedDeclsAreKnown<
              clang::VarDecl, clang::ParmVarDecl, clang::ImplicitParamDecl,
              clang::DecompositionDecl, clang::OMPCapturedExprDecl,
              clang::VarTemplateSpecializationDecl,
              clang::VarTemplatePartialSpecializationDecl>());

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::FunctionDecl *x) {
  // If we're already generating code (this FunctionDecl is nested), this is a
  // reference to the function.
  if (builder.getBlock())
    return referenceFunction(x);

  // If function is not on the list to be lowered, skip it.
  if (!needToLowerFunction(x))
    return std::nullopt;
  // If this function is a function template and not the specialization of the
  // function template, we skip it. We only want to lower template functions
  // that have their types resolved.
  if (x->getDescribedFunctionTemplate() &&
      !x->isFunctionTemplateSpecialization())
    return std::nullopt;

  LLVM_DEBUG(llvm::dbgs() << "found function to lower: "
                          << x->getQualifiedNameAsString() << '\n');

  for (unsigned i = 0; i < x->getNumTemplateParameterLists(); ++i) {
    if (auto *TPL = x->getTemplateParameterList(i)) {
      for (auto *D : *TPL)
        if (!traverseDecl(D))
          return fail();
      if (auto *requiresClause = TPL->getRequiresClause())
        if (!traverseStmt(requiresClause))
          return fail();
    }
  }

  // Convert the (reified) type of the function. The syntax that was written is
  // not visited, so a decl like `auto fn(auto p)` has a type. Converting the
  // type diagnoses any type that a kernel cannot have.
  auto funcType = convertType(x->getType());
  if (!funcType)
    return fail();

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
    return std::nullopt;

  resetCurrentFunctionName();
  // At present, the bridge only lowers kernels.
  auto funcTy = cast<FunctionType>(*funcType);
  auto [func, alreadyDefined] = getOrAddFunc(loc, funcName, funcTy);
  if (alreadyDefined)
    return std::nullopt;

  LLVM_DEBUG(llvm::dbgs() << "created function: " << funcName << " : "
                          << func.getFunctionType() << '\n');
  func.setPublic();
  createEntryBlock(func, x);
  builder.setInsertionPointToEnd(&func.front());
  skipCompoundScope = true;

  // Visit the trailing requires clause, if any.
  if (const auto &trailingRequiresClause = x->getTrailingRequiresClause();
      trailingRequiresClause.ConstraintExpr)
    if (!traverseStmt(
            const_cast<clang::Expr *>(trailingRequiresClause.ConstraintExpr)))
      return fail();

  if (auto *ctor = dyn_cast<clang::CXXConstructorDecl>(x)) {
    // Constructor initializers. (The ones that were not written are visited
    // only if implicit code is.)
    for (auto *I : ctor->inits()) {
      traverse(I);
      if (hasFailed())
        return std::nullopt;
    }
  }

  bool VisitBody = x->isThisDeclarationADefinition() &&
                   (!x->isDefaulted() || shouldVisitImplicitCode());

  if (VisitBody) {
    if (!traverseStmt(x->getBody()))
      return fail();
    // Body may contain using declarations whose shadows are parented to the
    // FunctionDecl itself.
    for (auto *Child : x->decls())
      if (isa<clang::UsingShadowDecl>(Child))
        if (!traverseDecl(Child))
          return fail();
  }
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
  if (!hasTerminator(builder.getBlock())) {
    auto loc = toLocation(x);
    SmallVector<Value> dummyResults;
    for (auto ty : funcTy.getResults())
      dummyResults.push_back(cc::UndefOp::create(builder, loc, ty));
    func::ReturnOp::create(builder, loc, dummyResults);
  }
  builder.clearInsertionPoint();
  return std::nullopt;
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::visit(clang::CXXScalarValueInitExpr *x) {
  // Value initialization of a scalar, `T()`, is zero.
  auto ty = convertType(x->getType());
  if (!ty)
    return fail();
  auto loc = toLocation(x);
  if (isa<IntegerType, FloatType>(*ty))
    return value(arith::getZeroConstant(builder, loc, *ty));
  TODO_x(loc, x, mangler, "value initialization of this type");
  return fail();
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::referenceFunction(clang::FunctionDecl *x) {
  assert(builder.getBlock() && "must be generating code");
  auto loc = toLocation(x);
  auto kernName = [&]() {
    if (isKernelEntryPoint(x))
      return generateCudaqKernelName(x);
    // create a special name for 'std::move()' so we can erase it.
    if (isInNamespace(x, "std") && x->getIdentifier() && x->getName() == "move")
      return std::string(cudaq::stdMoveBuiltin);
    return cxxMangledDeclName(x);
  }();
  auto kernSym = SymbolRefAttr::get(builder.getContext(), kernName);
  auto referencedTy = convertType(x->getType());
  if (!referencedTy)
    return fail();
  auto referencedFunctionTy = peelPointerFromFunction(*referencedTy);
  if (auto f = module.lookupSymbol<func::FuncOp>(kernSym)) {
    auto fTy = f.getFunctionType();
    auto fSym = f.getSymNameAttr();
    if (referencedFunctionTy != fTy) {
      // This may be a call to an entry-point kernel. Determine if that is the
      // case, and convert this to a direct call. Otherwise, this an calling
      // convention violation.
      bool found = false;
      for (auto pair : namesMap)
        if (pair.second == kernName) {
          if (auto f = module.lookupSymbol<func::FuncOp>(pair.first)) {
            fTy = f.getFunctionType();
            fSym = f.getSymNameAttr();
            found = true;
          }
          break;
        }
      if (!found) {
        reportClangError(
            x, mangler,
            "invalid call from kernel: calling convention violation");
        return fail();
      }
    }
    return BridgeResult{func::ConstantOp::create(builder, loc, fTy, fSym)};
  }
  auto [funcOp, alreadyAdded] =
      getOrAddFunc(loc, kernName, referencedFunctionTy);
  if (!alreadyAdded)
    funcOp.setPrivate();
  return BridgeResult{func::ConstantOp::create(
      builder, loc, funcOp.getFunctionType(), funcOp.getSymNameAttr())};
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::referenceSymbol(clang::NamedDecl *x) {
  if (!builder.getBlock())
    return std::nullopt;
  if (x->getIdentifier()) {
    // 1. Look for symbol in the local scope.
    auto name = getSymbolName(x);
    if (!symbolTable.count(name)) {
      // 2. TODO: If the symbol isn't in the local scope, it is a global.
      // Don't look for a global in the module here since we do not allow
      // kernels to access globals at present.
      cudaq::emitFatalError(toLocation(x->getSourceRange()),
                            "Cannot find " + x->getNameAsString() +
                                " in the symbol table.");
    }
    return value(symbolTable.lookup(name));
  }
  return std::nullopt;
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::ParmVarDecl *x) {
  defaultVisit(x);
  if (hasFailed())
    return std::nullopt;
  // If the builder has no insertion point, then this is a prototype.
  if (!builder.getBlock())
    return std::nullopt;

  if (!x->getIdentifier()) {
    // Parameter has no name, so cannot be referenced. Skip it.
    return std::nullopt;
  }

  auto name = getSymbolName(x);
  if (symbolTable.count(name))
    return value(symbolTable.lookup(name));

  // Something has gone very wrong.
  LLVM_DEBUG(llvm::dbgs() << "parameter was not found\n"; x->dump());
  llvm::report_fatal_error(
      "parameters for the current function must already be entered in the "
      "symbol table, but this parameter wasn't found.");
}

static bool isImplicitlyGlobalStorageClass(clang::StorageClass sc) {
  switch (sc) {
  case clang::SC_Extern:
  case clang::SC_Static:
  case clang::SC_PrivateExtern:
    return true;
  default:
    return false;
  }
}

// A variable declaration may or may not have an initializer. This custom
// traversal makes sure that the type of the variable is converted, whether an
// initialization expression is present or not, and that the initializer is
// visited first. The value of the initializer is used to declare the variable.
QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::VarDecl *x) {
  std::optional<Type> declaredType;
  auto result = lowerVariable(x, declaredType);
  if (hasFailed())
    poisonVariable(x, declaredType);
  return result;
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::lowerVariable(clang::VarDecl *x,
                                  std::optional<Type> &declaredType) {
  auto storageClass = x->getStorageClass();
  if (isImplicitlyGlobalStorageClass(storageClass)) {
    reportClangError(x, mangler, "variable has invalid storage class");
    return fail();
  }
  for (unsigned i = 0; i < x->getNumTemplateParameterLists(); i++) {
    if (auto *tpl = x->getTemplateParameterList(i)) {
      for (auto *decl : *tpl)
        if (!traverseDecl(decl))
          return fail();
      if (auto *requiresClause = tpl->getRequiresClause())
        if (!traverseStmt(requiresClause))
          return fail();
    }
  }
  auto type = typeVisitor.traverse(x->getType());
  if (typeVisitor.hasFailed()) {
    typeVisitor.clearFailure();
    return fail();
  }
  if (!type) {
    // A type that has no representation in a kernel.
    TODO_x(toLocation(x->getSourceRange()), x, mangler,
           "variable of a type that has no representation in a kernel");
    return fail();
  }
  declaredType = type;
  std::optional<Value> initValue;
  if (!x->isCXXForRangeDecl())
    if (auto *init = x->getInit()) {
      Result initResult = traverse(init);
      if (hasFailed())
        return std::nullopt;
      // An initializer may or may not have a value.
      if (initResult)
        if (auto *v = std::get_if<Value>(&*initResult))
          initValue = *v;
    }
  if (typeVisitor.allowUnknownRecordType) {
    // Processing a kernel's signature. Ignore variable decls.
    return std::nullopt;
  }
  return declareVariable(x, *type, initValue);
}

void QuakeBridgeVisitor::poisonVariable(clang::VarDecl *x,
                                        std::optional<Type> declaredType) {
  // Only code that is being generated has variables, and a kernel's signature
  // has none.
  if (!builder.getBlock() || typeVisitor.allowUnknownRecordType ||
      !x->getIdentifier())
    return;
  // The value of a variable is its address. Without a type there is no type to
  // point to.
  Type pointee = declaredType ? *declaredType : builder.getNoneType();
  auto poison = cc::PoisonOp::create(builder, toLocation(x->getSourceRange()),
                                     cc::PointerType::get(pointee));
  symbolTable.insert(x->getName(), poison);
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::declareVariable(clang::VarDecl *x, Type type,
                                    std::optional<Value> init) {
  if (x->hasInit() && !x->isCXXForRangeDecl() && init) {
    LLVM_DEBUG(llvm::dbgs() << "variable " << x->getName()
                            << " has initializer of " << *init << '\n');
    type = init->getType();
  }
  LLVM_DEBUG(llvm::dbgs() << "type for variable " << x->getName() << " is "
                          << type << '\n');
  assert(type && "variable must have a valid type");
  auto loc = toLocation(x->getSourceRange());
  auto name = x->getName();
  // Variables of quantum reference types can be declared without a value for
  // the initializer. Every other variable that is initialized needs one.
  if (x->getInit() && !x->isCXXForRangeDecl() && !init &&
      !isa<cudaq::quake::VeqType, cudaq::quake::RefType>(type)) {
    TODO_x(loc, x, mangler,
           "variable that is initialized by an expression with no value");
    return fail();
  }
  if (auto qType = dyn_cast<cudaq::quake::VeqType>(type)) {
    // Variable is of !quake.veq type.
    mlir::Value qreg;
    std::size_t qregSize = qType.getSize();
    if (qregSize == 0 || (x->hasInit() && init)) {
      // This is a `qreg q(N);` or `qreg &name = exp;`
      if (!init) {
        TODO_x(loc, x, mangler, "unsized veq variable without an initializer");
        return fail();
      }
      qreg = *init;
    } else {
      // this is a qreg<N> q;
      auto qregSizeVal = mlir::arith::ConstantIntOp::create(
          builder, loc, builder.getIntegerType(64), qregSize);
      if (qregSize != 0)
        qreg = cudaq::quake::AllocaOp::create(builder, loc, qType);
      else
        qreg = cudaq::quake::AllocaOp::create(builder, loc, qType, qregSizeVal);
    }
    symbolTable.insert(name, qreg);
    // allocated_qreg_names.push_back(name);
    return BridgeResult{qreg};
  }

  if (auto qType = dyn_cast<cudaq::quake::RefType>(type)) {
    // Variable is of !quake.ref type.
    if (x->hasInit() && init) {
      symbolTable.insert(name, *init);
      return std::nullopt;
    }
    auto zero = mlir::arith::ConstantIntOp::create(
        builder, loc, builder.getIntegerType(64), 0);
    auto qregSizeOne = cudaq::quake::AllocaOp::create(
        builder, loc, cudaq::quake::VeqType::get(builder.getContext(), 1));
    Value addressTheQubit =
        cudaq::quake::ExtractRefOp::create(builder, loc, qregSizeOne, zero);
    symbolTable.insert(name, addressTheQubit);
    return BridgeResult{addressTheQubit};
  }

  if (isa<cudaq::quake::StruqType>(type)) {
    // A pure quantum struct is just passed along by value. It cannot be stored
    // to a variable.
    symbolTable.insert(name, *init);
    return std::nullopt;
  }

  if (cudaq::cc::isDevicePtr(type)) {
    symbolTable.insert(name, *init);
    return std::nullopt;
  }

  // Here we maybe have something like auto var = mz(qreg)
  if (auto vecType = dyn_cast<cc::SequenceType>(type)) {
    // Variable is of !cc.sequence type.
    if (x->getInit()) {
      // At the very least, its a vector var = vec_init;
      auto initVec = *init;
      auto elementType = vecType.getElementType();

      // For `std::vector<measure_handle>` locals, allocate a descriptor stack
      // slot and store the initializer descriptor into it. This makes the
      // variable an lvalue in the memory domain (like scalar `measure_handle`
      // locals already are), so that subsequent reassignment lowers as
      // `cc.load`/`cc.store` against a stable per-name address rather than
      // requiring SSA-renaming of the descriptor value.
      //
      // Other element types (notably `std::vector<bool>` and numeric vectors)
      // keep the existing value-domain symbol-table entry; their assignment
      // story is unchanged by this patch.
      bool isHandleVec = isa<cc::MeasureHandleType>(elementType);
      if (isHandleVec) {
        // The initializer may itself be a pointer-form lvalue (e.g. reading
        // another handle-vector local). Normalize to the descriptor value
        // before storing.
        Value initDescr = initVec;
        if (isa<cc::PointerType>(initDescr.getType()))
          initDescr = cc::LoadOp::create(builder, loc, initDescr);
        Value slot = cc::AllocaOp::create(builder, loc, vecType);
        cc::StoreOp::create(builder, loc, initDescr, slot);
        symbolTable.insert(x->getName(), slot);
        // For register-name tagging below, walk from the descriptor value
        // form (not the slot pointer) so the defining-op chain matches the
        // pre-existing logic.
        initVec = initDescr;
      } else {
        symbolTable.insert(x->getName(), initVec);
      }

      // Let's try to see if this was a auto var = mz(qreg)
      // and if so, find the mz and tag it with the variable name

      // Accept both the `bool`-valued (`std::vector<bool>` form, lowered as
      // discriminate-of-measure-interface) and the handle-valued
      // (`std::vector<measure_handle>` form, lowered as the measure
      // interface directly) shapes.
      bool isI1Bits = elementType.isIntOrFloat() &&
                      elementType.getIntOrFloatBitWidth() == 1;
      if (!isI1Bits && !isHandleVec)
        return std::nullopt;

      // Assign `registerName`
      auto attachName = [&](cudaq::quake::MeasurementInterface meas) {
        meas.setRegisterName(builder.getStringAttr(x->getName()));
      };
      if (auto descr = initVec.getDefiningOp<cudaq::quake::DiscriminateOp>()) {
        if (auto meas =
                descr.getMeasurement()
                    .getDefiningOp<cudaq::quake::MeasurementInterface>())
          attachName(meas);
      } else if (auto meas =
                     initVec
                         .getDefiningOp<cudaq::quake::MeasurementInterface>()) {
        attachName(meas);
      }

      // Did this come from a sequence init op? If not drop out
      auto stdVecInit = initVec.getDefiningOp<cc::SequenceInitOp>();
      if (!stdVecInit)
        return std::nullopt;

      // Did the first operand come from an LLVM AllocaOp, if not drop out
      auto bitVecAllocation =
          stdVecInit.getOperand(0).getDefiningOp<cc::AllocaOp>();
      if (!bitVecAllocation)
        return std::nullopt;

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
          if (auto discr = result.getDefiningOp<cudaq::quake::DiscriminateOp>())
            if (auto mzOp = discr.getMeasurement()
                                .getDefiningOp<cudaq::quake::MzOp>()) {
              // Found it, tag it with the name.
              mzOp.setRegisterName(builder.getStringAttr(x->getName()));
              break;
            }
        }
      }

      return std::nullopt;
    }
  }

  if (auto callableTy = dyn_cast<cc::CallableType>(type)) {
    // Variable is of !cc.callable type. Callables are always in the value
    // domain.
    auto callable = *init;
    symbolTable.insert(name, callable);
    return BridgeResult{callable};
  }

  // Variable is of some basic type not already handled. Create a local stack
  // slot in which to save the value. This stack slot is the variable in the
  // memory domain.
  if (!x->getInit() || x->isCXXForRangeDecl()) {
    Value alloca = cc::AllocaOp::create(builder, loc, type);
    symbolTable.insert(x->getName(), alloca);
    return BridgeResult{alloca};
  }

  // Initialization expression is present.
  auto initValue = *init;

  // If this was an `auto var = mz(q)` (or `mx`/`my`), then we want to know the
  // `var` name, as it will serve as the classical bit register name. Two
  // shapes reach here: `bool b = mz(q);` and `auto h = mz(q);` Both route the
  // same register name through to the underlying measurement op via
  // `MeasurementInterface`.
  auto attachName = [&](cudaq::quake::MeasurementInterface meas) {
    meas.setRegisterName(builder.getStringAttr(x->getName()));
  };
  if (auto discr = initValue.getDefiningOp<cudaq::quake::DiscriminateOp>()) {
    if (auto meas = discr.getMeasurement()
                        .getDefiningOp<cudaq::quake::MeasurementInterface>())
      attachName(meas);
  } else if (auto meas =
                 initValue
                     .getDefiningOp<cudaq::quake::MeasurementInterface>()) {
    attachName(meas);
  }

  assert(initValue && "initializer value must be lowered");
  if (isa<IntegerType>(initValue.getType()) && isa<IntegerType>(type)) {
    if (initValue.getType().getIntOrFloatBitWidth() <
        type.getIntOrFloatBitWidth()) {
      // FIXME: Use zero-extend if this is unsigned!
      initValue = cudaq::cc::CastOp::create(builder, loc, type, initValue,
                                            cudaq::cc::CastOpMode::Signed);
    } else if (initValue.getType().getIntOrFloatBitWidth() >
               type.getIntOrFloatBitWidth()) {
      initValue = cudaq::cc::CastOp::create(builder, loc, type, initValue);
    }
  } else if (isa<IntegerType>(initValue.getType()) && isa<FloatType>(type)) {
    // FIXME: Use UIToFP if this is unsigned!
    initValue = cudaq::cc::CastOp::create(builder, loc, type, initValue,
                                          cudaq::cc::CastOpMode::Signed);
  }

  // The variable is the object that the initialization expression left in
  // memory, unless the variable is a pointer. A pointer variable is not the
  // object that it points to, it has storage of its own for the address.
  if (auto initObject = initValue.getDefiningOp<cc::AllocaOp>();
      initObject && !x->getType()->isPointerType()) {
    // Initialization expression already left an object in memory. This could be
    // because an object was constructed. TODO: this needs to also handle the
    // case that an object must be cloned instead of casted.
    assert(type == initObject.getType());
    symbolTable.insert(x->getName(), initValue);
    return BridgeResult{initValue};
  }
  auto qualTy = x->getType().getCanonicalType();
  auto isSequenceBoolReference = [&](clang::QualType &qualTy) {
    if (auto *recTy = dyn_cast<clang::RecordType>(qualTy.getTypePtr())) {
      auto *recDecl = recTy->getDecl();
      if (isInNamespace(recDecl, "std")) {
        auto name = recDecl->getNameAsString();
        return name == "_Bit_reference" || name == "__bit_reference" ||
               name == "__bit_const_reference";
      }
    }
    return false;
  };
  if (isSequenceBoolReference(qualTy) ||
      qualTy.getTypePtr()->isReferenceType()) {
    // A similar case is when the C++ variable is a reference to a subobject.
    assert(isa<cc::PointerType>(type));
    Value cast = cc::CastOp::create(builder, loc, type, initValue);
    symbolTable.insert(x->getName(), cast);
    return BridgeResult{cast};
  }

  // Don't allocate memory for a quantum or value-semantic struct.
  if (auto insertValOp = initValue.getDefiningOp<cc::InsertValueOp>()) {
    symbolTable.insert(x->getName(), initValue);
    return BridgeResult{initValue};
  }

  // Initialization expression resulted in a value. Create a variable and save
  // that value to the variable's memory address.
  Value alloca = cc::AllocaOp::create(builder, loc, type);
  cc::StoreOp::create(builder, loc, initValue, alloca);
  symbolTable.insert(x->getName(), alloca);
  return BridgeResult{alloca};
}

} // namespace cudaq::detail
