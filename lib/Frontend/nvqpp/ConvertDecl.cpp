/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Todo.h"

#define DEBUG_TYPE "lower-ast-decl"

using namespace mlir;

namespace cudaq::details {

bool QuakeBridgeVisitor::needToLowerFunction(const clang::FunctionDecl *decl) {
  if (!decl->hasBody())
    return false;
  // Check if this is a kernel entry point.
  for (auto fdPair : functionsToEmit) {
    if (decl == fdPair.second) {
      // This is an entry point.
      std::string entryName = generateCudaqKernelName(fdPair);
      setEntryName(entryName);
      // Extend the mangled kernel names map.
      auto mangledFuncName = cxxMangledDeclName(decl);
      namesMap.insert({entryName, mangledFuncName});
      valueStack.clear();
      return true;
    }
  }

  if (LOWERING_TRANSITIVE_CLOSURE) {
    // Not a kernel entry point. Test to see if it is some other function we
    // need to lower.
    for (auto *rf : reachableFunctions) {
      if ((decl == rf) && decl->getBody()) {
        // Create the function and set the builder.
        auto mangledFuncName = cxxMangledDeclName(decl);
        setCurrentFunctionName(mangledFuncName);
        valueStack.clear();
        return true;
      }
    }
  }

  // Skip this function. It is not part of the call graph of QPU code.
  return false;
}

bool QuakeBridgeVisitor::TraverseCXXDeductionGuideDecl(
    clang::CXXDeductionGuideDecl *x) {
  postOrderTraversal = false;
  if (!Base::TraverseCXXDeductionGuideDecl(x))
    return true;
  if (!hasTerminator(builder.getBlock()))
    builder.create<func::ReturnOp>(toLocation(x));
  return true;
}

bool QuakeBridgeVisitor::TraverseCXXMethodDecl(clang::CXXMethodDecl *x) {
  if (typeMode) {
    // Use the resolved type, not the syntactic type.
    return TraverseType(x->getType());
  }
  codeGenMethodDecl = true;
  postOrderTraversal = false;
  if (!Base::TraverseCXXMethodDecl(x)) {
    if (raisedError && x->getParent()->isLambda()) {
      auto &de = astContext->getDiagnostics();
      const auto id =
          de.getCustomDiagID(clang::DiagnosticsEngine::Remark,
                             "An inaccessible symbol in a callable expression "
                             "may be from an implicit capture of a variable "
                             "that is not present in a kernel marked __qpu__.");
      auto db = de.Report(x->getBeginLoc(), id);
      const auto range = x->getSourceRange();
      db.AddSourceRange(clang::CharSourceRange::getCharRange(range));
      raisedError = false;
    }
    return true;
  }
  if (!hasTerminator(builder.getBlock()))
    builder.create<func::ReturnOp>(toLocation(x));
  return true;
}

bool QuakeBridgeVisitor::TraverseFunctionDecl(clang::FunctionDecl *x) {
  postOrderTraversal = false;
  if (!Base::TraverseFunctionDecl(x))
    return true;
  if (!typeMode && !hasTerminator(builder.getBlock()))
    builder.create<func::ReturnOp>(toLocation(x));
  return true;
}

bool QuakeBridgeVisitor::WalkUpFromFunctionDecl(clang::FunctionDecl *x) {
  // Check if this function is one we want to lower to MLIR. If it is, enable
  // lowering and visit this node.
  if (needToLowerFunction(x)) {
    LLVM_DEBUG(llvm::dbgs() << "found function to lower: "
                            << x->getQualifiedNameAsString() << '\n');
    return VisitFunctionDecl(x);
  }
  // Otherwise, we can skip the visit entirely.
  return true;
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
      if (isa<cc::CallableType, cc::StdvecType, cc::ArrayType, cc::StructType,
              LLVM::LLVMStructType, FunctionType, quake::RefType,
              quake::VeqType>(parmTy)) {
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
      return {func, true};
    }
  } else {
    OpBuilder build(module.getBodyRegion());
    OpBuilder::InsertionGuard guard(build);
    build.setInsertionPointToEnd(module.getBody());
    SmallVector<NamedAttribute> attrs;
    func = build.create<func::FuncOp>(loc, funcName, funcTy, attrs);
    func.setPrivate();
  }
  return {func, false};
}

bool QuakeBridgeVisitor::VisitFunctionDecl(clang::FunctionDecl *x) {
  if (typeMode)
    return true;
  // If this function is a function template and not the specialization of the
  // function template, we skip it. We only want to lower template functions
  // that have their types resolved.
  if (x->getDescribedFunctionTemplate() &&
      !x->isFunctionTemplateSpecialization())
    return true;
  // Create the function and set the builder.
  auto funcName = getCurrentFunctionName();
  auto loc = toLocation(x);
  assert(!funcName.empty() && "function name must not be empty");
  resetCurrentFunctionName();
  // At present, the bridge only lowers kernels.
  auto optFuncTy = getFunctionType(x, /*isKernel=*/true);
  if (!optFuncTy)
    return false;
  auto [func, alreadyDefined] = getOrAddFunc(loc, funcName, *optFuncTy);
  if (alreadyDefined) {
    // Already lowered function func, skip it. Returning false here skips the
    // post-order traversal of the function declaration.
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "created function: " << funcName << " : "
                          << func.getFunctionType() << '\n');
  func.setPublic();
  createEntryBlock(func, x);
  builder.setInsertionPointToEnd(&func.front());
  skipCompoundScope = true;
  return true;
}

bool QuakeBridgeVisitor::VisitNamedDecl(clang::NamedDecl *x) {
  if (typeMode)
    return true;
  if (!symbolTable.count(x->getName())) {
    cudaq::emitFatalError(toLocation(x->getSourceRange()),
                          "Cannot find " + x->getNameAsString() +
                              " in the symbol table.");
  }
  return pushValue(symbolTable.lookup(x->getName()));
}

bool QuakeBridgeVisitor::VisitParmVarDecl(clang::ParmVarDecl *x) {
  if (typeMode)
    return true;
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
  x->dump();
  llvm::report_fatal_error(
      "parameters for the current function must already be entered in the "
      "symbol table, but this parameter wasn't found.");
}

bool QuakeBridgeVisitor::VisitVarDecl(clang::VarDecl *x) {
  if (typeMode)
    return true;
  auto type = genType(x->getType());
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
          auto mzOp = result.getDefiningOp<quake::MzOp>();
          if (mzOp) {
            // Found it, tag it with the name.
            mzOp->setAttr("registerName", builder.getStringAttr(x->getName()));
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
  if (!x->getInit()) {
    Value alloca = builder.create<cc::AllocaOp>(loc, type);
    symbolTable.insert(x->getName(), alloca);
    return pushValue(alloca);
  }

  // Initialization expression is present.
  auto initValue = popValue();

  // If this was an auto var = mz(q), then we want to know the
  // var name, as it will serve as the classical bit register name
  if (auto mz = initValue.getDefiningOp<quake::MzOp>())
    mz->setAttr("registerName", builder.getStringAttr(x->getName()));

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
    // Initialization expression already left an object in memory. This should
    // be because an object was constructed. Just cast the memory address of the
    // object to the expected type. TODO: this needs to also handle the case
    // that an object must be cloned instread of casted.
    Value cast =
        builder.create<cc::CastOp>(loc, cc::PointerType::get(type), initObject);
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
