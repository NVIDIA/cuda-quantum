/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Todo.h"

#define DEBUG_TYPE "lower-ast-type"

using namespace mlir;

static bool isArithmeticType(Type t) { return isa<IntegerType, FloatType>(t); }

static bool isQuantumType(Type t) {
  return isa<quake::VeqType, quake::RefType>(t);
}

static bool isArithmeticSequenceType(Type t) {
  if (auto vec = dyn_cast<cudaq::cc::StdvecType>(t))
    return isArithmeticType(vec.getElementType());
  if (auto vec = dyn_cast<cudaq::cc::ArrayType>(t))
    return isArithmeticType(vec.getElementType());
  return false;
}

static bool isKernelSignatureType(FunctionType t);

static bool isKernelCallable(Type t) {
  if (auto lambdaTy = dyn_cast<cudaq::cc::CallableType>(t))
    return isKernelSignatureType(lambdaTy.getSignature());
  return false;
}

static bool isFunctionCallable(Type t) {
  if (auto funcTy = dyn_cast<FunctionType>(t))
    return isKernelSignatureType(funcTy);
  return false;
}

static bool isKernelSignatureType(FunctionType t) {
  for (auto t : t.getInputs()) {
    if (isArithmeticType(t) || isArithmeticSequenceType(t) ||
        isQuantumType(t) || isKernelCallable(t) || isFunctionCallable(t) ||
        isa<cudaq::cc::StructType>(t)) {
      // Assume a class (LLVMStructType) is callable.
      continue;
    }
    return false;
  }
  for (auto t : t.getResults()) {
    if (isArithmeticType(t) || isArithmeticSequenceType(t))
      continue;
    return false;
  }
  return true;
}

static bool isReferenceToCallableRecord(Type t, clang::ParmVarDecl *arg) {
  // TODO: add check that the Decl is, in fact, a callable with a legal kernel
  // signature.
  return isa<cudaq::cc::StructType>(t);
}

namespace cudaq::details {

clang::FunctionDecl *
QuakeBridgeVisitor::findCallOperator(const clang::CXXRecordDecl *decl) {
  for (auto *m : decl->methods())
    if (m->isOverloadedOperator() &&
        cudaq::isCallOperator(m->getOverloadedOperator()))
      return m->getDefinition();
  return nullptr;
}

bool QuakeBridgeVisitor::TraverseRecordType(clang::RecordType *t) {
  auto *recDecl = t->getDecl();

  if (ignoredClass(recDecl))
    return true;
  auto reci = records.find(t);
  if (reci != records.end()) {
    pushType(reci->second);
    return true;
  }
  auto noneTy = builder.getNoneType();
  records.insert({t, noneTy});
  bool saveInRecType = inRecType;
  inRecType = true;
  auto typeStackDepth = typeStack.size();
  bool result;
  if (recDecl->isLambda()) {
    result = TraverseCXXRecordDecl(cast<clang::CXXRecordDecl>(recDecl));
  } else {
    result = TraverseDecl(recDecl);
  }
  inRecType = saveInRecType;
  if (!result)
    return false;
  if (typeStack.size() != typeStackDepth + 1) {
    assert(typeStack.size() == typeStackDepth);
    if (allowUnknownRecordType)
      pushType(noneTy);
    else {
      recDecl->dump();
      emitFatalError(toLocation(recDecl), "expected a type");
    }
  }
  records[t] = peekType();
  return true;
}

bool QuakeBridgeVisitor::VisitRecordDecl(clang::RecordDecl *x) {
  assert(!x->isLambda() && "expected lambda to be handled in traverse");
  // Note that we're generating a Type on the type stack.
  SmallVector<Type> fieldTys =
      lastTypes(std::distance(x->field_begin(), x->field_end()));
  StringRef name;
  if (auto ident = x->getIdentifier())
    name = ident->getName();
  auto *ctx = builder.getContext();
  if (name.empty())
    return pushType(cc::StructType::get(ctx, fieldTys));
  return pushType(cc::StructType::get(ctx, name, fieldTys));
}

bool QuakeBridgeVisitor::VisitFunctionProtoType(clang::FunctionProtoType *t) {
  assert(t->exceptions().empty() &&
         "exceptions are not supported in CUDA Quantum");
  if (t->getNoexceptExpr()) {
    // Throw away the boolean value from this clause.
    // TODO: Could enforce that it must be `true`.
    popValue();
  }
  SmallVector<Type> argTys = lastTypes(t->param_types().size());
  SmallVector<Type> resTys;
  auto funcRetTy = popType();
  if (!isa<NoneType>(funcRetTy))
    resTys.push_back(funcRetTy);
  return pushType(builder.getFunctionType(argTys, resTys));
}

/// Parallels the clang conversion from `clang::Type` to `llvm::Type`. In this
/// case, we translate `clang::Type` to `mlir::Type`. See
/// `clang::CodeGenTypes.ConvertType`.
Type QuakeBridgeVisitor::builtinTypeToType(const clang::BuiltinType *t) {
  using namespace clang;
  switch (t->getKind()) {
  case BuiltinType::Void:
    return builder.getNoneType();
  case BuiltinType::Bool:
    return builder.getI1Type();
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Int:
  case BuiltinType::UInt:
  case BuiltinType::Long:
  case BuiltinType::ULong:
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
  case BuiltinType::Char8:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::ShortAccum:
  case BuiltinType::Accum:
  case BuiltinType::LongAccum:
  case BuiltinType::UShortAccum:
  case BuiltinType::UAccum:
  case BuiltinType::ULongAccum:
  case BuiltinType::ShortFract:
  case BuiltinType::Fract:
  case BuiltinType::LongFract:
  case BuiltinType::UShortFract:
  case BuiltinType::UFract:
  case BuiltinType::ULongFract:
  case BuiltinType::SatShortAccum:
  case BuiltinType::SatAccum:
  case BuiltinType::SatLongAccum:
  case BuiltinType::SatUShortAccum:
  case BuiltinType::SatUAccum:
  case BuiltinType::SatULongAccum:
  case BuiltinType::SatShortFract:
  case BuiltinType::SatFract:
  case BuiltinType::SatLongFract:
  case BuiltinType::SatUShortFract:
  case BuiltinType::SatUFract:
  case BuiltinType::SatULongFract:
    return builder.getIntegerType(astContext->getTypeSize(t));
  case BuiltinType::Float16:
  case BuiltinType::Half:
    return builder.getF16Type();
  case BuiltinType::BFloat16:
    return builder.getBF16Type();
  case BuiltinType::Float:
    return builder.getF32Type();
  case BuiltinType::Double:
    return builder.getF64Type();
  case BuiltinType::LongDouble:
    return astContext->getTypeSize(t) == 64 ? builder.getF64Type()
                                            : builder.getF128Type();
  case BuiltinType::Float128:
  case BuiltinType::Ibm128: /* double double format -> {double, double} */
    return builder.getF128Type();
  case BuiltinType::NullPtr:
    return cc::PointerType::get(builder.getContext());
  case BuiltinType::UInt128:
  case BuiltinType::Int128:
    return builder.getIntegerType(128);
  default:
    LLVM_DEBUG(llvm::dbgs() << "builtin type not handled: "; t->dump());
    TODO("builtin type");
  }
}

bool QuakeBridgeVisitor::VisitBuiltinType(clang::BuiltinType *t) {
  return pushType(builtinTypeToType(t));
}

bool QuakeBridgeVisitor::VisitPointerType(clang::PointerType *t) {
  if (t->getPointeeType()->isUndeducedAutoType())
    return pushType(cc::PointerType::get(builder.getContext()));
  return pushType(cc::PointerType::get(popType()));
}

bool QuakeBridgeVisitor::VisitLValueReferenceType(
    clang::LValueReferenceType *t) {
  if (t->getPointeeType()->isUndeducedAutoType())
    return pushType(cc::PointerType::get(builder.getContext()));
  auto eleTy = popType();
  if (isa<cc::CallableType, cc::StdvecType, quake::VeqType, quake::RefType>(
          eleTy))
    return pushType(eleTy);
  return pushType(cc::PointerType::get(eleTy));
}

bool QuakeBridgeVisitor::VisitRValueReferenceType(
    clang::RValueReferenceType *t) {
  if (t->getPointeeType()->isUndeducedAutoType())
    return pushType(cc::PointerType::get(builder.getContext()));
  auto eleTy = popType();
  // FIXME: LLVMStructType is promoted as a temporary workaround.
  if (isa<cc::CallableType, cc::StdvecType, cc::ArrayType, cc::StructType,
          quake::VeqType, quake::RefType, LLVM::LLVMStructType>(eleTy))
    return pushType(eleTy);
  return pushType(cc::PointerType::get(eleTy));
}

bool QuakeBridgeVisitor::VisitConstantArrayType(clang::ConstantArrayType *t) {
  auto size = t->getSize().getZExtValue();
  return pushType(cc::ArrayType::get(builder.getContext(), popType(), size));
}

bool QuakeBridgeVisitor::pushType(Type t) {
  LLVM_DEBUG(llvm::dbgs() << std::string(typeStack.size(), ' ') << "push " << t
                          << '\n');
  typeStack.push_back(t);
  return true;
}

Type QuakeBridgeVisitor::popType() {
  assert(!typeStack.empty());
  Type result = peekType();
  LLVM_DEBUG(llvm::dbgs() << std::string(typeStack.size() - 1, ' ') << "(pop "
                          << result << ")\n");
  typeStack.pop_back();
  return result;
}

/// Return the last `n` types from the stack in left-to-right (natural)
/// order. For a signature, `f(T, U, V)` this can be used to return a list
/// `[type_T type_U type_V]`.
SmallVector<Type> QuakeBridgeVisitor::lastTypes(unsigned n) {
  assert(n <= typeStack.size() && "stack has fewer types than requested");
  SmallVector<Type> result(typeStack.end() - n, typeStack.end());
  LLVM_DEBUG(llvm::dbgs() << std::string(typeStack.size() - n, ' ') << "(pop <"
                          << n << ">)\n");
  typeStack.pop_back_n(n);
  return result;
}

// Do syntax checking on the signature of kernel \p x.
// Precondition: the top of the type stack is the kernel's `mlir::FunctionType`.
// Return true if and only if the kernel \p x has a legal signature.
bool QuakeBridgeVisitor::doSyntaxChecks(const clang::FunctionDecl *x) {
  auto funcTy = cast<FunctionType>(peekType());
  auto astTy = x->getType();
  // Verify the argument and return types are valid for a kernel.
  auto *protoTy = dyn_cast<clang::FunctionProtoType>(astTy.getTypePtr());
  if (!protoTy) {
    reportClangError(x, mangler, "kernel must have a prototype");
    return false;
  }
  if (protoTy->getNumParams() != funcTy.getNumInputs()) {
    // The arity of the function doesn't match, so report an error.
    reportClangError(x, mangler, "kernel has unexpected arguments");
    return false;
  }
  for (auto [t, p] : llvm::zip(funcTy.getInputs(), x->parameters())) {
    // Structs, lambdas, functions are valid callable objects. Also pure
    // device kernels may take veq and/or ref arguments.
    if (isArithmeticType(t) || isArithmeticSequenceType(t) ||
        isQuantumType(t) || isKernelCallable(t) || isFunctionCallable(t) ||
        isCharPointerType(t) || isReferenceToCallableRecord(t, p))
      continue;
    reportClangError(p, mangler, "kernel argument type not supported");
    return false;
  }
  for (auto t : funcTy.getResults()) {
    if (isArithmeticType(t) || isArithmeticSequenceType(t))
      continue;
    reportClangError(x, mangler, "kernel result type not supported");
    return false;
  }
  return true;
}

} // namespace cudaq::details
