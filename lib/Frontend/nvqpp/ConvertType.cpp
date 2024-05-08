/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
#include "clang/Basic/TargetInfo.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "lower-ast-type"

using namespace mlir;

static bool isArithmeticType(Type t) {
  return isa<IntegerType, FloatType, ComplexType>(t);
}

/// Is \p t a quantum reference type. In the bridge, quantum types are always
/// reference types.
static bool isQuantumType(Type t) {
  return isa<quake::VeqType, quake::RefType>(t);
}

/// Allow `array of [array of]* T`, where `T` is arithmetic.
static bool isStaticArithmeticSequenceType(Type t) {
  if (auto vec = dyn_cast<cudaq::cc::ArrayType>(t)) {
    auto eleTy = vec.getElementType();
    return isArithmeticType(eleTy) || isStaticArithmeticSequenceType(eleTy);
  }
  return false;
}

/// Returns true if and only if \p t is a struct of arithmetic, static sequence
/// of arithmetic (i.e., it has a constant length), or (recursive) struct of
/// arithmetic on all members.
static bool isStaticArithmeticProductType(Type t) {
  if (auto structTy = dyn_cast<cudaq::cc::StructType>(t)) {
    for (auto memTy : structTy.getMembers()) {
      if (isArithmeticType(memTy) || isStaticArithmeticSequenceType(memTy) ||
          isStaticArithmeticProductType(memTy))
        continue;
      return false;
    }
    return true;
  }
  return false;
}

/// Is \p t a recursive sequence of arithmetic types? The outer types are
/// allowed to be dynamic (vector), but the inner types must be static. An outer
/// type can only be a vector.
///
/// Return true if and only if \p t is
///    - `vector of [vector of]* [array of]* T` or
///    - `array of [array of]* T`,
/// where `T` is an arithmetic type or static product type of arithmetic types.
static bool isArithmeticSequenceType(Type t) {
  if (auto vec = dyn_cast<cudaq::cc::SpanLikeType>(t)) {
    auto eleTy = vec.getElementType();
    return isArithmeticType(eleTy) || isStaticArithmeticProductType(eleTy) ||
           isArithmeticSequenceType(eleTy);
  }
  return isStaticArithmeticSequenceType(t);
}

static bool isRecursiveArithmeticProductType(Type t);

/// Is \p t a recursive sequence of arithmetic types? This is a similar but more
/// relaxed test than isArithmeticSequenceType in that the outer types may
/// include product types and are not restricted to vectors. Only ArrayType is
/// considered an inner type.
static bool isRecursiveArithmeticSequenceType(Type t) {
  if (auto vec = dyn_cast<cudaq::cc::SpanLikeType>(t)) {
    auto eleTy = vec.getElementType();
    return isArithmeticType(eleTy) || isRecursiveArithmeticProductType(eleTy) ||
           isRecursiveArithmeticSequenceType(eleTy);
  }
  return isStaticArithmeticSequenceType(t);
}

/// Is \p t a recursive product of possibly dynamic arithmetic types? Returns
/// true if and only if \p t is a struct with members that are arithmetic,
/// dynamic sequences of arithmetic, or (recursively) products of possible
/// dynamic products of arithmetic types.
static bool isRecursiveArithmeticProductType(Type t) {
  if (auto structTy = dyn_cast<cudaq::cc::StructType>(t)) {
    for (auto memTy : structTy.getMembers()) {
      if (isArithmeticType(memTy) || isRecursiveArithmeticSequenceType(memTy) ||
          isRecursiveArithmeticProductType(memTy))
        continue;
      return false;
    }
    return true;
  }
  return isStaticArithmeticProductType(t);
}

/// Is \p t a recursively arithmetic type? This tests either for struct of
/// vector or vector of struct like arithmetic composed types.
///
/// Returns true if and only if \p t is
///    - a sequence of `T` such that `T` is composed of AT
///    - a product of `T`, `U`, ... such that all types are composed of AT
/// where AT is a recursively built type with leaves that are arithmetic.
static bool isComposedArithmeticType(Type t) {
  return isRecursiveArithmeticProductType(t) ||
         isRecursiveArithmeticSequenceType(t);
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

/// Return true if and only if \p t is a (simple) arithmetic type, an arithmetic
/// sequence type (possibly dynamic in length), or a static product type of
/// arithmetic types. Note that this means a product type with a dynamic
/// sequence of arithmetic types is \em disallowed.
static bool isKernelResultType(Type t) {
  return isArithmeticType(t) || isArithmeticSequenceType(t) ||
         isStaticArithmeticProductType(t);
}

/// Return true if and only if \p t is a (simple) arithmetic type, an possibly
/// dynamic type composed of arithmetic types, a quantum type, a callable
/// (function), or a string.
static bool isKernelArgumentType(Type t) {
  return isArithmeticType(t) || isComposedArithmeticType(t) ||
         isQuantumType(t) || isKernelCallable(t) || isFunctionCallable(t) ||
         // TODO: move from pointers to a builtin string type.
         cudaq::isCharPointerType(t);
}

static bool isKernelSignatureType(FunctionType t) {
  for (auto t : t.getInputs()) {
    // Assumes a class (cc::StructType) is callable. Must pass in the AST
    // parameter to verify the assumption.
    if (isKernelArgumentType(t) || isa<cudaq::cc::StructType>(t))
      continue;
    return false;
  }
  for (auto t : t.getResults())
    if (!isKernelResultType(t))
      return false;
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
    if (allowUnknownRecordType) {
      // This is a kernel's type signature, so add a NoneType. When finally
      // returning out of determining the kernel's type signature, a clang error
      // diagnsotic will be reported.
      pushType(noneTy);
    } else if (typeStack.size() != typeStackDepth) {
      emitWarning(toLocation(recDecl),
                  "compiler encountered type traversal issue");
      return false;
    } else {
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
  StringRef name;
  if (auto ident = x->getIdentifier())
    name = ident->getName();
  auto *ctx = builder.getContext();
  if (!x->getDefinition())
    return pushType(cc::StructType::get(ctx, name, /*isOpaque=*/true));
  SmallVector<Type> fieldTys =
      lastTypes(std::distance(x->field_begin(), x->field_end()));
  auto [width, alignInBytes] = [&]() -> std::pair<std::uint64_t, unsigned> {
    auto *defn = x->getDefinition();
    assert(defn && "struct must be defined here");
    auto ti = getContext()->getTypeInfo(defn->getTypeForDecl());
    return {ti.Width, llvm::PowerOf2Ceil(ti.Align) / 8};
  }();
  if (name.empty())
    return pushType(cc::StructType::get(ctx, fieldTys, width, alignInBytes));
  return pushType(
      cc::StructType::get(ctx, name, fieldTys, width, alignInBytes));
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
  case BuiltinType::LongDouble: {
    auto bitWidth = astContext->getTargetInfo().getLongDoubleWidth();
    if (bitWidth == 64)
      return builder.getF64Type();
    llvm::Triple triple(astContext->getTargetInfo().getTargetOpts().Triple);
    if (triple.isX86())
      return builder.getF80Type();
    return builder.getF128Type();
  }
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
  if (isa<cc::CallableType, cc::SpanLikeType, quake::VeqType, quake::RefType>(
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
  if (isa<cc::CallableType, cc::SpanLikeType, cc::ArrayType, cc::StructType,
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

static bool isReferenceToCudaqStateType(Type t) {
  if (auto ptrTy = dyn_cast<cc::PointerType>(t))
    return isCudaqStateType(ptrTy.getElementType());
  return false;
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
    if (isKernelArgumentType(t) || isReferenceToCallableRecord(t, p) ||
        isReferenceToCudaqStateType(t))
      continue;
    reportClangError(p, mangler, "kernel argument type not supported");
    return false;
  }
  for (auto t : funcTy.getResults()) {
    if (isKernelResultType(t))
      continue;
    reportClangError(x, mangler, "kernel result type not supported");
    return false;
  }
  return true;
}

} // namespace cudaq::details
