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

static std::size_t getQregArraySize(clang::QualType type) {
  if (auto templateSpecialization =
          type->getAs<clang::TemplateSpecializationType>()) {
    auto templArgs = templateSpecialization->template_arguments();
    if (templArgs.size()) {
      // should be the first one
      auto &arg = templArgs[0];
      if (auto constantExpr = dyn_cast<clang::ConstantExpr>(arg.getAsExpr())) {
        auto value =
            constantExpr->getAPValueResult().getInt().getLimitedValue();
        return value < std::numeric_limits<std::size_t>::max() ? value : 0;
      }
    }
  }
  return 0;
}

// Does `typeName` match the string `match` with or without the namespace name
// `nsPrefix`? If `toEnd` is true, then require an exact match, otherwise
// match a prefix up to its template argument list.
static bool matchTypeName(std::string nsPrefix, std::string match,
                          StringRef typeName, bool toEnd = false) {
  // Does the `typeName` start with "`nsPrefix` ::"?
  if (typeName.startswith(nsPrefix + "::"))
    typeName = typeName.substr(nsPrefix.size() + 2);
  if (toEnd)
    return typeName == match;

  // If `toEnd` is `false` then we expect a template type, so `typeName`
  // must end with "< `type-args-list` >".
  if (!typeName.startswith(match + "<"))
    return false;

  // `typeName` starts with the right prefix, so walk over the template
  // arguments text, pairing up any brackets as we encounter them. We
  // should reach the closing `>` (or the end of `typeName`). At that
  // point, a match is successful if and only if the end of the name is
  // the `>`. Otherwise, the match is a prefix to some other type.
  typeName = typeName.substr(match.size() + 1);
  SmallVector<char> nestedBrackets;
  std::size_t length = typeName.size();
  auto verifyNestedBracket = [&](char ch) {
    char token = nestedBrackets.back();
    nestedBrackets.pop_back();
    return token == ch;
  };
  while (length) {
    char cursor = typeName[0];
    switch (cursor) {
    case '<':
    case '(':
    case '[':
    case '{':
      nestedBrackets.push_back(cursor);
      break;
    case '>':
      if (nestedBrackets.empty())
        return typeName == ">";
      if (!verifyNestedBracket('<'))
        return false;
      break;
    case '}':
      if (!verifyNestedBracket('{'))
        return false;
      break;
    case ')':
      if (!verifyNestedBracket('('))
        return false;
      break;
    case ']':
      if (!verifyNestedBracket('['))
        return false;
      break;
    default:
      break;
    }
    typeName = typeName.substr(1);
    length--;
  }
  return false;
}

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
  if (auto callableTy = dyn_cast<cudaq::cc::CallableType>(t))
    return isKernelSignatureType(callableTy.getSignature());
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

bool QuakeBridgeVisitor::TraverseTypeLoc(clang::TypeLoc t) {
  if (codeGenMethodDecl) {
    // FIXME: hack to avoid bugs when visiting a method.
    codeGenMethodDecl = false;
    return true;
  }
  bool saveTypeMode = typeMode;
  typeMode = true;
  bool result = Base::TraverseTypeLoc(t);
  typeMode = saveTypeMode;
  return result;
}

// Force traversal through a TypedefType. Without this, typedefs like
// `std::size_t` don't recurse down to the builtin type.
bool QuakeBridgeVisitor::TraverseTypedefType(clang::TypedefType *t) {
  return TraverseType(t->desugar());
}

// Force traversal through a UsingType.
bool QuakeBridgeVisitor::TraverseUsingType(clang::UsingType *t) {
  return TraverseType(t->desugar());
}

bool QuakeBridgeVisitor::TraverseRecordType(clang::RecordType *t) {
  auto *recDecl = t->getDecl();
  if (recDecl->isLambda()) {
    // Traverse implicit code will traverse inline callables.
    visitImplicitCode = t->getDecl()->isLambda();
    return TraverseCXXRecordDecl(cast<clang::CXXRecordDecl>(recDecl));
  }
  if (std::find(records.begin(), records.end(), t) != records.end())
    return true;
  records.push_back(t);
  if (isa<clang::ClassTemplateSpecializationDecl>(recDecl))
    visitTemplateInstantiations = true;
  auto result = TraverseDecl(recDecl);
  records.pop_back();
  return result;
}

bool QuakeBridgeVisitor::convertToCallable() {
  // Should be called from a Visit...() method.
  if (isa<cudaq::cc::CallableType>(peekType()))
    return true;
  auto funcTy = dyn_cast<FunctionType>(popType());
  assert(funcTy &&
         "callable expression should be convertible from function type");
  return pushType(cudaq::cc::CallableType::get(builder.getContext(), funcTy));
}

bool QuakeBridgeVisitor::VisitCXXRecordDecl(clang::CXXRecordDecl *x) {
  if (x->isLambda())
    return convertToCallable();
  return true;
}

bool QuakeBridgeVisitor::VisitRecordDecl(clang::RecordDecl *x) {
  auto *ctx = builder.getContext();
  if (x->isLambda())
    return convertToCallable();
  auto name = x->getIdentifier()->getName();
  if (cudaq::isInNamespace(x, "cudaq")) {
    // Types from the `cudaq` namespace.
    if (name.equals("qudit") || name.equals("qubit"))
      return pushType(quake::RefType::get(ctx));
    if (name.equals("qspan") || name.equals("qreg"))
      return pushType(quake::VeqType::getUnsized(ctx));
    if (name.equals("ctrl"))
      return true;
    if (name.equals("qarray") || name.equals("qvector") ||
        name.equals("qview")) {
      reportClangError(x, mangler, "type is not yet implemented");
      return pushType(NoneType::get(ctx));
    }
    TODO("unhandled type, " + name + ", in cudaq namespace");
  }
  if (cudaq::isInNamespace(x, "std")) {
    if (name.equals("vector"))
      return pushType(cudaq::cc::StdvecType::get(ctx, popType()));
    // std::vector<bool>   =>   cc.stdvec<i1>
    if (name.equals("_Bit_reference")) {
      // Reference to a bit in a std::vector<bool>. Promote to a value.
      return pushType(builder.getI1Type());
    }
    LLVM_DEBUG(llvm::dbgs()
               << "in std namespace, " << name << " is not matched\n");
  }
  auto isLambda = [&]() {
    if (auto *cxxRec = dyn_cast<clang::CXXRecordDecl>(x))
      for (auto *meth : cxxRec->methods())
        if (meth->getOverloadedOperator() ==
            clang::OverloadedOperatorKind::OO_Call)
          return true;
    return false;
  };
  auto fieldCount = [&]() {
    return std::count_if(x->field_begin(), x->field_end(),
                         [](auto) { return true; });
  };
  if (isLambda()) {
    // This is not a callable CXXRecord, but it is a callable. So there will be
    // a function signature on the stack here that needs to be popped off.
    [[maybe_unused]] auto ty = popType();
    assert(isa<FunctionType>(ty) && fieldCount() == 0);
  }
  SmallVector<Type> fieldTys = lastTypes(fieldCount());
  if (name.empty())
    return pushType(cc::StructType::get(ctx, fieldTys));
  return pushType(cc::StructType::get(ctx, name, fieldTys));
}

bool QuakeBridgeVisitor::VisitFunctionProtoType(clang::FunctionProtoType *x) {
  if (!typeMode)
    return true;
  assert(x->exceptions().size() == 0 &&
         "exceptions are not supported in CUDA Quantum");
  SmallVector<Type> argTys = lastTypes(x->param_types().size());
  SmallVector<Type> resTys;
  auto funcRetTy = popType();
  if (!x->getReturnType().getTypePtr()->isVoidType())
    resTys.push_back(funcRetTy);
  auto funcTy = builder.getFunctionType(argTys, resTys);
  return pushType(funcTy);
}

bool QuakeBridgeVisitor::VisitElaboratedType(clang::ElaboratedType *t) {
  auto name = t->getNamedType().getAsString();
  auto *context = builder.getContext();
  if (matchTypeName("std", "vector", name))
    return pushType(cudaq::cc::StdvecType::get(context, popType()));
  if (matchTypeName("std", "_Bit_type", name, /*toEnd=*/true))
    return pushType(builder.getI64Type());
  if (matchTypeName("std", "_Bit_reference", name, /*toEnd=*/true))
    return pushType(builder.getI1Type());
  if (matchTypeName("std", "uint8_t", name, /*toEnd=*/true))
    return pushType(builder.getI8Type());
  if (matchTypeName("std", "function", name)) {
    // Note: technically this is a wrapper around a Callable and not exactly a
    // callable. (Callables are a generalization of closures, functions,
    // methods, etc.) Do we want to refine our CC type system here? For
    // expedience, pretend Callables and callables are "the same".
    auto fnTy = dyn_cast<FunctionType>(popType());
    assert(fnTy);
    return pushType(cudaq::cc::CallableType::get(context, fnTy));
  }
  if (matchTypeName("cudaq", "qubit", name, /*toEnd=*/true) ||
      matchTypeName("cudaq", "qudit<2UL>", name, /*toEnd=*/true))
    return pushType(quake::RefType::get(context));
  if (matchTypeName("cudaq", "qreg", name))
    return pushType(
        quake::VeqType::get(context, getQregArraySize(t->getNamedType())));
  if (matchTypeName("cudaq", "qspan", name))
    return pushType(quake::VeqType::get(context, 0));
  if (matchTypeName("cudaq", "qarray", name) ||
      matchTypeName("cudaq", "qvector", name) ||
      matchTypeName("cudaq", "qview", name)) {
    reportClangError(t->getAsRecordDecl(), mangler,
                     "type is not yet implemented");
    return pushType(NoneType::get(context));
  }
  // N.B.: always traversing the desugared type leads to infinite recursion.
  LLVM_DEBUG(llvm::dbgs() << " skipping elaborated type: " << name << '\n';
             t->dump(llvm::dbgs(), *astContext));
  return true;
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
  return pushType(cc::PointerType::get(popType()));
}

bool QuakeBridgeVisitor::VisitLValueReferenceType(
    clang::LValueReferenceType *t) {
  auto eleTy = popType();
  if (isa<cc::CallableType, cc::StdvecType, quake::VeqType, quake::RefType>(
          eleTy))
    return pushType(eleTy);
  return pushType(cc::PointerType::get(eleTy));
}

bool QuakeBridgeVisitor::VisitRValueReferenceType(
    clang::RValueReferenceType *t) {
  auto eleTy = popType();
  // FIXME: LLVMStructType is promoted as a temporary workaround.
  if (isa<cc::CallableType, cc::StdvecType, cc::ArrayType, cc::StructType,
          quake::VeqType, quake::RefType, LLVM::LLVMStructType>(eleTy))
    return pushType(eleTy);
  return pushType(cc::PointerType::get(eleTy));
}

bool QuakeBridgeVisitor::pushType(Type t) {
  LLVM_DEBUG(llvm::dbgs() << "push " << t << '\n');
  typeStack.push_back(t);
  return true;
}

Type QuakeBridgeVisitor::popType() {
  assert(!typeStack.empty());
  Type result = peekType();
  LLVM_DEBUG(llvm::dbgs() << "(pop " << result << ")\n");
  typeStack.pop_back();
  return result;
}

/// Return the last `n` types from the stack in left-to-right (natural)
/// order. For a signature, `f(T, U, V)` this can be used to return a list
/// `[type_T type_U type_V]`.
SmallVector<Type> QuakeBridgeVisitor::lastTypes(unsigned n) {
  assert(n <= typeStack.size() && "stack has fewer types than requested");
  SmallVector<Type> result(typeStack.end() - n, typeStack.end());
  LLVM_DEBUG(llvm::dbgs() << "(pop <" << n << ">)\n");
  typeStack.pop_back_n(n);
  return result;
}

Type QuakeBridgeVisitor::genType(const clang::QualType &ty) {
  LLVM_DEBUG(llvm::dbgs() << "type to generate: " << ty << '\n');
  QuakeBridgeVisitor vis(*this);
  vis.typeMode = true;
  bool res = vis.TraverseType(ty);
  if (!res)
    TODO("type conversion to MLIR type");
  return vis.popType();
}

std::optional<FunctionType>
QuakeBridgeVisitor::getFunctionType(const clang::FunctionDecl *x,
                                    bool isKernel) {
  auto astTy = x->getType();
  auto funcTy = dyn_cast<FunctionType>(genType(astTy));
  assert(funcTy && "expecting a function type");
  if (isKernel) {
    // Verify the argument and return types are valid for a kernel.
    auto *protoTy = dyn_cast<clang::FunctionProtoType>(astTy.getTypePtr());
    if (!protoTy) {
      reportClangError(x, mangler, "kernel must have a prototype");
      return {};
    }
    if (protoTy->getNumParams() != funcTy.getNumInputs()) {
      // The arity of the function doesn't match, so report an error.
      reportClangError(x, mangler, "kernel has unexpected arguments");
      return {};
    }
    for (auto [t, p] : llvm::zip(funcTy.getInputs(), x->parameters())) {
      // Structs, callables, functions are valid callable objects. Also pure
      // device kernels may take veq and/or ref arguments.
      if (isArithmeticType(t) || isArithmeticSequenceType(t) ||
          isQuantumType(t) || isKernelCallable(t) || isFunctionCallable(t) ||
          isReferenceToCallableRecord(t, p))
        continue;
      reportClangError(p, mangler, "kernel argument type not supported");
      return {};
    }
    for (auto t : funcTy.getResults()) {
      if (isArithmeticType(t) || isArithmeticSequenceType(t))
        continue;
      reportClangError(x, mangler, "kernel result type not supported");
      return {};
    }
  }
  return funcTy;
}

} // namespace cudaq::details
