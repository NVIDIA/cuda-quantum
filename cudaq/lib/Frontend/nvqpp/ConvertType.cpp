/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <span>

#define DEBUG_TYPE "lower-ast-type"

using namespace mlir;

static bool isArithmeticType(Type t) {
  return isa<IntegerType, FloatType, ComplexType>(t);
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

static bool isRecursiveArithmeticProductType(Type t);

/// Is \p t a recursive sequence of arithmetic types? The outer types may be
/// dynamic (vector) or product types. Only ArrayType is considered an inner
/// type.
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
  if (auto lambdaTy = dyn_cast<cudaq::cc::IndirectCallableType>(t))
    return isKernelSignatureType(lambdaTy.getSignature());
  return false;
}

static bool isFunctionCallable(Type t) {
  if (auto funcTy = dyn_cast<FunctionType>(t))
    return isKernelSignatureType(funcTy);
  return false;
}

/// Return true if and only if \p t is a (simple) arithmetic type or a possibly
/// dynamic type composed of arithmetic types: a vector, a struct, or any
/// nesting of these, such as a vector of vectors or a struct with a vector
/// member. See the return statement in ConvertStmt.cpp for how the heap
/// storage of a dynamic result is made to outlive the kernel.
///
/// `cudaq::measure_handle` (and any aggregate that transitively names it) is
/// also allowed in pure-device kernels.
static bool isKernelResultType(Type t) {
  return isArithmeticType(t) || isComposedArithmeticType(t) ||
         cudaq::cc::containsMeasureHandle(t);
}

/// Return true if and only if \p t is a (simple) arithmetic type, an possibly
/// dynamic type composed of arithmetic types, a quantum type, a callable
/// (function), or a string. Types that
/// transitively contain `cudaq::measure_handle` are also allowed in pure-device
/// kernels.
static bool isKernelArgumentType(Type t) {
  return isArithmeticType(t) || isComposedArithmeticType(t) ||
         cudaq::quake::isQuantumReferenceType(t) || isKernelCallable(t) ||
         isFunctionCallable(t) ||
         // TODO: move from pointers to a builtin string type.
         cudaq::isCharPointerType(t) || cudaq::cc::containsMeasureHandle(t);
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

namespace cudaq::detail {

clang::FunctionDecl *findCallOperator(const clang::CXXRecordDecl *decl) {
  for (auto *m : decl->methods())
    if (m->isOverloadedOperator() &&
        cudaq::isCallOperator(m->getOverloadedOperator()))
      return m->getDefinition();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// QuakeTypeVisitor
//===----------------------------------------------------------------------===//

Location QuakeTypeVisitor::toLocation(const clang::SourceRange &range) {
  return toSourceLocation(builder.getContext(), astContext, range);
}

QuakeTypeVisitor::Result QuakeTypeVisitor::requireType(clang::SourceRange range,
                                                       clang::QualType qt) {
  auto result = traverse(qt);
  if (!result && !hasFailed())
    emitFatalError(toLocation(range), "expected a type");
  return result;
}

static StringRef recordName(clang::RecordDecl *x) {
  if (auto *ident = x->getIdentifier())
    return ident->getName();
  return {};
}

QuakeTypeVisitor::Result QuakeTypeVisitor::visit(clang::RecordType *t) {
  auto *recDecl = t->getDecl();
  if (ignoredClass(recDecl))
    return std::nullopt;
  // A record has one type, however it is spelled (`S`, `struct S`).
  const clang::RecordDecl *key =
      recDecl->getDefinition() ? recDecl->getDefinition() : recDecl;
  if (converting.contains(key)) {
    // This record is part of its own definition. Kernels do not support
    // recursive types, since they cannot be a finite type.
    reportClangError(key, mangler,
                     "recursive types are not allowed in kernels");
    return fail();
  }
  if (auto iter = records.find(key); iter != records.end())
    return iter->second;
  converting.insert(key);
  Result result = convertRecord(recDecl);
  converting.erase(key);
  if (hasFailed())
    return std::nullopt;
  if (!result) {
    if (!allowUnknownRecordType) {
      recDecl->dump();
      emitFatalError(toLocation(recDecl->getSourceRange()), "expected a type");
    }
    // This is a kernel's type signature, so use a NoneType. When finally
    // returning out of determining the kernel's type signature, a clang error
    // diagnostic will be reported.
    result = builder.getNoneType();
  }
  records[key] = *result;
  return result;
}

QuakeTypeVisitor::Result QuakeTypeVisitor::convertRecord(clang::RecordDecl *x) {
  bool intercepted = false;
  Result replacement = interceptRecordDecl(x, intercepted);
  if (intercepted || hasFailed())
    return replacement;

  if (x->isLambda()) {
    // A lambda is a callable with the signature of its call operator.
    auto *funcDecl = findCallOperator(cast<clang::CXXRecordDecl>(x));
    auto funcTy = requireType(funcDecl->getSourceRange(), funcDecl->getType());
    if (!funcTy)
      return std::nullopt;
    return cc::CallableType::get(cast<FunctionType>(*funcTy));
  }

  if (isa<clang::CXXRecordDecl>(x) && x->isUnion()) {
    reportClangError(x, mangler, "union types are not allowed in kernels");
    return fail();
  }

  auto *ctx = builder.getContext();
  if (!x->getDefinition())
    return cc::StructType::get(ctx, recordName(x), /*isOpaque=*/true);

  // The member types of the StructType are the types of the fields.
  SmallVector<Type> fieldTys;
  for (auto *field : x->fields()) {
    auto fieldTy = traverse(field->getType());
    if (hasFailed())
      return std::nullopt;
    if (fieldTy)
      fieldTys.push_back(*fieldTy);
  }
  return convertProductType(x, fieldTys);
}

std::pair<std::uint64_t, unsigned>
QuakeTypeVisitor::getWidthAndAlignment(clang::RecordDecl *x) {
  auto *defn = x->getDefinition();
  assert(defn && "struct must be defined here");
  auto qualTy = astContext->getCanonicalTagType(defn);
  if (qualTy->isDependentType())
    return {0, 0};
  auto ti = astContext->getTypeInfo(qualTy);
  return {ti.Width, llvm::PowerOf2Ceil(ti.Align) / 8};
}

QuakeTypeVisitor::Result
QuakeTypeVisitor::convertProductType(clang::RecordDecl *x,
                                     ArrayRef<Type> fieldTys) {
  StringRef name = recordName(x);
  auto *ctx = builder.getContext();
  auto [width, alignInBytes] = getWidthAndAlignment(x);

  // This is a struq if it is not empty and all members are quantum references.
  bool isStruq = !fieldTys.empty();
  bool quantumMembers = false;
  for (auto ty : fieldTys) {
    if (cudaq::quake::isQuantumType(ty))
      quantumMembers = true;
    if (!quake::isQuantumReferenceType(ty))
      isStruq = false;
  }
  if (quantumMembers && !isStruq) {
    reportClangError(x, mangler,
                     "hybrid quantum-classical struct types are not allowed");
    return fail();
  }

  auto ty = [&]() -> Type {
    if (isStruq)
      return cudaq::quake::StruqType::get(ctx, fieldTys);
    if (name.empty())
      return cc::StructType::get(ctx, fieldTys, width, alignInBytes);
    return cc::StructType::get(ctx, name, fieldTys, width, alignInBytes);
  }();

  // Do some error analysis on the product type. Check the following:

  // - If this is a struq:
  if (isa<cudaq::quake::StruqType>(ty)) {
    // -- does it contain invalid C++ types?
    for (auto *field : x->fields()) {
      auto *ty = field->getType().getTypePtr();
      if (ty->isLValueReferenceType()) {
        auto *lref = cast<clang::LValueReferenceType>(ty);
        ty = lref->getPointeeType().getTypePtr();
      }
      if (auto *tyDecl = ty->getAsRecordDecl()) {
        if (auto *ident = tyDecl->getIdentifier()) {
          auto name = ident->getName();
          if (isInNamespace(tyDecl, "cudaq")) {
            //  can be owning container; so can be qubit, qarray, or qvector
            if ((name == "qudit" || name == "qubit" || name == "qvector" ||
                 name == "qarray"))
              continue;
            // must be qview or qview&
            if (name == "qview")
              continue;
          }
        }
      }
      reportClangError(x, mangler, "quantum struct has invalid member type.");
    }
    // -- does it contain contain a struq member? Not allowed.
    for (auto fieldTy : fieldTys)
      if (isa<cudaq::quake::StruqType>(fieldTy))
        reportClangError(x, mangler,
                         "recursive quantum struct types are not allowed.");
  }

  // - Is this a struct does it have quantum types? Not allowed.
  if (!isa<cudaq::quake::StruqType>(ty))
    for (auto fieldTy : fieldTys)
      if (cudaq::quake::isQuakeType(fieldTy))
        reportClangError(
            x, mangler,
            "hybrid quantum-classical struct types are not allowed.");

  // - Does this product type have (user-defined) member functions? Not allowed.
  if (auto *cxxRd = dyn_cast<clang::CXXRecordDecl>(x)) {
    auto numMethods = [&cxxRd]() {
      std::size_t count = 0;
      for (auto methodIter = cxxRd->method_begin();
           methodIter != cxxRd->method_end(); ++methodIter) {
        // Don't check if this is a __qpu__ struct method
        if (auto attr = (*methodIter)->getAttr<clang::AnnotateAttr>();
            attr && attr->getAnnotation().str() == cudaq::kernelAnnotation)
          continue;
        // Check if the method is not implicit (i.e., user-defined)
        if (!(*methodIter)->isImplicit())
          count++;
      }
      return count;
    }();

    if (numMethods > 0)
      reportClangError(
          x, mangler,
          "struct with user-defined methods is not allowed in quantum kernel.");
  }

  return ty;
}

QuakeTypeVisitor::Result QuakeTypeVisitor::visit(clang::FunctionProtoType *t) {
  assert(t->exceptions().empty() && "exceptions are not supported in CUDA-Q");
  // The noexcept expression, if any, has no semantics other than for
  // inferring a type, so it is not visited.
  auto funcRetTy = traverse(t->getReturnType());
  SmallVector<Type> argTys;
  for (auto paramTy : t->param_types()) {
    if (auto argTy = traverse(paramTy))
      argTys.push_back(*argTy);
    if (hasFailed())
      return std::nullopt;
  }
  SmallVector<Type> resTys;
  if (funcRetTy && !isa<NoneType>(*funcRetTy))
    resTys.push_back(*funcRetTy);
  return builder.getFunctionType(argTys, resTys);
}

/// Parallels the clang conversion from `clang::Type` to `llvm::Type`. In this
/// case, we translate `clang::Type` to `mlir::Type`. See
/// `clang::CodeGenTypes.ConvertType`.
Type QuakeTypeVisitor::builtinTypeToType(const clang::BuiltinType *t) {
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

void QuakeTypeVisitor::unsupported(clang::Type *t) {
  auto &de = astContext->getDiagnostics();
  const auto id =
      de.getCustomDiagID(clang::DiagnosticsEngine::Error,
                         "type '%0' is not yet supported in a kernel");
  de.Report(id) << t->getTypeClassName();
  fail();
}

QuakeTypeVisitor::Result QuakeTypeVisitor::visit(clang::BuiltinType *t) {
  return builtinTypeToType(t);
}

/// An enumeration is represented by its underlying integer type.
QuakeTypeVisitor::Result QuakeTypeVisitor::visit(clang::EnumType *t) {
  auto underlying = t->getDecl()->getIntegerType();
  if (underlying.isNull()) {
    // An enumeration without a definition has no known representation.
    return std::nullopt;
  }
  return traverse(underlying);
}

QuakeTypeVisitor::Result QuakeTypeVisitor::visit(clang::PointerType *t) {
  if (t->getPointeeType()->isUndeducedAutoType())
    return cc::PointerType::get(builder.getContext());
  auto eleTy = traverse(t->getPointeeType());
  if (!eleTy)
    return std::nullopt;
  return cc::PointerType::get(*eleTy);
}

QuakeTypeVisitor::Result
QuakeTypeVisitor::visit(clang::LValueReferenceType *t) {
  if (t->getPointeeType()->isUndeducedAutoType())
    return cc::PointerType::get(builder.getContext());
  auto eleTy = traverse(t->getPointeeType());
  if (!eleTy)
    return std::nullopt;
  if (isa<cc::CallableType, cc::IndirectCallableType, cc::SpanLikeType,
          cudaq::quake::VeqType, cudaq::quake::RefType,
          cudaq::quake::StruqType>(*eleTy))
    return eleTy;
  return cc::PointerType::get(*eleTy);
}

QuakeTypeVisitor::Result
QuakeTypeVisitor::visit(clang::RValueReferenceType *t) {
  if (t->getPointeeType()->isUndeducedAutoType())
    return cc::PointerType::get(builder.getContext());
  auto eleTy = traverse(t->getPointeeType());
  if (!eleTy)
    return std::nullopt;
  // FIXME: LLVMStructType is promoted as a temporary workaround.
  if (isa<cc::ArrayType, cc::CallableType, cc::IndirectCallableType,
          cc::SpanLikeType, cc::StructType, cudaq::quake::VeqType,
          cudaq::quake::RefType, cudaq::quake::StruqType, LLVM::LLVMStructType>(
          *eleTy))
    return eleTy;
  return cc::PointerType::get(*eleTy);
}

QuakeTypeVisitor::Result QuakeTypeVisitor::visit(clang::ConstantArrayType *t) {
  auto size = t->getSize().getZExtValue();
  auto ty = traverse(t->getElementType());
  if (!ty)
    return std::nullopt;
  if (cudaq::quake::isQuantumType(*ty)) {
    auto *ctx = builder.getContext();
    if (*ty == cudaq::quake::RefType::get(ctx))
      return cudaq::quake::VeqType::getUnsized(ctx);
    emitFatalError(builder.getUnknownLoc(),
                   "array element type is not supported");
  }
  return cc::ArrayType::get(builder.getContext(), *ty, size);
}

QuakeTypeVisitor::Result
QuakeTypeVisitor::interceptRecordDecl(clang::RecordDecl *x, bool &intercepted) {
  // Some decls will be intercepted and replaced with high-level types in quake.
  // Do this here to avoid traversing their fields, etc. Any path that returns
  // without setting `intercepted` to false was intercepted. An intercepted
  // record may not have a type (which is not the same as an error).
  intercepted = true;
  auto notIntercepted = [&]() -> Result {
    intercepted = false;
    return std::nullopt;
  };
  auto *ident = x->getIdentifier();
  if (!ident || x->isLambda())
    return notIntercepted();
  auto name = ident->getName();
  auto *ctx = builder.getContext();
  auto *cts = dyn_cast<clang::ClassTemplateSpecializationDecl>(x);
  // Convert the type of template argument \p i.
  auto argType = [&](unsigned i) -> Result {
    return requireType(x->getSourceRange(),
                       cts->getTemplateArgs()[i].getAsType());
  };
  if (isInNamespace(x, "cudaq")) {
    // Types from the `cudaq` namespace.
    // A qubit is a qudit<LEVEL=2>.
    if (name == "qudit" || name == "qubit")
      return cudaq::quake::RefType::get(ctx);
    // qreg<SIZE,LEVEL>, qarray<SIZE,LEVEL>, qspan<SIZE,LEVEL>
    if (name == "qspan" || name == "qreg" || name == "qarray") {
      // If the first template argument is not `std::dynamic_extent` then we
      // have a constant sized VeqType.
      if (cts) {
        auto templArg = cts->getTemplateArgs()[0];
        assert(templArg.getKind() ==
               clang::TemplateArgument::ArgKind::Integral);
        auto getExtValueHelper = [](auto v) -> std::int64_t {
          if (v.isUnsigned())
            return static_cast<std::int64_t>(v.getZExtValue());
          return v.getSExtValue();
        };
        std::int64_t size = getExtValueHelper(templArg.getAsIntegral());
        if (size != static_cast<std::int64_t>(std::dynamic_extent))
          return cudaq::quake::VeqType::get(ctx, size);
      }
      return cudaq::quake::VeqType::getUnsized(ctx);
    }
    // qvector<LEVEL>, qview<LEVEL>
    if (name == "qvector" || name == "qview")
      return cudaq::quake::VeqType::getUnsized(ctx);
    if (name == "state")
      return cudaq::quake::StateType::get(ctx);
    if (name == "pauli_word")
      return cc::CharspanType::get(ctx);
    if (name == "measure_handle")
      return cc::MeasureHandleType::get(ctx);
    if (name == "qkernel") {
      // Template argument 0 is the function's signature.
      auto fnTy = argType(0);
      if (!fnTy)
        return std::nullopt;
      return cc::IndirectCallableType::get(cast<FunctionType>(*fnTy));
    }
    if (!isInNamespace(x, "solvers") && !isInNamespace(x, "qec")) {
      auto loc = toLocation(x->getSourceRange());
      TODO_loc(loc, "unhandled type, " + name + ", in cudaq namespace");
    }
  }
  if (isInNamespace(x, "std")) {
    if (name == "vector") {
      // Template argument 0 is the vector's element type.
      if (!cts)
        return std::nullopt;
      auto ty = argType(0);
      if (!ty)
        return std::nullopt;
      if (cudaq::quake::isQuantumType(*ty)) {
        if (*ty == cudaq::quake::RefType::get(ctx))
          return cudaq::quake::VeqType::getUnsized(ctx);
        cudaq::emitFatalError(toLocation(x->getSourceRange()),
                              "std::vector element type is not supported");
      }
      return cc::SequenceType::get(ctx, *ty);
    }
    // std::vector<bool>   =>   cc.sequence<i1>
    if (name == "_Bit_reference" || name == "__bit_reference" ||
        name == "__bit_const_reference") {
      // Reference to a bit in a std::vector<bool>. Promote to a value.
      return builder.getI1Type();
    }
    if (name == "_Bit_type")
      return builder.getI64Type();
    if (name == "complex") {
      // Template argument 0 is the complex's element type.
      if (!cts)
        return std::nullopt;
      auto memTy = argType(0);
      if (!memTy)
        return std::nullopt;
      return ComplexType::get(*memTy);
    }
    if (name == "initializer_list") {
      // Template argument 0 is the initializer list's element type.
      if (!cts)
        return std::nullopt;
      auto memTy = argType(0);
      if (!memTy)
        return std::nullopt;
      return cc::ArrayType::get(*memTy);
    }
    if (name == "function") {
      // Template argument 0 is the function's signature.
      auto fnTy = argType(0);
      if (!fnTy)
        return std::nullopt;
      return cc::CallableType::get(ctx, cast<FunctionType>(*fnTy));
    }
    if (name == "reference_wrapper") {
      auto refTy = argType(0);
      if (!refTy)
        return std::nullopt;
      if (isa<cudaq::quake::RefType, cudaq::quake::VeqType>(*refTy))
        return refTy;
      return cc::PointerType::get(ctx, *refTy);
    }
    if (name == "basic_string") {
      if (allowUnknownRecordType) {
        // Kernel argument list contains a `std::string` type. Intercept it and
        // generate a clang diagnostic when returning out of determining the
        // kernel's type signature.
        return std::nullopt;
      }
      TODO_x(toLocation(x->getSourceRange()), x, mangler, "std::string type");
      return fail();
    }
    if (name == "__wrap_iter") {
      // An iterator is represented by its element type.
      return argType(0);
    }
    if (name == "pair") {
      SmallVector<Type> members;
      for (unsigned i = 0; i < 2; ++i) {
        auto memTy = argType(i);
        if (!memTy)
          return std::nullopt;
        members.push_back(*memTy);
      }
      auto [width, align] = getWidthAndAlignment(x);
      return cc::StructType::get(ctx, members, width, align);
    }
    if (name == "tuple") {
      auto &templateArg = cts->getTemplateArgs()[0];
      if (templateArg.getKind() != clang::TemplateArgument::Pack)
        return notIntercepted();
      SmallVector<Type> members;
      for (auto &ta : templateArg.pack_elements()) {
        auto memTy = requireType(x->getSourceRange(), ta.getAsType());
        if (!memTy)
          return std::nullopt;
        members.push_back(*memTy);
      }
      auto [width, align] = getWidthAndAlignment(x);
      if (tuplesAreReversed) {
        std::reverse(members.begin(), members.end());
        // Resets are for libstdc++ calling convention compatibility.
        width = 0;
        align = 0;
      }
      return cc::StructType::get(ctx, members, width, align);
    }
    if (ignoredClass(x))
      return std::nullopt;
    if (allowUnknownRecordType) {
      // This is a catch all for other container types (deque, map, set, etc.)
      // that the user may try to pass as arguments to a kernel. Having no type
      // here will cause the kernel's signature to emit a diagnostic.
      return std::nullopt;
    }
    // Any other standard library class is not supported. The failure is
    // silent; the caller diagnoses the construct that required the type.
    LLVM_DEBUG(llvm::dbgs()
               << "in std namespace, " << name << " is not matched\n");
    return fail();
  }

  if (isInNamespace(x, "__gnu_cxx")) {
    if (name == "__promote" || name == "__promote_2") {
      // Recover the typedef in this class. Then find the canonical type
      // resolved for that typedef and use that as the type.
      for (auto *d : x->decls())
        if (auto *tdDecl = dyn_cast<clang::TypedefDecl>(d))
          return requireType(x->getSourceRange(),
                             tdDecl->getUnderlyingType().getCanonicalType());
      return std::nullopt;
    }
    if (name == "__normal_iterator") {
      // An iterator is represented by its element type.
      return argType(0);
    }
  }
  return notIntercepted();
}

static bool isReferenceToCudaqStateType(Type t) {
  if (auto ptrTy = dyn_cast<cc::PointerType>(t))
    return isa<cudaq::quake::StateType>(ptrTy.getElementType());
  return false;
}

// Do syntax checking on the signature of kernel \p x, whose type is \p funcTy.
// Return true if and only if the kernel \p x has a legal signature.
bool QuakeBridgeVisitor::doSyntaxChecks(const clang::FunctionDecl *x,
                                        FunctionType funcTy) {
  auto astTy = x->getType();
  // Verify the argument and return types are valid for a kernel.
  auto *protoTy = dyn_cast<clang::FunctionProtoType>(astTy.getTypePtr());
  auto syntaxError = [&]<unsigned N>(const char (&msg)[N]) -> bool {
    reportClangError(x, mangler, msg);
    LLVM_DEBUG(llvm::dbgs() << "invalid type: " << funcTy << '\n');
    return false;
  };
  if (!protoTy)
    return syntaxError("kernel must have a prototype");
  if (protoTy->getNumParams() != funcTy.getNumInputs()) {
    // The arity of the function doesn't match, so report an error.
    return syntaxError("kernel has unexpected arguments");
  }
  for (auto [t, p] : llvm::zip(funcTy.getInputs(), x->parameters())) {
    // Structs, lambdas, functions are valid callable objects. Also pure
    // device kernels may take veq and/or ref arguments.
    if (isKernelArgumentType(t) || isReferenceToCallableRecord(t, p) ||
        isReferenceToCudaqStateType(t))
      continue;
    return syntaxError("kernel argument type not supported");
  }
  for (auto t : funcTy.getResults()) {
    if (isKernelResultType(t))
      continue;
    return syntaxError("kernel result type not supported");
  }
  return true;
}

} // namespace cudaq::detail
