/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Frontend/nvqpp/QisBuilder.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-ast-expr"

using namespace mlir;

// Get the result type if \p ty is a function type or just return \p ty.
static Type getResultType(Type ty) {
  assert(ty && "Type cannot be null");
  if (auto funcTy = dyn_cast<FunctionType>(ty)) {
    assert(funcTy.getNumResults() == 1);
    return funcTy.getResult(0);
  }
  return ty;
}

/// Convert a name, value pair into a symbol name allocated in `allocator`.
static llvm::StringRef
createQubitSymbolTableName(StringRef qregName, Value idxVal,
                           llvm::BumpPtrAllocator &allocator) {
  std::string name;
  if (auto idxIntVal = idxVal.getDefiningOp<arith::ConstantIntOp>())
    name = qregName.str() + "%" + std::to_string(idxIntVal.value());
  else if (auto idxIdxVal = idxVal.getDefiningOp<arith::ConstantIndexOp>())
    name = qregName.str() + "%" + std::to_string(idxIdxVal.value());
  else {
    // idxVal is a general value, like a loop idx
    std::stringstream ss;
    ss << qregName.str() << "%" << idxVal.getAsOpaquePointer();
    name = ss.str();
  }

  // move `name` to heap memory allocated by the allocator
  char *namePtr = allocator.Allocate<char>(name.size());
  std::memcpy(namePtr, name.data(), name.size());
  return llvm::StringRef(namePtr, name.size());
}

/// Helper to get the declaration of a decl-ref expression.
/// Precondition: \p expr must be a pointer to a DeclRefExpr.
static clang::NamedDecl *getNamedDecl(clang::Expr *expr) {
  auto *call = cast<clang::DeclRefExpr>(expr);
  return call->getDecl()->getUnderlyingDecl();
}

static std::pair<SmallVector<Value>, SmallVector<Value>>
maybeUnpackOperands(OpBuilder &builder, Location loc, ValueRange operands,
                    bool isControl = false, size_t targetCount = 1) {
  // If this is not a controlled op, then we just keep all operands as targets.
  if (!isControl)
    return std::make_pair(operands, SmallVector<Value>{});

  if (operands.size() > 1)
    return std::make_pair(SmallVector<Value>{operands.take_back(targetCount)},
                          SmallVector<Value>{operands.drop_back(targetCount)});

  SmallVector<Value> targets = operands.take_back(targetCount);
  Value last_target = operands.back();

  if (isa<quake::VeqType>(last_target.getType())) {
    // Split the vector. Last `targetCount` are targets, front `N-targetCount`
    // are controls.
    auto vecSize = builder.create<quake::VeqSizeOp>(
        loc, builder.getIntegerType(64), targets);
    auto size = builder.create<cudaq::cc::CastOp>(
        loc, builder.getI64Type(), vecSize, cudaq::cc::CastOpMode::Unsigned);

    auto numTargets =
        builder.create<arith::ConstantIntOp>(loc, targetCount, 64);
    auto offset = builder.create<arith::SubIOp>(loc, size, numTargets);
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    auto last = builder.create<arith::SubIOp>(loc, offset, numTargets);
    // The canonicalizer will compute a constant size, if possible.
    auto unsizedVeqTy = quake::VeqType::getUnsized(builder.getContext());

    // Get the subvector of all targets
    Value targetSubveq = builder.create<quake::SubVeqOp>(
        loc, unsizedVeqTy, last_target, zero, offset);
    // Get the subvector of all qubits excluding the last one: controls.
    Value ctrlSubveq = builder.create<quake::SubVeqOp>(loc, unsizedVeqTy,
                                                       last_target, zero, last);
    return std::make_pair(SmallVector<Value>{targetSubveq},
                          SmallVector<Value>{ctrlSubveq});
  }
  return std::make_pair(targets, SmallVector<Value>{});
}

namespace {
// Type used to specialize the buildOp function. This extends the cases below by
// prefixing a single parameter value to the list of arguments for cases 1
// and 2. A Param does not have a case 3 defined.
class Param {};
} // namespace

/// Create a negated controls attribute from a range of controls, \p ctrls, and
/// a list of which ones should be negated, \p negations.
static DenseBoolArrayAttr
negatedControlsAttribute(MLIRContext *ctx, ValueRange ctrls,
                         SmallVector<Value> &negations) {
  if (negations.empty())
    return {};
  SmallVector<bool> negatedControls(ctrls.size());
  for (auto v : llvm::enumerate(ctrls))
    negatedControls[v.index()] = std::find(negations.begin(), negations.end(),
                                           v.value()) != negations.end();
  auto boolVecAttr = DenseBoolArrayAttr::get(ctx, negatedControls);
  negations.clear();
  return {boolVecAttr};
}

// There are three basic overloads of the "single target" CUDA-Q ops.
//
// 1. op(qubit...)
//    This form takes the last qubit as the target and all qubits to
//    the left as controls.
// 2. op(qurange, qubit)
//    Similar to above except the control qubits are packed in a
//    range container.
// 3. op(qurange)
//    This is not like the other 2. This is syntactic sugar for
//    invoking the op elementally across the entire range container.
//    There are no controls.
//
// In the future, it may be decided to add more overloads to this family (e.g.,
// adding controls to case 3).
template <typename A, typename P = void>
bool buildOp(OpBuilder &builder, Location loc, ValueRange operands,
             SmallVector<Value> &negations,
             llvm::function_ref<void()> reportNegateError,
             bool isAdjoint = false, bool isControl = false,
             size_t paramCount = 1) {
  if constexpr (std::is_same_v<P, Param>) {
    assert(operands.size() >= 2 && "must be at least 2 operands");
    auto params = operands.take_front(paramCount);
    auto [target, ctrls] = maybeUnpackOperands(
        builder, loc, operands.drop_front(paramCount), isControl);
    for (auto v : target)
      if (std::find(negations.begin(), negations.end(), v) != negations.end())
        reportNegateError();
    auto negs =
        negatedControlsAttribute(builder.getContext(), ctrls, negations);
    if (ctrls.empty())
      for (auto t : target)
        builder.create<A>(loc, isAdjoint, params, ctrls, t, negs);
    else {
      assert(target.size() == 1 &&
             "can only have a single target with control qubits.");
      builder.create<A>(loc, isAdjoint, params, ctrls, target, negs);
    }
  } else {
    assert(operands.size() >= 1 && "must be at least 1 operand");
    if ((operands.size() == 1) && isa<quake::VeqType>(operands[0].getType())) {
      auto target = operands[0];
      if (!negations.empty())
        reportNegateError();
      Type i64Ty = builder.getI64Type();
      auto size = builder.create<quake::VeqSizeOp>(
          loc, builder.getIntegerType(64), target);
      Value rank = builder.create<cudaq::cc::CastOp>(
          loc, i64Ty, size, cudaq::cc::CastOpMode::Unsigned);
      auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &,
                             Block &block) {
        Value ref = builder.create<quake::ExtractRefOp>(loc, target,
                                                        block.getArgument(0));
        builder.create<A>(loc, ValueRange(), ref);
      };
      cudaq::opt::factory::createInvariantLoop(builder, loc, rank, bodyBuilder);
    } else {
      auto [target, ctrls] =
          maybeUnpackOperands(builder, loc, operands, isControl);
      for (auto v : target)
        if (std::find(negations.begin(), negations.end(), v) != negations.end())
          reportNegateError();
      auto negs =
          negatedControlsAttribute(builder.getContext(), ctrls, negations);
      if (ctrls.empty())
        // May have multiple targets, but no controls, op(q, r, s, ...)
        for (auto t : target)
          builder.create<A>(loc, isAdjoint, ValueRange(), ValueRange(), t,
                            negs);
      else {
        assert(target.size() == 1 &&
               "can only have a single target with control qubits.");
        builder.create<A>(loc, isAdjoint, ValueRange(), ctrls, target, negs);
      }
    }
  }
  return true;
}

static Value getConstantInt(OpBuilder &builder, Location loc,
                            const uint64_t value, const int bitwidth) {
  return builder.create<arith::ConstantIntOp>(loc, value,
                                              builder.getIntegerType(bitwidth));
}

static Value getConstantInt(OpBuilder &builder, Location loc,
                            const uint64_t value, Type intTy) {
  assert(isa<IntegerType>(intTy));
  return builder.create<arith::ConstantIntOp>(loc, value, intTy);
}

template <auto KindConst, typename T,
          typename = std::enable_if_t<std::is_same_v<decltype(KindConst), T>>>
bool isOperatorKind(T kindValue) {
  return kindValue == KindConst;
}

/// Is \p kind the `operator[]` function?
static bool isSubscriptOperator(clang::OverloadedOperatorKind kind) {
  return isOperatorKind<clang::OverloadedOperatorKind::OO_Subscript>(kind);
}

/// Is \p kind the `operator==` function?
static bool isCompareEqualOperator(clang::OverloadedOperatorKind kind) {
  return isOperatorKind<clang::OverloadedOperatorKind::OO_EqualEqual>(kind);
}

/// Is \p kind the `operator=` function?
static bool isAssignmentOperator(clang::OverloadedOperatorKind kind) {
  return isOperatorKind<clang::OverloadedOperatorKind::OO_Equal>(kind);
}

/// Is \p kind the `operator!` function?
static bool isExclaimOperator(clang::OverloadedOperatorKind kind) {
  return isOperatorKind<clang::OverloadedOperatorKind::OO_Exclaim>(kind);
}

// Perform the standard type coercions when the syntactic expression from the
// AST has arguments of different types.
static void castToSameType(OpBuilder builder, Location loc,
                           const clang::Type *lhsType, Value &lhs,
                           const clang::Type *rhsType, Value &rhs) {
  if (lhs.getType().getIntOrFloatBitWidth() ==
      rhs.getType().getIntOrFloatBitWidth())
    return;
  auto lhsTy = lhs.getType();
  auto rhsTy = rhs.getType();
  if (isa<IntegerType>(lhsTy) && isa<IntegerType>(rhsTy)) {
    if (lhsTy.getIntOrFloatBitWidth() < rhsTy.getIntOrFloatBitWidth()) {
      auto mode = (lhsType && lhsType->isUnsignedIntegerOrEnumerationType())
                      ? cudaq::cc::CastOpMode::Unsigned
                      : cudaq::cc::CastOpMode::Signed;
      lhs = builder.create<cudaq::cc::CastOp>(loc, rhs.getType(), lhs, mode);
      return;
    }
    auto mode = (rhsType && rhsType->isUnsignedIntegerOrEnumerationType())
                    ? cudaq::cc::CastOpMode::Unsigned
                    : cudaq::cc::CastOpMode::Signed;
    rhs = builder.create<cudaq::cc::CastOp>(loc, lhs.getType(), rhs, mode);
    return;
  }
  if (isa<FloatType>(lhsTy) && isa<FloatType>(rhsTy)) {
    if (lhsTy.getIntOrFloatBitWidth() < rhsTy.getIntOrFloatBitWidth()) {
      lhs = builder.create<cudaq::cc::CastOp>(loc, rhs.getType(), lhs);
      return;
    }
    rhs = builder.create<cudaq::cc::CastOp>(loc, lhs.getType(), rhs);
    return;
  }
  if (isa<FloatType>(lhsTy) && isa<IntegerType>(rhsTy)) {
    auto mode = (rhsType && rhsType->isUnsignedIntegerOrEnumerationType())
                    ? cudaq::cc::CastOpMode::Unsigned
                    : cudaq::cc::CastOpMode::Signed;
    rhs = builder.create<cudaq::cc::CastOp>(loc, lhs.getType(), rhs, mode);
    return;
  }
  if (isa<IntegerType>(lhsTy) && isa<FloatType>(rhsTy)) {
    auto mode = (lhsType && lhsType->isUnsignedIntegerOrEnumerationType())
                    ? cudaq::cc::CastOpMode::Unsigned
                    : cudaq::cc::CastOpMode::Signed;
    lhs = builder.create<cudaq::cc::CastOp>(loc, rhs.getType(), lhs, mode);
    return;
  }
  TODO_loc(loc, "conversion of operands in binary expression");
}

static clang::CXXRecordDecl *
classDeclFromTemplateArgument(clang::FunctionDecl &func,
                              std::size_t argumentPosition,
                              clang::ASTContext &astContext) {
  if (auto *paramDecl = func.getParamDecl(argumentPosition))
    if (auto *defn = paramDecl->getDefinition(astContext)) {
      // Check `auto &&` case.
      if (auto *rvalueRefTy = dyn_cast<clang::RValueReferenceType>(
              defn->getType().getTypePtr()))
        if (auto *substTmpl = dyn_cast<clang::SubstTemplateTypeParmType>(
                rvalueRefTy->getPointeeType().getTypePtr())) {
          auto qualTy = substTmpl->getReplacementType();
          return qualTy.getTypePtr()->getAsCXXRecordDecl();
        }
      // Check `class-name &` case.
      if (auto *lvalueRefTy = dyn_cast<clang::LValueReferenceType>(
              defn->getType().getTypePtr()))
        return lvalueRefTy->getPointeeType().getTypePtr()->getAsCXXRecordDecl();
    }
  return nullptr;
}

/// Is this type name one of the `cudaq` types that map to a VeqType?
static bool isCudaQType(StringRef tn) {
  return tn == "qreg" || tn == "qspan" || tn == "qarray" || tn == "qview" ||
         tn == "qvector";
}

namespace cudaq::details {
/// Is \p x the `operator()` function?
static bool isCallOperator(clang::CXXOperatorCallExpr *x) {
  return cudaq::isCallOperator(x->getOperator());
}

FunctionType QuakeBridgeVisitor::peelPointerFromFunction(Type ty) {
  if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(ty))
    ty = ptrTy.getElementType();
  return cast<FunctionType>(ty);
}

bool QuakeBridgeVisitor::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  auto rhs = popValue();
  auto lhs = popValue();
  Type eleTy = [&]() {
    // NB: Check both arguments as expression may be inverted.
    if (auto ptrTy = dyn_cast<cc::PointerType>(lhs.getType()))
      return ptrTy.getElementType();
    return cast<cc::PointerType>(rhs.getType()).getElementType();
  }();
  Type arrEleTy = [&]() {
    // FIXME: The following dyn_cast should never fail.
    if (auto arrTy = dyn_cast<cc::ArrayType>(eleTy))
      return arrTy.getElementType();
    return eleTy;
  }();
  auto elePtrTy = cc::PointerType::get(arrEleTy);
  return pushValue(builder.create<cc::ComputePtrOp>(loc, elePtrTy, lhs, rhs));
}

bool QuakeBridgeVisitor::VisitFloatingLiteral(clang::FloatingLiteral *x) {
  // Literals do not push a type on the type stack.
  auto loc = toLocation(x->getSourceRange());
  auto bltTy = cast<clang::BuiltinType>(x->getType().getTypePtr());
  auto fltTy = cast<FloatType>(builtinTypeToType(bltTy));
  auto fltVal = x->getValue();
  return pushValue(
      opt::factory::createFloatConstant(loc, builder, fltVal, fltTy));
}

bool QuakeBridgeVisitor::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  auto intTy =
      builtinTypeToType(cast<clang::BuiltinType>(x->getType().getTypePtr()));
  auto intVal = x->getValue();
  return pushValue(getConstantInt(builder, loc, intVal ? 1 : 0, intTy));
}

bool QuakeBridgeVisitor::VisitIntegerLiteral(clang::IntegerLiteral *x) {
  auto loc = toLocation(x->getSourceRange());
  auto intTy =
      builtinTypeToType(cast<clang::BuiltinType>(x->getType().getTypePtr()));
  auto intVal = x->getValue().getLimitedValue();
  return pushValue(getConstantInt(builder, loc, intVal, intTy));
}

bool QuakeBridgeVisitor::VisitCharacterLiteral(clang::CharacterLiteral *x) {
  auto loc = toLocation(x->getSourceRange());
  auto intTy =
      builtinTypeToType(cast<clang::BuiltinType>(x->getType().getTypePtr()));
  auto intVal = x->getValue();
  return pushValue(builder.create<arith::ConstantIntOp>(loc, intVal, intTy));
}

bool QuakeBridgeVisitor::VisitUnaryOperator(clang::UnaryOperator *x) {
  auto loc = toLocation(x->getSourceRange());
  switch (x->getOpcode()) {
  case clang::UnaryOperatorKind::UO_PostInc: {
    auto var = popValue();
    auto loaded = builder.create<cc::LoadOp>(loc, var);
    auto incremented = builder.create<arith::AddIOp>(
        loc, loaded,
        getConstantInt(builder, loc, 1,
                       loaded.getType().getIntOrFloatBitWidth()));
    builder.create<cc::StoreOp>(loc, incremented, var);
    return pushValue(loaded);
  }
  case clang::UnaryOperatorKind::UO_PreInc: {
    auto var = popValue();
    auto loaded = builder.create<cc::LoadOp>(loc, var);
    auto incremented = builder.create<arith::AddIOp>(
        loc, loaded,
        getConstantInt(builder, loc, 1,
                       loaded.getType().getIntOrFloatBitWidth()));
    builder.create<cc::StoreOp>(loc, incremented, var);
    return pushValue(incremented);
  }
  case clang::UnaryOperatorKind::UO_PostDec: {
    auto var = popValue();
    auto loaded = builder.create<cc::LoadOp>(loc, var);
    auto decremented = builder.create<arith::SubIOp>(
        loc, loaded,
        getConstantInt(builder, loc, 1,
                       loaded.getType().getIntOrFloatBitWidth()));
    builder.create<cc::StoreOp>(loc, decremented, var);
    return pushValue(loaded);
  }
  case clang::UnaryOperatorKind::UO_PreDec: {
    auto var = popValue();
    auto loaded = builder.create<cc::LoadOp>(loc, var);
    auto decremented = builder.create<arith::SubIOp>(
        loc, loaded,
        getConstantInt(builder, loc, 1,
                       loaded.getType().getIntOrFloatBitWidth()));
    builder.create<cc::StoreOp>(loc, decremented, var);
    return pushValue(decremented);
  }
  case clang::UnaryOperatorKind::UO_LNot: {
    auto var = popValue();
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, var.getType());
    Value unaryNot =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, var, zero);
    return pushValue(unaryNot);
  }
  case clang::UnaryOperatorKind::UO_Minus: {
    auto subExpr = popValue();
    auto resTy = subExpr.getType();
    if (isa<IntegerType>(resTy))
      return pushValue(builder.create<arith::MulIOp>(
          loc, subExpr,
          getConstantInt(builder, loc, -1, resTy.getIntOrFloatBitWidth())));

    if (isa<FloatType>(resTy)) {
      auto neg_one = opt::factory::createFloatConstant(loc, builder, -1.0,
                                                       cast<FloatType>(resTy));
      return pushValue(builder.create<arith::MulFOp>(loc, subExpr, neg_one));
    }
    TODO_x(loc, x, mangler, "unknown type for unary minus");
    return false;
  }
  case clang::UnaryOperatorKind::UO_Deref: {
    auto subExpr = popValue();
    assert(isa<cc::PointerType>(subExpr.getType()));
    return pushValue(builder.create<cc::LoadOp>(loc, subExpr));
  }
  case clang::UnaryOperatorKind::UO_AddrOf: {
    auto subExpr = peekValue();
    assert(isa<cc::PointerType>(subExpr.getType()));
    return true;
  }
  case clang::UnaryOperatorKind::UO_Extension: {
    TODO_x(loc, x, mangler, "__extension__ operator");
    return false;
  }
  case clang::UnaryOperatorKind::UO_Coawait: {
    TODO_x(loc, x, mangler, "co_await operator");
    return false;
  }
  }
  TODO_x(loc, x, mangler, "unprocessed unary operator");
  return false;
}

Value QuakeBridgeVisitor::floatingPointCoercion(Location loc, Type toType,
                                                Value value) {
  auto fromType = value.getType();
  if (toType == fromType)
    return value;
  assert(isa<FloatType>(fromType) && isa<FloatType>(toType));
  return builder.create<cudaq::cc::CastOp>(loc, toType, value);
}

Value QuakeBridgeVisitor::integerCoercion(Location loc,
                                          const clang::QualType &clangTy,
                                          Type dstTy, Value srcVal) {
  auto fromTy = getResultType(srcVal.getType());
  if (dstTy == fromTy)
    return srcVal;

  assert(isa<IntegerType>(fromTy) && isa<IntegerType>(dstTy));
  if (fromTy.getIntOrFloatBitWidth() < dstTy.getIntOrFloatBitWidth()) {
    auto mode = (clangTy->isUnsignedIntegerOrEnumerationType())
                    ? cudaq::cc::CastOpMode::Unsigned
                    : cudaq::cc::CastOpMode::Signed;
    return builder.create<cudaq::cc::CastOp>(loc, dstTy, srcVal, mode);
  }
  assert(fromTy.getIntOrFloatBitWidth() > dstTy.getIntOrFloatBitWidth());
  return builder.create<cudaq::cc::CastOp>(loc, dstTy, srcVal);
}

/// Generalized kernel argument morphing. When traversing the AST, the calling
/// context's argument values that have already been created may be similar to
/// but not identical to the callee's signature types. This function deals with
/// adding the glue code to make the call strongly (exactly) type conforming.
SmallVector<Value> QuakeBridgeVisitor::convertKernelArgs(
    Location loc, std::size_t dropFrontNum, const SmallVector<Value> &args,
    ArrayRef<Type> kernelArgTys, clang::CallExpr *x) {
  SmallVector<Value> result;
  assert(args.size() - dropFrontNum == kernelArgTys.size());
  for (auto i = dropFrontNum, end = args.size(); i < end; ++i) {
    auto v = args[i];
    auto vTy = v.getType();
    auto kTy = kernelArgTys[i - dropFrontNum];
    if (vTy == kTy) {
      result.push_back(v);
      continue;
    }
    if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(vTy)) {
      auto eleTy = ptrTy.getElementType();
      if (eleTy == kTy) {
        // Promote pass-by-reference to pass-by-value.
        auto load = builder.create<cudaq::cc::LoadOp>(loc, v);
        result.push_back(load);
        continue;
      }

      // We've passed clang++'s semantics checks but the types are distinct.
      if (isa<cudaq::cc::PointerType>(kTy)) {
        result.push_back(builder.create<cudaq::cc::CastOp>(loc, kTy, v));
        continue;
      }
      auto load = builder.create<cudaq::cc::LoadOp>(loc, v);
      auto loadTy = load.getType();
      Value castTo;
      if (isa<IntegerType>(loadTy) && isa<IntegerType>(kTy)) {
        castTo = integerCoercion(loc, x->getArg(i)->getType(), kTy, load);
      } else if (isa<FloatType>(loadTy) && isa<FloatType>(kTy)) {
        castTo = floatingPointCoercion(loc, kTy, load);
      } else if (isa<FloatType>(loadTy) && isa<IntegerType>(kTy)) {
        TODO_loc(loc, "implicit argument type conversion (fp to integral)");
      } else if (isa<IntegerType>(loadTy) && isa<FloatType>(kTy)) {
        TODO_loc(loc, "implicit argument type conversion (integral to fp)");
      }
      result.push_back(castTo);
      continue;
    }
    if (auto vVecTy = dyn_cast<quake::VeqType>(vTy))
      if (auto kVecTy = dyn_cast<quake::VeqType>(kTy)) {
        // Both are Veq but the Veq are not identical. If the callee has a
        // dynamic size, we can relax the size from the calling context.
        if (vVecTy.hasSpecifiedSize() && !kVecTy.hasSpecifiedSize()) {
          auto relax = builder.create<quake::RelaxSizeOp>(loc, kVecTy, v);
          result.push_back(relax);
          continue;
        }
      }

    LLVM_DEBUG(llvm::dbgs() << "convert: " << v << "\nto:" << kTy << '\n');
    TODO_loc(loc, "argument type conversion");
  }
  return result;
}

bool QuakeBridgeVisitor::TraverseCastExpr(clang::CastExpr *x,
                                          DataRecursionQueue *) {
  // RecursiveASTVisitor is tuned for dumping surface syntax so doesn't
  // necessarily visit the type. Override so that the casted to type is visited
  // and pushed on the stack.
  [[maybe_unused]] auto typeStackDepth = typeStack.size();
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (!TraverseType(x->getType()))
    return false;
  assert(typeStack.size() == typeStackDepth + 1 && "must push a type");
  for (auto *sub : getStmtChildren(x))
    if (!TraverseStmt(sub))
      return false;
  bool result = WalkUpFromCastExpr(x);
  assert((!result || typeStack.size() == typeStackDepth) &&
         "must be original depth");
  return result;
}

bool QuakeBridgeVisitor::VisitCastExpr(clang::CastExpr *x) {
  // The type to cast the expression into is pushed during the traversal of the
  // ImplicitCastExpr in non-error cases.
  auto castToTy = popType();
  auto loc = toLocation(x);
  auto intToIntCast = [&](Location locSub, Value mlirVal) {
    clang::QualType srcTy = x->getSubExpr()->getType();
    // Check for and handle reference to integer cases.
    if (isa<cudaq::cc::PointerType>(mlirVal.getType()))
      mlirVal = builder.create<cudaq::cc::LoadOp>(loc, mlirVal);
    return pushValue(integerCoercion(locSub, srcTy, castToTy, mlirVal));
  };

  switch (x->getCastKind()) {
  case clang::CastKind::CK_LValueToRValue: {
    auto subValue = loadLValue(popValue());
    return pushValue(subValue);
  }
  case clang::CastKind::CK_BitCast: {
    auto value = popValue();
    return pushValue(builder.create<cudaq::cc::CastOp>(loc, castToTy, value));
  }
  case clang::CastKind::CK_FloatingCast: {
    [[maybe_unused]] auto dstType = x->getType();
    [[maybe_unused]] auto val = x->getSubExpr();
    assert(val->getType()->isFloatingType() && dstType->isFloatingType());
    auto value = popValue();
    auto toType = cast<FloatType>(castToTy);
    auto fromType = cast<FloatType>(value.getType());
    assert(toType && fromType);
    if (toType == fromType)
      return pushValue(value);
    return pushValue(builder.create<cudaq::cc::CastOp>(loc, toType, value));
  }
  case clang::CastKind::CK_IntegralCast: {
    auto locSub = toLocation(x->getSubExpr());
    auto result = intToIntCast(locSub, popValue());
    assert(result && "integer conversion failed");
    return result;
  }
  case clang::CastKind::CK_FunctionToPointerDecay:
  case clang::CastKind::CK_ArrayToPointerDecay:
  case clang::CastKind::CK_NoOp:
  case clang::CastKind::CK_ToVoid:
  case clang::CastKind::CK_BuiltinFnToFnPtr:
    return true;
  case clang::CastKind::CK_FloatingToIntegral: {
    auto qualTy = x->getType();
    auto mode = qualTy->isUnsignedIntegerOrEnumerationType()
                    ? cudaq::cc::CastOpMode::Unsigned
                    : cudaq::cc::CastOpMode::Signed;
    return pushValue(
        builder.create<cudaq::cc::CastOp>(loc, castToTy, popValue(), mode));
  }
  case clang::CastKind::CK_IntegralToFloating: {
    auto mode =
        (x->getSubExpr()->getType()->isUnsignedIntegerOrEnumerationType())
            ? cudaq::cc::CastOpMode::Unsigned
            : cudaq::cc::CastOpMode::Signed;
    return pushValue(
        builder.create<cudaq::cc::CastOp>(loc, castToTy, popValue(), mode));
  }
  case clang::CastKind::CK_IntegralToBoolean: {
    auto last = popValue();
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, last.getType());
    return pushValue(builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, last, zero));
  }
  case clang::CastKind::CK_FloatingToBoolean: {
    auto last = popValue();
    Value zero = opt::factory::createFloatConstant(
        loc, builder, 0.0, cast<FloatType>(last.getType()));
    return pushValue(builder.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNE, last, zero));
  }
  case clang::CastKind::CK_UserDefinedConversion: {
    auto sub = popValue();
    // castToTy is the converion function signature.
    castToTy = popType();
    if (isa<IntegerType>(castToTy) && isa<IntegerType>(sub.getType())) {
      auto locSub = toLocation(x->getSubExpr());
      bool result = intToIntCast(locSub, sub);
      assert(result && "integer conversion failed");
      return result;
    }
    TODO_loc(loc, "unhandled user-defined implicit conversion");
  }
  case clang::CastKind::CK_ConstructorConversion: {
    // Enable implicit conversion of surface types, which both map to VeqType.
    if (isa<quake::VeqType>(castToTy))
      if (auto cxxExpr = dyn_cast<clang::CXXConstructExpr>(x->getSubExpr()))
        if (cxxExpr->getNumArgs() == 1 &&
            isa<quake::VeqType>(peekValue().getType()))
          return true;
    // ... or which both map to RefType.
    if (isa<quake::RefType>(castToTy))
      if (auto cxxExpr = dyn_cast<clang::CXXConstructExpr>(x->getSubExpr()))
        if (cxxExpr->getNumArgs() == 1 &&
            isa<quake::RefType>(peekValue().getType()))
          return true;

    // Enable implicit conversion of lambda -> std::function, which are both
    // cc::CallableType.
    if (isa<cc::CallableType>(castToTy)) {
      // Enable implicit conversion of callable -> std::function.
      if (auto cxxExpr = dyn_cast<clang::CXXConstructExpr>(x->getSubExpr()))
        if (cxxExpr->getNumArgs() == 1)
          return true;
    }
    if (isa<ComplexType>(castToTy) && isa<ComplexType>(peekValue().getType()))
      return true;
    if (isa<quake::StateType>(castToTy))
      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(peekValue().getType()))
        if (isa<quake::StateType>(ptrTy.getElementType()))
          return pushValue(builder.create<cudaq::cc::LoadOp>(loc, popValue()));
    if (auto funcTy = peelPointerFromFunction(castToTy))
      if (auto fromTy = dyn_cast<cc::CallableType>(peekValue().getType())) {
        auto inputs = funcTy.getInputs();
        if (!inputs.empty() && inputs[0] == fromTy)
          return true;
      }

    TODO_loc(loc, "unhandled implicit cast expression: constructor conversion");
  }
  }

  // Handle the case where we have if ( vec[i] ), where vec == vector<i32>.
  // This leads to an ImplicitCastExpr (IntegralToBoolean) -> ImplicitCastExpr
  // (LvalueToRvalue)
  if (auto anotherCast = dyn_cast<clang::ImplicitCastExpr>(x->getSubExpr())) {
    if (!VisitImplicitCastExpr(anotherCast))
      return false;
    if (x->getCastKind() == clang::CastKind::CK_IntegralToBoolean) {
      auto last = popValue();
      Value zero = builder.create<arith::ConstantIntOp>(loc, 0, last.getType());
      return pushValue(builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, last, zero));
    }
  }
  TODO_loc(loc, "unhandled implicit cast expression");
}

bool QuakeBridgeVisitor::TraverseBinaryOperator(clang::BinaryOperator *x,
                                                DataRecursionQueue *) {
  bool shortCircuitWhenTrue =
      x->getOpcode() == clang::BinaryOperatorKind::BO_LOr;

  // The && and || operators are semantically if statements. Traverse them
  // differently than other expressions since both sides of the expression are
  // not always evaluated.
  switch (x->getOpcode()) {
  case clang::BinaryOperatorKind::BO_LAnd:
  case clang::BinaryOperatorKind::BO_LOr: {
    auto *lhs = x->getLHS();
    if (!TraverseStmt(lhs))
      return false;
    auto lhsVal = popValue();
    auto loc = toLocation(x->getSourceRange());
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, lhsVal.getType());
    Value cond = builder.create<arith::CmpIOp>(loc,
                                               shortCircuitWhenTrue
                                                   ? arith::CmpIPredicate::ne
                                                   : arith::CmpIPredicate::eq,
                                               lhsVal, zero);
    bool result = true;
    auto ifOp = builder.create<cc::IfOp>(
        loc, TypeRange{cond.getType()}, cond,
        // Value if `cond` is true
        // For `BO_LAnd`, that means Value if lhs is     zero (i.e. false)
        // For `BO_LOr`,  that means Value if lhs is non-zero (i.e. true)
        [=](OpBuilder &builder, Location loc, Region &region) {
          // Short-circuit taken: return the result of the lhs and do not
          // evaluate the rhs at all.
          region.push_back(new Block{});
          auto &bodyBlock = region.front();
          OpBuilder::InsertionGuard guad(builder);
          builder.setInsertionPointToStart(&bodyBlock);
          if (x->getOpcode() == clang::BinaryOperatorKind::BO_LAnd) {
            // Return false out of this block in order to avoid evaluating rhs
            auto constantFalse =
                builder
                    .create<arith::ConstantOp>(loc, builder.getBoolAttr(false))
                    .getResult();
            builder.create<cc::ContinueOp>(loc, TypeRange{}, constantFalse);
          } else {
            builder.create<cc::ContinueOp>(loc, TypeRange{}, cond);
          }
        },
        // Value if `cond` is false
        // For `BO_LAnd`, that means Value if lhs is non-zero (i.e. true)
        // For `BO_LOr`,  that means Value if lhs is     zero (i.e. false)
        [&result, this, rhs = x->getRHS()](OpBuilder &builder, Location loc,
                                           Region &region) {
          // Short-circuit not taken: evaluate the rhs and return that value.
          region.push_back(new Block{});
          auto &bodyBlock = region.front();
          OpBuilder::InsertionGuard guad(builder);
          builder.setInsertionPointToStart(&bodyBlock);
          if (!TraverseStmt(rhs)) {
            result = false;
            return;
          }
          auto rhsVal = popValue();
          builder.create<cc::ContinueOp>(loc, TypeRange{}, rhsVal);
        });
    if (!result)
      return result;
    return pushValue(ifOp.getResult(0));
  }
  default:
    break;
  }
  return Base::TraverseBinaryOperator(x);
}

bool QuakeBridgeVisitor::VisitBinaryOperator(clang::BinaryOperator *x) {
  auto rhs = popValue();
  auto lhs = popValue();
  auto loc = toLocation(x->getSourceRange());
  auto maybeLoadValue = [&](Value v) -> Value {
    if (isa<cc::PointerType>(v.getType()))
      return builder.create<cc::LoadOp>(loc, v);
    return v;
  };

  if (x->isRelationalOp() ||
      x->getOpcode() == clang::BinaryOperatorKind::BO_EQ ||
      x->getOpcode() == clang::BinaryOperatorKind::BO_NE) {
    rhs = maybeLoadValue(rhs);
    lhs = maybeLoadValue(lhs);
    // Floating point comparison?
    if (isa<FloatType>(lhs.getType())) {
      arith::CmpFPredicate pred;
      switch (x->getOpcode()) {
      case clang::BinaryOperatorKind::BO_LT:
        pred = arith::CmpFPredicate::OLT;
        break;
      case clang::BinaryOperatorKind::BO_LE:
        pred = arith::CmpFPredicate::OLE;
        break;
      case clang::BinaryOperatorKind::BO_EQ:
        pred = arith::CmpFPredicate::OEQ;
        break;
      case clang::BinaryOperatorKind::BO_GT:
        pred = arith::CmpFPredicate::OGT;
        break;
      case clang::BinaryOperatorKind::BO_GE:
        pred = arith::CmpFPredicate::OGE;
        break;
      case clang::BinaryOperatorKind::BO_NE:
        pred = arith::CmpFPredicate::ONE;
        break;
      default:
        TODO_loc(loc, "floating-point comparison");
      }
      return pushValue(builder.create<arith::CmpFOp>(loc, pred, lhs, rhs));
    }
    arith::CmpIPredicate pred;
    auto lhsTy = x->getLHS()->getType();
    auto rhsTy = x->getRHS()->getType();
    // Favor unsigned comparisons via implicit conversion.
    bool isUnsigned = lhsTy->isUnsignedIntegerOrEnumerationType() ||
                      rhsTy->isUnsignedIntegerOrEnumerationType();
    switch (x->getOpcode()) {
    case clang::BinaryOperatorKind::BO_LT:
      pred = isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
      break;
    case clang::BinaryOperatorKind::BO_LE:
      pred = isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
      break;
    case clang::BinaryOperatorKind::BO_EQ:
      pred = arith::CmpIPredicate::eq;
      break;
    case clang::BinaryOperatorKind::BO_GT:
      pred = isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
      break;
    case clang::BinaryOperatorKind::BO_GE:
      pred = isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge;
      break;
    case clang::BinaryOperatorKind::BO_NE:
      pred = arith::CmpIPredicate::ne;
      break;
    default:
      TODO_loc(loc, "integer comparison");
    }
    return pushValue(builder.create<arith::CmpIOp>(loc, pred, lhs, rhs));
  }

  switch (x->getOpcode()) {
  case clang::BinaryOperatorKind::BO_Assign: {
    builder.create<cc::StoreOp>(loc, rhs, lhs);
    return pushValue(lhs);
  }
  case clang::BinaryOperatorKind::BO_AddAssign:
  case clang::BinaryOperatorKind::BO_SubAssign:
  case clang::BinaryOperatorKind::BO_MulAssign:
  case clang::BinaryOperatorKind::BO_DivAssign:
  case clang::BinaryOperatorKind::BO_ShlAssign:
  case clang::BinaryOperatorKind::BO_ShrAssign:
  case clang::BinaryOperatorKind::BO_OrAssign:
  case clang::BinaryOperatorKind::BO_XorAssign:
  case clang::BinaryOperatorKind::BO_AndAssign:
    return true; // see CompoundAssignOperator
  default:
    break;
  }
  rhs = maybeLoadValue(rhs);
  lhs = maybeLoadValue(lhs);
  castToSameType(builder, loc, x->getLHS()->getType().getTypePtrOrNull(), lhs,
                 x->getRHS()->getType().getTypePtrOrNull(), rhs);
  switch (x->getOpcode()) {
  case clang::BinaryOperatorKind::BO_Add: {
    if (x->getType()->isIntegerType())
      return pushValue(builder.create<arith::AddIOp>(loc, lhs, rhs));
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::AddFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_add binary op");
  }
  case clang::BinaryOperatorKind::BO_Rem: {
    if (x->getType()->isIntegerType()) {
      if (x->getType()->isUnsignedIntegerOrEnumerationType())
        return pushValue(builder.create<arith::RemUIOp>(loc, lhs, rhs));
      return pushValue(builder.create<arith::RemSIOp>(loc, lhs, rhs));
    }
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::AddFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_add binary op");
  }
  case clang::BinaryOperatorKind::BO_Sub: {
    if (x->getType()->isIntegerType())
      return pushValue(builder.create<arith::SubIOp>(loc, lhs, rhs));
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::SubFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_add binary op");
  }

  case clang::BinaryOperatorKind::BO_Mul: {
    if (x->getType()->isIntegerType())
      return pushValue(builder.create<arith::MulIOp>(loc, lhs, rhs));
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::MulFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_mul binary op");
  }

  case clang::BinaryOperatorKind::BO_Div: {
    if (x->getType()->isIntegerType()) {
      if (x->getType()->isUnsignedIntegerOrEnumerationType())
        return pushValue(builder.create<arith::DivUIOp>(loc, lhs, rhs));
      return pushValue(builder.create<arith::DivSIOp>(loc, lhs, rhs));
    }
    if (x->getType()->isFloatingType())
      return pushValue(builder.create<arith::DivFOp>(loc, lhs, rhs));
    TODO_loc(loc, "error in bo_div binary op");
  }

  case clang::BinaryOperatorKind::BO_Shl:
    return pushValue(builder.create<arith::ShLIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_Shr:
    if (x->getLHS()->getType()->isUnsignedIntegerOrEnumerationType())
      return pushValue(builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs));
    return pushValue(builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_Or:
    return pushValue(builder.create<arith::OrIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_Xor:
    return pushValue(builder.create<arith::XOrIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_And:
    return pushValue(builder.create<arith::AndIOp>(loc, lhs, rhs));
  case clang::BinaryOperatorKind::BO_LAnd:
  case clang::BinaryOperatorKind::BO_LOr:
    emitFatalError(loc, "&& and || ops are handled elsewhere.");
  }
  TODO_loc(loc, "unknown binary kind operator");
}

std::string QuakeBridgeVisitor::genLoweredName(clang::FunctionDecl *x,
                                               FunctionType funcTy) {
  auto loc = toLocation(x->getSourceRange());
  std::string result = [&]() {
    for (auto &pair : functionsToEmit)
      if (x == pair.second)
        return generateCudaqKernelName(pair);
    return cxxMangledDeclName(x);
  }();
  // Add the called function to the module as needed.
  getOrAddFunc(loc, result, funcTy);
  return result;
}

bool QuakeBridgeVisitor::TraverseConditionalOperator(
    clang::ConditionalOperator *x, DataRecursionQueue *q) {
  bool result = true;
  auto loc = toLocation(x->getSourceRange());
  if (!TraverseStmt(x->getCond()))
    return false;
  auto condVal = popValue();
  Type resultTy = builder.getI64Type();

  // Create shared lambda for the x->getTrueExpr() and x->getFalseExpr()
  // expressions
  auto thenElseLambda = [&](clang::Expr *thenOrElse) {
    return [&, thenOrElse](OpBuilder &builder, Location loc, Region &region) {
      region.push_back(new Block{});
      auto &bodyBlock = region.front();
      OpBuilder::InsertionGuard guad(builder);
      builder.setInsertionPointToStart(&bodyBlock);
      if (!TraverseStmt(thenOrElse)) {
        result = false;
        return;
      }
      Value resultVal = popValue();
      builder.create<cc::ContinueOp>(loc, TypeRange{}, resultVal);
      resultTy = resultVal.getType();
    };
  };

  auto ifOp = builder.create<cc::IfOp>(loc, TypeRange{resultTy}, condVal,
                                       thenElseLambda(x->getTrueExpr()),
                                       thenElseLambda(x->getFalseExpr()));

  if (!result)
    return result;
  ifOp.getResult(0).setType(resultTy);
  return pushValue(ifOp.getResult(0));
}

bool QuakeBridgeVisitor::VisitMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  auto ty = peekValue().getType();

  // The following cases are λ expressions, quantum data, or a std::vector view.
  // In those cases, there is nothing to materialize, so we can just pass the
  // Value on the top of the stack.
  if (isa<cc::CallableType, quake::VeqType, quake::RefType, cc::SpanLikeType,
          quake::StateType>(ty))
    return true;

  // If not one of the above special cases, then materialize the value to a
  // temporary memory location and push the address to the stack.

  // Is it already materialized in memory?
  if (isa<cc::PointerType>(ty))
    return true;

  // Materialize the value into a glvalue location in memory.
  auto materialize = builder.create<cc::AllocaOp>(loc, ty);
  builder.create<cc::StoreOp>(loc, popValue(), materialize);
  return pushValue(materialize);
}

bool QuakeBridgeVisitor::TraverseLambdaExpr(clang::LambdaExpr *x,
                                            DataRecursionQueue *) {
  auto loc = toLocation(x->getSourceRange());
  bool result = true;
  if (!x->explicit_captures().empty()) {
    // Lambda expression with explicit capture list is not supported yet.
    TODO_x(loc, x, mangler, "lambda expression with explicit captures");
  }
  if (!TraverseType(x->getType()))
    return false;
  auto callableTy = cast<cc::CallableType>(popType());
  auto lambdaInstance = builder.create<cc::CreateLambdaOp>(
      loc, callableTy, [&](OpBuilder &builder, Location loc) {
        // FIXME: the capture list, etc. should be visited in an appropriate
        // context here, not as part of lowering the body of the lambda.
        auto *entryBlock = builder.getInsertionBlock();
        SymbolTableScope argsScope(symbolTable);
        addArgumentSymbols(entryBlock, x->getCallOperator()->parameters());
        if (!TraverseStmt(x->getBody())) {
          result = false;
          return;
        }
        builder.create<cc::ReturnOp>(loc);
      });
  pushValue(lambdaInstance);
  return result;
}

bool QuakeBridgeVisitor::TraverseMemberExpr(clang::MemberExpr *x,
                                            DataRecursionQueue *) {
  if (auto *methodDecl = dyn_cast<clang::CXXMethodDecl>(x->getMemberDecl())) {
    // For function members, we want to push the type of the function, since
    // the visit to CallExpr requires a type to have been pushed.
    [[maybe_unused]] auto typeStackDepth = typeStack.size();
    if (!TraverseType(methodDecl->getType()))
      return false;
    assert(typeStack.size() == typeStackDepth + 1);
  }
  if (auto *field = dyn_cast<clang::FieldDecl>(x->getMemberDecl())) {
    [[maybe_unused]] auto typeStackDepth = typeStack.size();
    if (!TraverseType(field->getType()))
      return false;
    assert(typeStack.size() == typeStackDepth + 1);
  }
  return Base::TraverseMemberExpr(x);
}

bool QuakeBridgeVisitor::VisitMemberExpr(clang::MemberExpr *x) {
  if (auto *field = dyn_cast<clang::FieldDecl>(x->getMemberDecl())) {
    auto loc = toLocation(x->getSourceRange());
    auto object = popValue(); // DeclRefExpr
    auto ty = popType();
    std::int32_t offset = field->getFieldIndex();
    if (isa<quake::StruqType>(object.getType())) {
      return pushValue(
          builder.create<quake::GetMemberOp>(loc, ty, object, offset));
    }
    if (!isa<cc::PointerType>(object.getType())) {
      reportClangError(x, mangler,
                       "internal error: struct must be an object in memory");
      return false;
    }
    auto eleTy = cast<cc::PointerType>(object.getType()).getElementType();
    SmallVector<cc::ComputePtrArg> offsets;
    if (auto arrTy = dyn_cast<cc::ArrayType>(eleTy))
      if (arrTy.isUnknownSize())
        offsets.push_back(0);
    offsets.push_back(offset);
    return pushValue(builder.create<cc::ComputePtrOp>(
        loc, cc::PointerType::get(ty), object, offsets));
  }
  return true;
}

bool QuakeBridgeVisitor::VisitUnaryExprOrTypeTraitExpr(
    clang::UnaryExprOrTypeTraitExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  auto i64Ty = builder.getI64Type();
  switch (x->getKind()) {
  case clang::UnaryExprOrTypeTrait::UETT_SizeOf:
    return pushValue(
        builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, popType()));
  default:
    break;
  }
  return Base::VisitUnaryExprOrTypeTraitExpr(x);
}

bool QuakeBridgeVisitor::visitMathLibFunc(clang::CallExpr *x,
                                          clang::FunctionDecl *func,
                                          Location loc, StringRef funcName) {
  // Handle any std::pow(N,M)
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "pow" || funcName == "powf")) {
    auto funcArity = func->getNumParams();
    SmallVector<Value> args = lastValues(funcArity);
    auto powFun = popValue();

    // Get the values involved
    auto peelIntToFloat = [&](Value v) {
      if (auto op = v.getDefiningOp<cudaq::cc::CastOp>())
        return op.getOperand();
      return v;
    };
    Value base = peelIntToFloat(args[0]);
    Value power = peelIntToFloat(args[1]);
    Type baseType = base.getType();
    Type powerType = power.getType();

    // Create the power op based on the types of the arguments.
    if (isa<IntegerType>(powerType)) {
      if (isa<IntegerType>(baseType)) {
        auto calleeTy = peelPointerFromFunction(powFun.getType());
        auto resTy = calleeTy.getResult(0);
        castToSameType(builder, loc, x->getArg(0)->getType().getTypePtrOrNull(),
                       base, x->getArg(1)->getType().getTypePtrOrNull(), power);
        auto ipow = builder.create<math::IPowIOp>(loc, base, power);
        if (isa<FloatType>(resTy))
          return pushValue(builder.create<cudaq::cc::CastOp>(
              loc, resTy, ipow, cudaq::cc::CastOpMode::Signed));
        assert(resTy == ipow.getType());
        return pushValue(ipow);
      }
      return pushValue(builder.create<math::FPowIOp>(loc, base, power));
    }
    return pushValue(builder.create<math::PowFOp>(loc, base, power));
  }

  auto floatOperator = [&]<typename Op>(Op, const char *dblName) -> bool {
    assert(func->getNumParams() == 1 && "must be unary");
    Value arg = popValue();
    [[maybe_unused]] auto funcConst = popValue();
    if (isa<IntegerType>(arg.getType()))
      arg = builder.create<cudaq::cc::CastOp>(
          loc,
          funcName == dblName ? builder.getF64Type() : builder.getF32Type(),
          arg,
          x->getArg(0)->getType()->isUnsignedIntegerOrEnumerationType()
              ? cudaq::cc::CastOpMode::Unsigned
              : cudaq::cc::CastOpMode::Signed);
    return pushValue(builder.create<Op>(loc, arg));
  };

  // Handle std::sqrt
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "sqrt" || funcName == "sqrtf"))
    return floatOperator(math::SqrtOp{}, "sqrt");

  // Handle std::round
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "round" || funcName == "roundf"))
    return floatOperator(math::RoundOp{}, "round");

  // Handle std::abs
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "abs" || funcName == "fabs" || funcName == "fabsf")) {
    assert(func->getNumParams() == 1 && "must be unary");
    Value arg = popValue();
    [[maybe_unused]] auto funcConst = popValue();
    if (isa<IntegerType>(arg.getType()))
      return pushValue(builder.create<math::AbsIOp>(loc, arg));
    return pushValue(builder.create<math::AbsFOp>(loc, arg));
  }

  // Handle std::sin
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "sin" || funcName == "sinf"))
    return floatOperator(math::SinOp{}, "sin");

  // Handle std::cos
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "cos" || funcName == "cosf"))
    return floatOperator(math::CosOp{}, "cos");

  // Handle std::tan
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "tan" || funcName == "tanf"))
    return floatOperator(math::TanOp{}, "tan");

  // Handle std::exp
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "exp" || funcName == "expf"))
    return floatOperator(math::ExpOp{}, "exp");

  // Handle std::log
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "log" || funcName == "logf"))
    return floatOperator(math::LogOp{}, "log");

  // Handle std::ceil
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "ceil" || funcName == "ceilf"))
    return floatOperator(math::CeilOp{}, "ceil");

  // Handle std::floor
  if ((isInNamespace(func, "std") || isNotInANamespace(func)) &&
      (funcName == "floor" || funcName == "floorf"))
    return floatOperator(math::FloorOp{}, "floor");

  return false;
}

bool QuakeBridgeVisitor::VisitCallExpr(clang::CallExpr *x) {
  auto loc = toLocation(x->getSourceRange());
  // The called function is reified as a Value in the IR.
  auto *callee = x->getCalleeDecl();
  auto *func = dyn_cast<clang::FunctionDecl>(callee);
  if (!func)
    TODO_loc(loc, "call doesn't have function decl");
  assert(valueStack.size() >= x->getNumArgs() + 1 &&
         "stack must contain all arguments plus the expression to call");
  StringRef funcName;
  if (auto *id = func->getIdentifier())
    funcName = id->getName();

  if (visitMathLibFunc(x, func, loc, funcName))
    return true;

  // Handle std::complex member functions
  if (isInClassInNamespace(func, "complex", "std")) {
    auto value = popValue();
    if (isa<cc::PointerType>(value.getType()))
      value = builder.create<cc::LoadOp>(loc, value);
    if (funcName == "real") {
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          return pushValue(builder.create<complex::ReOp>(loc, value));
        }
    }
    if (funcName == "imag") {
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          return pushValue(builder.create<complex::ImOp>(loc, value));
        }
    }
  }

  // Dealing with our std::vector as a view data structures. If we have some θ
  // with the type `std::vector<double/float/int>`, and in the kernel, θ.size()
  // is called, we need to convert that to loading the size field of the pair.
  // For θ.empty(), the size is loaded and compared to zero.
  if (isInClassInNamespace(func, "vector", "std")) {
    // Get the size of the std::vector.
    auto svec = popValue();
    if (isa<cc::PointerType>(svec.getType()))
      svec = builder.create<cc::LoadOp>(loc, svec);
    auto ext =
        builder.create<cc::StdvecSizeOp>(loc, builder.getI64Type(), svec);
    if (funcName == "size")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          return pushValue(ext);
        }
    if (funcName == "empty")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          return pushValue(builder.create<mlir::arith::CmpIOp>(
              ext->getLoc(), arith::CmpIPredicate(arith::CmpIPredicate::eq),
              ext.getResult(),
              getConstantInt(
                  builder, ext->getLoc(), 0,
                  ext.getResult().getType().getIntOrFloatBitWidth())));
        }
    if (funcName == "front" || funcName == "begin")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto eleTy = cast<cc::SpanLikeType>(svec.getType()).getElementType();
          auto elePtrTy = cc::PointerType::get(eleTy);
          return pushValue(
              builder.create<cc::StdvecDataOp>(loc, elePtrTy, svec));
        }
    if (funcName == "back" || funcName == "rbegin")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto negativeOneIndex = getConstantInt(builder, loc, -1, 64);
          auto eleTy = cast<cc::SpanLikeType>(svec.getType()).getElementType();
          auto eleArrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
          auto elePtrTy = cc::PointerType::get(eleTy);
          auto *ctx = eleTy.getContext();
          auto i64Ty = mlir::IntegerType::get(ctx, 64);
          auto vecPtr = builder.create<cc::StdvecDataOp>(loc, eleArrTy, svec);
          auto vecLen = builder.create<cc::StdvecSizeOp>(loc, i64Ty, svec);
          Value vecLenMinusOne =
              builder.create<arith::AddIOp>(loc, vecLen, negativeOneIndex);
          return pushValue(builder.create<cc::ComputePtrOp>(
              loc, elePtrTy, vecPtr, ValueRange{vecLenMinusOne}));
        }
    if (funcName == "end")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto eleTy = cast<cc::SpanLikeType>(svec.getType()).getElementType();
          auto elePtrTy = cc::PointerType::get(eleTy);
          auto eleArrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
          auto *ctx = eleTy.getContext();
          auto i64Ty = mlir::IntegerType::get(ctx, 64);
          auto vecPtr = builder.create<cc::StdvecDataOp>(loc, eleArrTy, svec);
          Value vecLen = builder.create<cc::StdvecSizeOp>(loc, i64Ty, svec);
          return pushValue(builder.create<cc::ComputePtrOp>(
              loc, elePtrTy, vecPtr, ValueRange{vecLen}));
        }
    if (funcName == "rend")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          Value negativeOneIndex =
              builder.create<arith::ConstantIntOp>(loc, -1, 64);
          auto eleTy = cast<cc::SpanLikeType>(svec.getType()).getElementType();
          auto elePtrTy = cc::PointerType::get(eleTy);
          auto eleArrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
          auto vecPtr = builder.create<cc::StdvecDataOp>(loc, eleArrTy, svec);
          return pushValue(builder.create<cc::ComputePtrOp>(
              loc, elePtrTy, vecPtr, ValueRange{negativeOneIndex}));
        }
    if (funcName == "data")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          // data() returns a pointer to a sequence of elements.
          auto eleTy = cast<cc::SpanLikeType>(svec.getType()).getElementType();
          auto eleArrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
          return pushValue(
              builder.create<cc::StdvecDataOp>(loc, eleArrTy, svec));
        }

    TODO_loc(loc, "unhandled std::vector member function, " + funcName);
  }

  if (isInClassInNamespace(func, "_Bit_reference", "std") ||
      isInClassInNamespace(func, "__bit_reference", "std") ||
      isInClassInNamespace(func, "__bit_const_reference", "std")) {
    // Calling std::_Bit_reference::method().
    auto loadFromReference = [&](mlir::Value ref) -> Value {
      if (auto mrTy = dyn_cast<cc::PointerType>(ref.getType())) {
        auto loadVal = builder.create<cc::LoadOp>(loc, ref);
        if (mrTy.getElementType() == builder.getI8Type())
          return builder.create<cc::CastOp>(loc, builder.getI1Type(), loadVal);
        return loadVal;
      }
      assert(ref.getType() == builder.getI1Type());
      return ref;
    };
    if (isa<clang::CXXConversionDecl>(func)) {
      assert(isa<cc::PointerType>(peekValue().getType()));
      return pushValue(builder.create<cc::LoadOp>(loc, popValue()));
    }
    if (func->isOverloadedOperator()) {
      auto overloadedOperator = func->getOverloadedOperator();
      if (isCompareEqualOperator(overloadedOperator)) {
        auto rhs = loadFromReference(popValue());
        auto lhs = loadFromReference(popValue());
        popValue(); // The compare equal operator address.
        return pushValue(builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, lhs, rhs));
      }
      if (isAssignmentOperator(overloadedOperator)) {
        auto rhs = loadFromReference(popValue());
        auto lhs = popValue();
        popValue(); // The assignment operator address.
        if (rhs.getType() == builder.getI1Type()) {
          // If we're storing a bool, we may have to zext the value to a byte.
          auto ptrTy = cast<cc::PointerType>(lhs.getType());
          auto eleTy = ptrTy.getElementType();
          if (auto arrTy = dyn_cast<cc::ArrayType>(eleTy))
            eleTy = arrTy.getElementType();
          if (eleTy != rhs.getType())
            rhs = builder.create<cc::CastOp>(loc, eleTy, rhs,
                                             cc::CastOpMode::Unsigned);
        }
        builder.create<cc::StoreOp>(loc, rhs, lhs);
        return pushValue(loadFromReference(lhs));
      }
      if (isSubscriptOperator(overloadedOperator)) {
        auto rhs = loadFromReference(popValue());
        auto lhs = popValue();
        popValue(); // The subscript operator address.
        auto bytePtrTy = cc::PointerType::get(builder.getI8Type());
        return pushValue(
            builder.create<cc::ComputePtrOp>(loc, bytePtrTy, lhs, rhs));
      }
    }
    TODO_loc(loc, "unhandled std::vector<bool> member function, " + funcName);
  }

  if (isInClassInNamespace(func, "qreg", "cudaq") ||
      isInClassInNamespace(func, "qvector", "cudaq") ||
      isInClassInNamespace(func, "qarray", "cudaq") ||
      isInClassInNamespace(func, "qspan", "cudaq") ||
      isInClassInNamespace(func, "qview", "cudaq")) {
    // This handles conversion of qreg.size()
    if (funcName == "size")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto qregArg = popValue();
          auto qrSize = builder.create<quake::VeqSizeOp>(
              loc, builder.getI64Type(), qregArg);
          return pushValue(qrSize);
        }

    if (funcName == "front")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto actArgs = lastValues(x->getNumArgs());
          auto qregArg = popValue();
          auto zero = getConstantInt(builder, loc, 0, 64);
          if (actArgs.size() == 1) {
            // Handle `r.front(n)` case.
            auto qrSize = actArgs.front();
            auto one = getConstantInt(builder, loc, 1, 64);
            auto offset = builder.create<arith::SubIOp>(loc, qrSize, one);
            auto unsizedVecTy =
                quake::VeqType::getUnsized(builder.getContext());
            return pushValue(builder.create<quake::SubVeqOp>(
                loc, unsizedVecTy, qregArg, zero, offset));
          }
          assert(actArgs.size() == 0);
          return pushValue(
              builder.create<quake::ExtractRefOp>(loc, qregArg, zero));
        }

    if (funcName == "back")
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto actArgs = lastValues(x->getNumArgs());
          auto qregArg = popValue();
          auto qrSize = builder.create<quake::VeqSizeOp>(
              loc, builder.getI64Type(), qregArg);
          auto one = getConstantInt(builder, loc, 1, 64);
          auto endOff = builder.create<arith::SubIOp>(loc, qrSize, one);
          if (actArgs.size() == 1) {
            // Handle `r.back(n)` case.
            auto startOff =
                builder.create<arith::SubIOp>(loc, qrSize, actArgs.front());
            auto unsizedVecTy =
                quake::VeqType::getUnsized(builder.getContext());
            return pushValue(builder.create<quake::SubVeqOp>(
                loc, unsizedVecTy, qregArg, startOff, endOff));
          }
          assert(actArgs.size() == 0);
          return pushValue(
              builder.create<quake::ExtractRefOp>(loc, qregArg, endOff));
        }

    if (funcName == "slice") {
      if (auto memberCall = dyn_cast<clang::CXXMemberCallExpr>(x))
        if (memberCall->getImplicitObjectArgument()) {
          [[maybe_unused]] auto calleeTy = popType();
          assert(isa<FunctionType>(calleeTy));
          auto actArgs = lastValues(x->getNumArgs());
          auto qregArg = popValue();
          auto start = actArgs[0];
          auto count = actArgs[1];

          auto one = getConstantInt(builder, loc, 1, 64);
          Value offset = builder.create<arith::AddIOp>(loc, start, count);
          offset = builder.create<arith::SubIOp>(loc, offset, one);
          auto unsizedVecTy = quake::VeqType::getUnsized(builder.getContext());
          return pushValue(builder.create<quake::SubVeqOp>(
              loc, unsizedVecTy, qregArg, start, offset));
        }
    }

    TODO_loc(loc, "unhandled cudaq member function, " + funcName);
  }

  auto funcArity = func->getNumParams();
  SmallVector<Value> args = lastValues(funcArity);
  if (isa<clang::CXXMethodDecl>(func)) {
    [[maybe_unused]] auto thisPtrValue = popValue();
  }
  auto calleeOp = popValue();

  if (isInNamespace(func, "cudaq")) {
    // Check and see if this quantum operation is adjoint
    bool isAdjoint = false;
    bool isControl = false;
    auto *functionDecl = x->getCalleeDecl()->getAsFunction();
    if (auto *templateArgs = functionDecl->getTemplateSpecializationArgs())
      if (templateArgs->size() > 0) {
        auto gateModifierArg = templateArgs->asArray()[0];
        if (gateModifierArg.getKind() == clang::TemplateArgument::ArgKind::Type)
          if (auto *structTy =
                  gateModifierArg.getAsType()->getAsStructureType())
            if (auto structTypeAsRecord = structTy->getAsCXXRecordDecl()) {
              isAdjoint = structTypeAsRecord->getName() == "adj";
              isControl = structTypeAsRecord->getName() == "ctrl";
            }
      }

    if (funcName == "exp_pauli") {
      assert(args.size() > 2);
      SmallVector<Value> parameters;
      SmallVector<Value> targets;
      Value pauliWord;
      auto addTheString = [&](Value v) {
        // The C-string argument (char*) may be loaded by an lvalue to rvalue
        // cast. Here, we must pass the pointer and not the first character's
        // value.
        if (isCharPointerType(v.getType())) {
          pauliWord = v;
        } else if (auto load = v.getDefiningOp<cudaq::cc::LoadOp>()) {
          pauliWord = load.getPtrvalue();
        } else if (isCharspanPointerType(v.getType())) {
          // Load the char span, which is a char*
          auto span = builder.create<cc::LoadOp>(loc, v);
          pauliWord = span;
        } else if (isa<cudaq::cc::CharspanType>(v.getType())) {
          pauliWord = v;
        } else {
          reportClangError(x, mangler, "could not determine string argument");
        }
      };
      if (args.size() == 3 && isa<quake::VeqType>(args[1].getType())) {
        // Have f64, veq, string
        parameters.push_back(args[0]);
        targets.push_back(args[1]);
        addTheString(args[2]);
      } else {
        // should have f64, string, qubits...
        // need f64, veq, string, so process here

        // add f64 value
        parameters.push_back(args[0]);

        // concat the qubits to a veq
        SmallVector<Value> quantumArgs;
        for (std::size_t i = 2; i < args.size(); i++)
          quantumArgs.push_back(args[i]);
        targets.push_back(builder.create<quake::ConcatOp>(
            loc, quake::VeqType::get(builder.getContext(), quantumArgs.size()),
            quantumArgs));
        addTheString(args[1]);
      }

      builder.create<quake::ExpPauliOp>(loc, parameters, ValueRange{}, targets,
                                        pauliWord);
      return true;
    }

    if (funcName == "apply_noise") {
      SmallVector<Value> params;
      SmallVector<Value> qubits;
      bool inParams = true;
      for (auto iter : llvm::enumerate(args)) {
        auto a = iter.value();
        Type aTy = a.getType();
        if (inParams) {
          if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(aTy))
            if (isa<FloatType>(ptrTy.getElementType())) {
              params.push_back(a);
              continue;
            }
          if (auto stdvecTy = dyn_cast<cudaq::cc::StdvecType>(aTy))
            if (stdvecTy.getElementType() == builder.getF64Type() &&
                iter.index() == 0) {
              params.push_back(a);
              inParams = false;
              continue;
            }
          inParams = false;
        }
        // The first argument that is not floating-point must be a qubit. If
        // the user has interleaved floating-point and qubit arguments, that's
        // an error.
        if (isa<quake::RefType, quake::VeqType>(aTy)) {
          qubits.push_back(a);
        } else {
          reportClangError(x, mangler,
                           "apply_noise argument types not supported.");
          return false;
        }
      }

      if (auto callee = calleeOp.getDefiningOp<func::ConstantOp>()) {
        StringRef calleeName = callee.getValue();
        builder.create<quake::ApplyNoiseOp>(loc, calleeName, params, qubits);

        // Add the declaration of the function to the module.
        SmallVector<Type> argTys;
        for (auto p : params)
          argTys.push_back(p.getType());
        for (auto q : qubits)
          argTys.push_back(q.getType());
        auto calleeTy = FunctionType::get(builder.getContext(), argTys, {});
        cudaq::opt::factory::getOrAddFunc(loc, calleeName, calleeTy, module);
        return true;
      }

      reportClangError(x, mangler,
                       "apply_noise with a vector argument is deprecated.");
      return false;
    }

    if (funcName == "mx" || funcName == "my" || funcName == "mz") {
      // Measurements always return a bool or a std::vector<bool>.
      bool useStdvec =
          (args.size() > 1) ||
          (args.size() == 1 && isa<quake::VeqType>(args[0].getType()));
      auto measure = [&]() -> Value {
        Type measTy = quake::MeasureType::get(builder.getContext());
        if (useStdvec)
          measTy = cc::StdvecType::get(measTy);
        if (funcName == "mx")
          return builder.create<quake::MxOp>(loc, measTy, args).getMeasOut();
        if (funcName == "my")
          return builder.create<quake::MyOp>(loc, measTy, args).getMeasOut();
        return builder.create<quake::MzOp>(loc, measTy, args).getMeasOut();
      }();
      Type resTy = builder.getI1Type();
      if (useStdvec)
        resTy = cc::StdvecType::get(resTy);
      return pushValue(
          builder.create<quake::DiscriminateOp>(loc, resTy, measure));
    }

    // Handle the quantum gate set.
    auto reportNegateError = [&]() {
      reportClangError(x, mangler, "target qubit cannot be negated");
    };
    if (funcName == "h")
      return buildOp<quake::HOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/false,
                                 isControl);
    if (funcName == "ch")
      return buildOp<quake::HOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/false,
                                 /*control=*/true);
    if (funcName == "x")
      return buildOp<quake::XOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/false,
                                 isControl);
    if (funcName == "cnot" || funcName == "cx" || funcName == "ccx")
      return buildOp<quake::XOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/false,
                                 /*control=*/true);
    if (funcName == "y")
      return buildOp<quake::YOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/false,
                                 isControl);
    if (funcName == "cy")
      return buildOp<quake::YOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/false,
                                 /*control=*/true);
    if (funcName == "z")
      return buildOp<quake::ZOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/false,
                                 isControl);
    if (funcName == "cz")
      return buildOp<quake::ZOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/false,
                                 /*control=*/true);
    if (funcName == "s")
      return buildOp<quake::SOp>(builder, loc, args, negations,
                                 reportNegateError, isAdjoint, isControl);
    if (funcName == "cs")
      return buildOp<quake::SOp>(builder, loc, args, negations,
                                 reportNegateError, isAdjoint,
                                 /*control=*/true);
    if (funcName == "sdg")
      return buildOp<quake::SOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/true,
                                 isControl);
    if (funcName == "t")
      return buildOp<quake::TOp>(builder, loc, args, negations,
                                 reportNegateError, isAdjoint, isControl);
    if (funcName == "ct")
      return buildOp<quake::TOp>(builder, loc, args, negations,
                                 reportNegateError, isAdjoint,
                                 /*control=*/true);
    if (funcName == "tdg")
      return buildOp<quake::TOp>(builder, loc, args, negations,
                                 reportNegateError, /*adjoint=*/true,
                                 isControl);

    if (funcName == "reset") {
      if (!negations.empty())
        reportNegateError();
      return builder.create<quake::ResetOp>(loc, TypeRange{}, args[0]);
    }
    if (funcName == "swap") {
      const auto size = args.size();
      assert(size >= 2);
      SmallVector<Value> targets(args.begin() + size - 2, args.end());
      for (auto v : targets)
        if (std::find(negations.begin(), negations.end(), v) != negations.end())
          reportNegateError();
      SmallVector<Value> ctrls(args.begin(), args.begin() + size - 2);
      auto negs =
          negatedControlsAttribute(builder.getContext(), ctrls, negations);
      auto swap = builder.create<quake::SwapOp>(loc, ctrls, targets);
      if (negs)
        swap->setAttr("negated_qubit_controls", negs);
      return true;
    }
    if (funcName == "p" || funcName == "r1")
      return buildOp<quake::R1Op, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         isControl);
    if (funcName == "cr1")
      return buildOp<quake::R1Op, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         /*control=*/true);
    if (funcName == "rx")
      return buildOp<quake::RxOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         isControl);
    if (funcName == "crx")
      return buildOp<quake::RxOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         /*control=*/true);
    if (funcName == "ry")
      return buildOp<quake::RyOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         isControl);
    if (funcName == "cry")
      return buildOp<quake::RyOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         /*control=*/true);
    if (funcName == "rz")
      return buildOp<quake::RzOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         isControl);
    if (funcName == "crz")
      return buildOp<quake::RzOp, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         /*control=*/true);

    if (funcName == "u3")
      return buildOp<quake::U3Op, Param>(builder, loc, args, negations,
                                         reportNegateError, isAdjoint,
                                         isControl, /*paramCount=*/3);

    // See if this is a custom unitary.
    std::string maybeUnitaryGenerator = funcName.str() + "_generator_";
    // Extract number of targets
    std::size_t targetCount = 0;
    std::string genFuncName;
    for (auto name : customOperationNames) {
      if (name.first.find(maybeUnitaryGenerator) != std::string::npos) {
        targetCount =
            std::stoi(name.first.substr(maybeUnitaryGenerator.length()));
        genFuncName = getCudaqKernelName(name.second);
        break;
      }
    }
    if (targetCount > 0) {
      // Extract number of parameters
      std::size_t paramCount = [&]() {
        std::size_t count = 0;
        for (auto arg : args) {
          auto argTy = arg.getType();
          if (isa<FloatType>(argTy)) {
            count++;
          } else if (auto ptrTy = dyn_cast<cc::PointerType>(argTy)) {
            emitFatalError(
                loc,
                "passing parameters by reference or pointer not supported");
          }
        }
        return count;
      }();
      auto srefAttr = SymbolRefAttr::get(
          StringAttr::get(builder.getContext(), genFuncName));
      ValueRange operands(args);
      assert(operands.size() >= 1 && "must be at least 1 operand");
      if ((operands.size() == 1) &&
          isa<quake::VeqType>(operands[0].getType())) {
        auto target = operands[0];
        if (!negations.empty())
          reportNegateError();
        Type i64Ty = builder.getI64Type();
        auto size = builder.create<quake::VeqSizeOp>(
            loc, builder.getIntegerType(64), target);
        Value rank = builder.create<cudaq::cc::CastOp>(
            loc, i64Ty, size, cudaq::cc::CastOpMode::Unsigned);
        auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &,
                               Block &block) {
          Value ref = builder.create<quake::ExtractRefOp>(loc, target,
                                                          block.getArgument(0));
          builder.create<quake::CustomUnitarySymbolOp>(loc, srefAttr,
                                                       ValueRange(), ref);
        };
        cudaq::opt::factory::createInvariantLoop(builder, loc, rank,
                                                 bodyBuilder);
      } else {
        auto [targets, ctrls] =
            maybeUnpackOperands(builder, loc, operands.drop_front(paramCount),
                                isControl, targetCount);
        for (auto v : targets)
          if (std::find(negations.begin(), negations.end(), v) !=
              negations.end())
            reportNegateError();
        auto negs =
            negatedControlsAttribute(builder.getContext(), ctrls, negations);
        SmallVector<Value> params;
        for (auto p : operands.take_front(paramCount))
          if (isa<cudaq::cc::PointerType>(p.getType()))
            params.push_back(builder.create<cudaq::cc::LoadOp>(loc, p));
        builder.create<quake::CustomUnitarySymbolOp>(
            loc, srefAttr, isAdjoint, params, ctrls, targets, negs);
      }
      return true;
    }

    if (funcName == "control") {
      // Expect the first argument to be an instance of a Callable. Need to
      // construct the name of the operator() call to make here.
      Value calleeValue = args[0];
      Value ctrlValues = args[1];
      SymbolRefAttr calleeSymbol;
      auto *ctx = builder.getContext();

      // Expand the negations inline around the quake.apply. This will result in
      // less duplication of code than threading the negated sense of the
      // control recursively through the callable.
      auto inlinedStartControlNegations = [&]() {
        if (!negations.empty()) {
          // Loop over the ctrlValues and negate (apply an XOp) those in the
          // negations list.
          if (auto concat = ctrlValues.getDefiningOp<quake::ConcatOp>()) {
            for (auto v : concat.getQbits())
              if (std::find(negations.begin(), negations.end(), v) !=
                  negations.end()) {
                if (isa<quake::VeqType>(v.getType())) {
                  reportClangError(
                      x, mangler, "cannot negate an entire register of qubits");
                } else {
                  SmallVector<Value> dummy;
                  buildOp<quake::XOp>(builder, loc, v, dummy, []() {});
                }
              }
          } else if (isa<quake::VeqType>(ctrlValues.getType())) {
            assert(negations.size() == 1 && negations[0] == ctrlValues);
            reportClangError(x, mangler,
                             "cannot negate an entire register of qubits");
          } else {
            assert(isa<quake::RefType>(ctrlValues.getType()));
            assert(negations.size() == 1 && negations[0] == ctrlValues);
            SmallVector<Value> dummy;
            buildOp<quake::XOp>(builder, loc, ctrlValues, dummy, []() {});
          }
        }
      };

      // Finish (uncompute) the inlined control negations. Generates the same
      // code pattern as the starting negations. Specifically, we invoke an XOp
      // on each negated control.
      auto inlinedFinishControlNegations = [&]() {
        inlinedStartControlNegations();
        negations.clear();
        return true;
      };
      auto callableObjectStructType = [&](Value v) {
        Type ty = v.getType();
        if (auto ptrTy = dyn_cast<cc::PointerType>(ty))
          ty = ptrTy.getElementType();
        return dyn_cast<cc::StructType>(ty);
      };

      if (auto ty = callableObjectStructType(calleeValue)) {
        auto *classDecl = classDeclFromTemplateArgument(*func, 0, *astContext);
        if (!classDecl) {
          // This shouldn't happen if the cudaq headers are used, but add a
          // check here just in case.
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "expected cudaq::control to be a specific template");
          de.Report(x->getBeginLoc(), id);
          return false;
        }
        auto *kernelCallOper = findCallOperator(classDecl);
        if (!kernelCallOper) {
          // This should be caught by the concepts used in the header file, but
          // add a check here just in case.
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "first argument to cudaq::control must be a callable");
          de.Report(x->getBeginLoc(), id);
          return false;
        }
        auto calleeName = generateCudaqKernelName(kernelCallOper);
        calleeSymbol = SymbolRefAttr::get(ctx, calleeName);
        auto kernelFunc = module.lookupSymbol<func::FuncOp>(calleeName);
        assert(kernelFunc && "kernel call operator must be present");
        auto kernelTy = kernelFunc.getFunctionType();
        auto kernelArgs =
            convertKernelArgs(loc, 2, args, kernelTy.getInputs(), x);
        inlinedStartControlNegations();
        builder.create<quake::ApplyOp>(loc, TypeRange{}, calleeSymbol,
                                       /*isAdjoint=*/false, ctrlValues,
                                       kernelArgs);
        return inlinedFinishControlNegations();
      }
      if (auto func = calleeValue.getDefiningOp<func::ConstantOp>()) {
        auto funcTy = cast<FunctionType>(func.getType());
        auto callableSym = func.getValueAttr();
        inlinedStartControlNegations();
        auto kernelArgs =
            convertKernelArgs(loc, 2, args, funcTy.getInputs(), x);
        builder.create<quake::ApplyOp>(loc, funcTy.getResults(), callableSym,
                                       /*isAdjoint=*/false, ctrlValues,
                                       kernelArgs);
        return inlinedFinishControlNegations();
      }
      if (auto ty = dyn_cast<cc::CallableType>(calleeValue.getType())) {
        // In order to autogenerate the control form of the called kernel, we
        // have to be able to determine precisely which kernel is being called
        // at this point. If this is a local lambda expression, it is handled
        // elsewhere. If this is a lambda expression argument, then we have to
        // recover it or give a compilation error.
        auto *tyPtr = x->getArg(0)->getType().getTypePtr();
        auto *recTy = dyn_cast<clang::RecordType>(tyPtr);
        if (!recTy && isa<clang::AutoType>(tyPtr)) {
          recTy = dyn_cast_or_null<clang::RecordType>(
              cast<clang::AutoType>(tyPtr)->desugar().getTypePtr());
        }
        if (!recTy && isa<clang::SubstTemplateTypeParmType>(tyPtr)) {
          auto *ty = cast<clang::SubstTemplateTypeParmType>(tyPtr);
          recTy = dyn_cast_or_null<clang::RecordType>(
              ty->getReplacementType().getTypePtr());
        }
        if (!recTy) {
          TODO_loc(loc,
                   "control does not appear to be on a user-defined kernel");
        }
        auto *decl = recTy->getDecl();
        if (decl->isLambda()) {
          auto *lambdaClass = cast<clang::CXXRecordDecl>(decl);
          auto *callOperDecl = findCallOperator(lambdaClass);
          if (isKernelEntryPoint(callOperDecl)) {
            // This callable is a kernel that will be lowered. Substitute the
            // name of the kernel.
            auto mangledName = generateCudaqKernelName(callOperDecl);
            calleeSymbol = SymbolRefAttr::get(ctx, mangledName);
          }
          auto funcTy = ty.getSignature();
          inlinedStartControlNegations();
          auto kernelArgs =
              convertKernelArgs(loc, 2, args, funcTy.getInputs(), x);
          if (isKernelEntryPoint(callOperDecl)) {
            builder.create<quake::ApplyOp>(
                loc, funcTy.getResults(), calleeSymbol,
                /*isAdjoint=*/false, ctrlValues, kernelArgs);
          } else {
            builder.create<quake::ApplyOp>(
                loc, funcTy.getResults(), calleeValue,
                /*isAdjoint=*/false, ctrlValues, kernelArgs);
          }
          return inlinedFinishControlNegations();
        }
        TODO_loc(loc, "value has !cc.lambda type but decl isn't a lambda");
      }
      TODO_loc(loc, "unexpected callable argument");
    }

    if (funcName == "adjoint") {
      // Expect the following declaration from qubit_qis.h:
      //
      // template <typename QuantumKernel, typename... Args>
      //   requires isCallableVoidKernel<QuantumKernel, Args...>
      // void adjoint(QuantumKernel &&kernel, Args &&...args);
      //
      // The first argument must be an instance of a Callable and a quantum
      // kernel. Traverse the AST here to construct the name of the operator()
      // to be called.
      auto kernelValue = args[0];
      SymbolRefAttr kernelSymbol;
      auto kernelTy = kernelValue.getType();
      if (auto ptrTy = dyn_cast<cc::PointerType>(kernelTy))
        kernelTy = ptrTy.getElementType();
      if (auto ty = dyn_cast<cc::StructType>(kernelTy)) {
        auto *ctx = builder.getContext();
        auto *classDecl = classDeclFromTemplateArgument(*func, 0, *astContext);
        if (!classDecl) {
          // This shouldn't happen if the cudaq headers are used, but add a
          // check here just in case.
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "expected cudaq::adjoint to be a specific template");
          de.Report(x->getBeginLoc(), id);
          return {};
        }
        auto *kernelCallOper = findCallOperator(classDecl);
        if (!kernelCallOper) {
          // This should be caught by the concepts used in the header file, but
          // add a check here just in case.
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "first argument to cudaq::adjoint must be a callable");
          de.Report(x->getBeginLoc(), id);
          return {};
        }
        auto kernelName = generateCudaqKernelName(kernelCallOper);
        kernelSymbol = SymbolRefAttr::get(ctx, kernelName);
        auto kernFunc = module.lookupSymbol<func::FuncOp>(kernelName);
        assert(kernFunc && "kernel call operator must be present");
        auto kernTy = kernFunc.getFunctionType();
        auto kernArgs = convertKernelArgs(loc, 1, args, kernTy.getInputs(), x);
        return builder.create<quake::ApplyOp>(loc, TypeRange{}, kernelSymbol,
                                              /*isAdjoint=*/true, ValueRange{},
                                              kernArgs);
      }
      if (auto func = kernelValue.getDefiningOp<func::ConstantOp>()) {
        auto kernSym = func.getValueAttr();
        auto funcTy = cast<FunctionType>(func.getType());
        auto kernArgs = convertKernelArgs(loc, 1, args, funcTy.getInputs(), x);
        return builder.create<quake::ApplyOp>(loc, funcTy.getResults(), kernSym,
                                              /*isAdjoint=*/true, ValueRange{},
                                              kernArgs);
      }
      if (auto ty = dyn_cast<cc::CallableType>(kernelTy)) {
        // In order to autogenerate the control form of the called kernel, we
        // have to be able to determine precisely which kernel is being called
        // at this point. If this is a local lambda expression, it is handled
        // elsewhere. If this is a lambda expression argument, then we have to
        // recover it or give a compilation error.
        auto *tyPtr = x->getArg(0)->getType().getTypePtr();
        auto *recTy = dyn_cast<clang::RecordType>(tyPtr);
        if (!recTy && isa<clang::AutoType>(tyPtr)) {
          recTy = dyn_cast_or_null<clang::RecordType>(
              cast<clang::AutoType>(tyPtr)->desugar().getTypePtr());
        }
        if (!recTy && isa<clang::SubstTemplateTypeParmType>(tyPtr)) {
          auto *ty = cast<clang::SubstTemplateTypeParmType>(tyPtr);
          recTy = dyn_cast_or_null<clang::RecordType>(
              ty->getReplacementType().getTypePtr());
        }
        if (!recTy) {
          TODO_loc(loc,
                   "adjoint does not appear to be on a user-defined kernel");
        }
        auto *decl = recTy->getDecl();
        if (decl->isLambda()) {
          auto *lambdaClass = cast<clang::CXXRecordDecl>(decl);
          auto *callOperDecl = findCallOperator(lambdaClass);
          if (isKernelEntryPoint(callOperDecl)) {
            auto mangledName = generateCudaqKernelName(callOperDecl);
            kernelSymbol =
                SymbolRefAttr::get(builder.getContext(), mangledName);
          }
          auto funcTy = ty.getSignature();
          auto kernelArgs =
              convertKernelArgs(loc, 1, args, funcTy.getInputs(), x);
          if (isKernelEntryPoint(callOperDecl)) {
            return builder.create<quake::ApplyOp>(
                loc, funcTy.getResults(), kernelSymbol,
                /*isAdjoint=*/true, ValueRange{}, kernelArgs);
          }
          return builder.create<quake::ApplyOp>(
              loc, funcTy.getResults(), kernelValue,
              /*isAdjoint=*/true, ValueRange{}, kernelArgs);
        }
        TODO_loc(loc, "value has !cc.lambda type but decl isn't a lambda");
      }
      TODO_loc(loc, "adjoint does not appear to be on a user-defined kernel");
    }

    if (funcName == "compute_action") {
      builder.create<quake::ComputeActionOp>(loc, /*is_dagger=*/false, args[0],
                                             args[1]);
      return true;
    }
    if (funcName == "compute_dag_action") {
      builder.create<quake::ComputeActionOp>(loc, /*is_dagger=*/true, args[0],
                                             args[1]);
      return true;
    }

    if (funcName == "toInteger" || funcName == "to_integer") {
      IRBuilder irBuilder(builder.getContext());
      if (failed(irBuilder.loadIntrinsic(module, cudaqConvertToInteger))) {
        reportClangError(x, mangler, "cannot load cudaqConvertToInteger");
        return false;
      }
      auto i64Ty = builder.getI64Type();
      return pushValue(
          builder.create<func::CallOp>(loc, i64Ty, cudaqConvertToInteger, args)
              .getResult(0));
    }

    if (funcName == "slice_vector") {
      auto svecTy = dyn_cast<cc::SpanLikeType>(args[0].getType());
      auto eleTy = svecTy.getElementType();
      assert(svecTy && "first argument must be std::vector");
      Value offset = args[1];
      Type ptrTy;
      Value vecPtr;
      if (eleTy == builder.getI1Type()) {
        eleTy = cc::ArrayType::get(builder.getI8Type());
        ptrTy = cc::PointerType::get(eleTy);
        vecPtr = builder.create<cc::StdvecDataOp>(loc, ptrTy, args[0]);
        auto bits = svecTy.getElementType().getIntOrFloatBitWidth();
        assert(bits > 0);
        auto scale = builder.create<arith::ConstantIntOp>(loc, (bits + 7) / 8,
                                                          args[1].getType());
        offset = builder.create<arith::MulIOp>(loc, scale, args[1]);
      } else {
        ptrTy = cc::PointerType::get(eleTy);
        auto arrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
        vecPtr = builder.create<cc::StdvecDataOp>(loc, arrTy, args[0]);
      }
      auto ptr = builder.create<cc::ComputePtrOp>(loc, ptrTy, vecPtr,
                                                  ArrayRef<Value>{offset});
      return pushValue(
          builder.create<cc::StdvecInitOp>(loc, svecTy, ptr, args[2]));
    }

    if (funcName == "range") {
      IRBuilder irBuilder(builder.getContext());
      auto i64Ty = builder.getI64Type(); // element type
      if (funcArity == 1) {
        [[maybe_unused]] auto result =
            irBuilder.loadIntrinsic(module, setCudaqRangeVector);
        assert(succeeded(result) && "loading intrinsic should never fail");
        auto upVal = args[0];
        auto upper = builder.create<cc::CastOp>(loc, i64Ty, upVal,
                                                cc::CastOpMode::Unsigned);
        auto buffer = builder.create<cc::AllocaOp>(loc, i64Ty, upper);
        auto stdvecTy = cc::StdvecType::get(i64Ty);
        auto call = builder.create<func::CallOp>(
            loc, stdvecTy, setCudaqRangeVector, ValueRange{buffer, upper});
        return pushValue(call.getResult(0));
      }
      assert(funcArity == 3);
      [[maybe_unused]] auto result =
          irBuilder.loadIntrinsic(module, setCudaqRangeVectorTriple);
      assert(succeeded(result) && "loading intrinsic should never fail");
      Value start = builder.create<cc::CastOp>(loc, i64Ty, args[0],
                                               cc::CastOpMode::Signed);
      Value stop = builder.create<cc::CastOp>(loc, i64Ty, args[1],
                                              cc::CastOpMode::Signed);
      Value step = builder.create<cc::CastOp>(loc, i64Ty, args[2],
                                              cc::CastOpMode::Signed);
      auto lengthCall = builder.create<func::CallOp>(
          loc, i64Ty, getCudaqSizeFromTriple, ValueRange{start, stop, step});
      Value length = lengthCall.getResult(0);
      auto buffer = builder.create<cc::AllocaOp>(loc, i64Ty, length);
      auto stdvecTy = cc::StdvecType::get(i64Ty);
      auto call =
          builder.create<func::CallOp>(loc, stdvecTy, setCudaqRangeVectorTriple,
                                       ValueRange{buffer, start, stop, step});
      return pushValue(call.getResult(0));
    }

    if (funcName == "device_call") {
      // Generate the high-level DeviceCallOp.
      Value deviceId;
      auto funcConst = args[0].getDefiningOp<func::ConstantOp>();
      std::size_t argsOffset = 1;
      if (!funcConst) {
        deviceId = args[0];
        funcConst = args[1].getDefiningOp<func::ConstantOp>();
        ++argsOffset;
      }
      auto symbol = funcConst.getValue();
      auto devFunc = module.lookupSymbol<func::FuncOp>(symbol);
      devFunc->setAttr(cudaq::deviceCallAttrName, builder.getUnitAttr());
      auto devFuncTy = cast<FunctionType>(funcConst.getType());

      auto maybeGPULaunchParams =
          [&]() -> std::optional<std::pair<std::size_t, std::size_t>> {
        if (!functionDecl->isTemplateInstantiation())
          return std::nullopt;

        auto *tsi = functionDecl->getTemplateSpecializationInfo();
        if (!tsi)
          return std::nullopt;

        const auto *tArgs = tsi->TemplateArguments;
        // Should have 4 TemplateParams here: 2 for the block and thread/block
        // and 2 for the function and variadic pack types.
        if (tArgs->size() != 4)
          return std::nullopt;

        const clang::TemplateArgument &numBlocksArg = tArgs->get(0);
        if (numBlocksArg.getKind() != clang::TemplateArgument::Integral)
          return std::nullopt;
        std::size_t numBlocks = numBlocksArg.getAsIntegral().getLimitedValue();

        // Extract numThreads (second argument)
        const clang::TemplateArgument &numThreadsArg = tArgs->get(1);
        if (numThreadsArg.getKind() != clang::TemplateArgument::Integral)
          return std::nullopt;
        std::size_t numThreads =
            numThreadsArg.getAsIntegral().getLimitedValue();

        return std::make_pair(numBlocks, numThreads);
      }();

      SmallVector<Value> processedArgs;
      for (std::size_t i = 0; i < args.size(); i++) {
        if (i < argsOffset || !cc::isDevicePtr(args[i].getType())) {
          processedArgs.push_back(args[i]);
          continue;
        }
        // Resolve the raw pointer from this device_ptr handle.
        Value result = builder.create<cc::ResolveDevicePtrOp>(
            loc, devFuncTy.getInputs()[i - argsOffset], args[i]);
        processedArgs.push_back(result);
      }

      auto callArgs = convertKernelArgs(loc, argsOffset, processedArgs,
                                        devFuncTy.getInputs(), x);

      auto devCall = [&]() {
        if (maybeGPULaunchParams) {
          auto [numBlocks, numThreads] = maybeGPULaunchParams.value();
          Value blocks =
              builder.create<arith::ConstantIntOp>(loc, numBlocks, 64);
          Value threadsPerBlock =
              builder.create<arith::ConstantIntOp>(loc, numThreads, 64);
          return builder.create<cc::DeviceCallOp>(
              loc, devFuncTy.getResults(), symbol, ValueRange{blocks},
              ValueRange{threadsPerBlock}, deviceId, callArgs);
        }
        return builder.create<cc::DeviceCallOp>(loc, devFuncTy.getResults(),
                                                symbol, deviceId, callArgs);
      }();
      if (devFuncTy.getResults().empty())
        return true;
      return pushValue(devCall.getResult(0));
    }

    // Finally, flag the call as an error except anything in cudaq::solvers or
    // cudaq::qec.
    if (!isInNamespace(func, "solvers") && !isInNamespace(func, "qec")) {
      TODO_loc(loc, "unknown function, " + funcName + ", in cudaq namespace");
    }
  } // end in cudaq namespace

  if (isInNamespace(func, "std")) {
    if (funcName == "reverse") {
      // For a `std::vector<T>`, the arguments will be pointers into the data
      // buffer. Create a loop that interchanges pairs as $(a_0, a_1-1)$,
      // $(a_0+1, a_1-2)$, ... until $a_0 + n \ge a_1 - n - 1$.
      auto i64Ty = builder.getI64Type();
      auto hiInt = builder.create<cc::CastOp>(loc, i64Ty, args[1]);
      auto loInt = builder.create<cc::CastOp>(loc, i64Ty, args[0]);
      auto ptrTy = cast<cc::PointerType>(args[0].getType());
      auto eleTy = ptrTy.getElementType();
      auto arrTy = dyn_cast<cc::ArrayType>(eleTy);
      if (arrTy) {
        eleTy = arrTy.getElementType();
        ptrTy = cc::PointerType::get(eleTy);
      } else {
        arrTy = cc::ArrayType::get(eleTy);
      }
      auto eleSize = eleTy.getIntOrFloatBitWidth();
      auto adjust = getConstantInt(builder, loc, eleSize / 4, i64Ty);
      auto dist = builder.create<arith::SubIOp>(loc, hiInt, loInt);
      Value iters = builder.create<arith::DivSIOp>(loc, dist, adjust);
      auto ptrArrTy = cc::PointerType::get(arrTy);
      Value basePtr = builder.create<cc::CastOp>(loc, ptrArrTy, args[0]);
      auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &,
                             Block &block) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&block);
        auto iterIdx = block.getArgument(0);
        auto ptrA =
            builder.create<cc::ComputePtrOp>(loc, ptrTy, basePtr, iterIdx);
        auto one = builder.create<arith::ConstantIntOp>(loc, 1, i64Ty);
        auto iters1 = builder.create<arith::SubIOp>(loc, iters, one);
        Value hiIdx = builder.create<arith::SubIOp>(loc, iters1, iterIdx);
        auto ptrB =
            builder.create<cc::ComputePtrOp>(loc, ptrTy, basePtr, hiIdx);
        Value loadA = builder.create<cc::LoadOp>(loc, ptrA);
        Value loadB = builder.create<cc::LoadOp>(loc, ptrB);
        builder.create<cc::StoreOp>(loc, loadA, ptrB);
        builder.create<cc::StoreOp>(loc, loadB, ptrA);
      };
      auto idxTy = builder.getI64Type();
      auto idxIters = builder.create<cudaq::cc::CastOp>(
          loc, idxTy, iters, cudaq::cc::CastOpMode::Unsigned);
      opt::factory::createInvariantLoop(builder, loc, idxIters, bodyBuilder);
      return true;
    }
    if (funcName == "get") {
      auto *stdGetSpec = cast<clang::FunctionDecl>(callee);
      auto &specArgs = *stdGetSpec->getTemplateSpecializationArgs();
      auto resultTy = cc::PointerType::get(peekType());
      auto fixIfTuple = [&](std::int32_t &offset) {
        if (tuplesAreReversed) {
          auto *rd = x->getArg(0)->getType().getTypePtr()->getAsRecordDecl();
          if (rd->getName() == "tuple") {
            auto ptrTy = cast<cc::PointerType>(args[0].getType());
            auto strTy = cast<cc::StructType>(ptrTy.getElementType());
            offset = strTy.getMembers().size() - offset - 1;
          }
        }
      };
      // The first specialization arg is either a type or an integer value.
      if (specArgs[0].getKind() == clang::TemplateArgument::ArgKind::Integral) {
        std::int32_t offset = specArgs[0].getAsIntegral().getExtValue();
        fixIfTuple(offset);
        auto ptr = builder.create<cc::ComputePtrOp>(
            loc, resultTy, args[0], ArrayRef<cc::ComputePtrArg>{offset});
        return pushValue(builder.create<cc::LoadOp>(loc, ptr));
      }
      auto *selectTy = specArgs[0].getAsType().getTypePtr();
      assert(specArgs[1].getKind() == clang::TemplateArgument::ArgKind::Pack);
      std::int32_t offset = 0;
      for (auto &templateArg : specArgs[1].pack_elements()) {
        if (templateArg.getAsType().getTypePtr() == selectTy) {
          fixIfTuple(offset);
          auto ptr = builder.create<cc::ComputePtrOp>(
              loc, resultTy, args[0], ArrayRef<cc::ComputePtrArg>{offset});
          return pushValue(builder.create<cc::LoadOp>(loc, ptr));
        }
        ++offset;
      }
    }
  }

  if (func->isVariadic()) {
    reportClangError(x, mangler,
                     "cannot call variadic function from quantum kernel");
    return false;
  }

  // If we get here, and the CallExpr takes qubits or qreg and it must be
  // another kernel call.
  auto mlirFuncTy = cast<FunctionType>(calleeOp.getType());
  auto funcResults = mlirFuncTy.getResults();
  auto convertedArgs =
      convertKernelArgs(loc, 0, args, mlirFuncTy.getInputs(), x);
  auto call = builder.create<func::CallIndirectOp>(loc, funcResults, calleeOp,
                                                   convertedArgs);
  if (call.getNumResults() > 0) {
    if (call.getNumResults() != 1) {
      reportClangError(x, mangler, "expect exactly one return value");
      return false;
    }
    if (auto vecTy =
            dyn_cast<cudaq::cc::StdvecType>(call.getResult(0).getType())) {
      auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
      if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_vectorCopyToStack")))
        module.emitError("failed to load intrinsic");
      auto eleTy = [&]() -> Type {
        auto et = vecTy.getElementType();
        if (et == builder.getI1Type())
          return builder.getI8Type();
        return et;
      }();
      auto data = builder.create<cudaq::cc::StdvecDataOp>(
          loc, cudaq::cc::PointerType::get(eleTy), call.getResult(0));
      auto i64Ty = builder.getI64Type();
      auto len = builder.create<cudaq::cc::StdvecSizeOp>(loc, i64Ty,
                                                         call.getResult(0));
      auto eleSize = builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, eleTy);
      auto size = builder.create<arith::MulIOp>(loc, len, eleSize);
      auto buffer = builder.create<cudaq::cc::AllocaOp>(loc, eleTy, size);
      auto i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());
      auto cbuffer = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, buffer);
      auto cdata = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, data);
      builder.create<func::CallOp>(loc, TypeRange{},
                                   "__nvqpp_vectorCopyToStack",
                                   ValueRange{cbuffer, cdata, size});
      Value newSpan =
          builder.create<cudaq::cc::StdvecInitOp>(loc, vecTy, buffer, len);
      return pushValue(newSpan);
    }
    return pushValue(call.getResult(0));
  }
  return true;
}

std::optional<std::string> QuakeBridgeVisitor::isInterceptedSubscriptOperator(
    clang::CXXOperatorCallExpr *x) {
  if (isSubscriptOperator(x->getOperator())) {
    if (auto decl = dyn_cast<clang::CXXMethodDecl>(x->getCalleeDecl())) {
      auto typeName = decl->getParent()->getNameAsString();
      if (isInNamespace(decl, "cudaq")) {
        if (isCudaQType(typeName))
          return {typeName};
      } else if (isInNamespace(decl, "std")) {
        if (typeName == "vector")
          return {typeName};
      } else if (isInNamespace(decl, "std")) {
        if (typeName == "_Bit_reference" || typeName == "__bit_reference" ||
            typeName == "__bit_const_reference")
          return {typeName};
      }
    }
  }
  return {};
}

bool QuakeBridgeVisitor::WalkUpFromCXXOperatorCallExpr(
    clang::CXXOperatorCallExpr *x) {
  // Is this an operator[] that we're converting?
  if (isInterceptedSubscriptOperator(x)) {
    // Yes, so skip walking the superclass, CallExpr.
    return VisitCXXOperatorCallExpr(x);
  }
  if (auto *func = dyn_cast_or_null<clang::FunctionDecl>(x->getCalleeDecl())) {
    if (isCallOperator(x) || (isInClassInNamespace(func, "qudit", "cudaq") &&
                              isExclaimOperator(x->getOperator())))
      return VisitCXXOperatorCallExpr(x);
  }

  // Otherwise, handle with default traversal.
  return WalkUpFromCallExpr(x) && VisitCXXOperatorCallExpr(x);
}

bool QuakeBridgeVisitor::hasTOSEntryKernel() {
  if (auto fn = peekValue().getDefiningOp<func::ConstantOp>()) {
    auto name = fn.getValue().str();
    for (auto fdPair : functionsToEmit)
      if (getCudaqKernelName(fdPair.first) == name)
        return true;
  }
  return false;
}

bool QuakeBridgeVisitor::VisitCXXOperatorCallExpr(
    clang::CXXOperatorCallExpr *x) {
  auto loc = toLocation(x->getSourceRange());

  // Helper to replace the operator[] function name with the value, v.
  auto replaceTOSValue = [&](Value v) {
    [[maybe_unused]] auto funcVal = popValue();
    assert(funcVal.getDefiningOp<func::ConstantOp>());
    return pushValue(v);
  };
  if (auto typeNameOpt = isInterceptedSubscriptOperator(x)) {
    auto &typeName = *typeNameOpt;
    if (isCudaQType(typeName)) {
      auto idx_var = popValue();
      auto qreg_var = popValue();
      auto *arg0 = x->getArg(0);
      if (isa<clang::MemberExpr>(arg0)) {
        // This is a subscript operator on a data member and the type is a
        // quantum type (likely a `qview`). This can only happen in a quantum
        // `struct`, which the spec says must be one-level deep at most and must
        // only contain references to qubits explicitly allocated in other
        // variables. `qreg_var` will be a `quake.get_member`. Do not add this
        // extract `Op` to the symbol table, but always generate a new
        // `quake.extract_ref` `Op` to get the exact qubit (reference) value.
        auto address_qubit =
            builder.create<quake::ExtractRefOp>(loc, qreg_var, idx_var);
        return replaceTOSValue(address_qubit);
      }
      // Get name of the qreg, e.g. qr, and use it to construct a name for the
      // element, which is intended to be qr%n when n is the index of the
      // accessed qubit.
      if (!isa<clang::DeclRefExpr>(arg0))
        reportClangError(x, mangler,
                         "internal error: expected a variable name");
      StringRef qregName = getNamedDecl(arg0)->getName();
      auto name = createQubitSymbolTableName(qregName, idx_var, allocator);

      // If the name exists in the symbol table, return its stored value.
      if (symbolTable.count(name))
        return replaceTOSValue(symbolTable.lookup(name));

      // Otherwise create an operation to access the qubit, store that value
      // in the symbol table, and return the AddressQubit operation's
      // resulting value.
      auto address_qubit =
          builder.create<quake::ExtractRefOp>(loc, qreg_var, idx_var);

      // NB: varName is built from the variable name *and* the index value. This
      // front-end optimization is likely unnecessary as the compiler can always
      // canonicalize and merge identical quake.extract_ref operations.
      symbolTable.insert(name, address_qubit);
      return replaceTOSValue(address_qubit);
    }
    if (typeName == "vector") {
      // Here we have something like vector<float> theta, and in the kernel, we
      // are accessing it like theta[i].
      auto indexVar = popValue();
      auto svec = popValue();
      if (isa<cc::PointerType>(svec.getType()))
        svec = builder.create<cc::LoadOp>(loc, svec);
      if (!isa<cc::StdvecType>(svec.getType())) {
        TODO_x(loc, x, mangler, "vector dereference");
        return false;
      }
      auto eleTy = cast<cc::StdvecType>(svec.getType()).getElementType();
      if (eleTy == builder.getI1Type())
        eleTy = builder.getI8Type();
      auto elePtrTy = cc::PointerType::get(eleTy);
      auto eleArrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
      auto vecPtr = builder.create<cc::StdvecDataOp>(loc, eleArrTy, svec);
      auto eleAddr = builder.create<cc::ComputePtrOp>(loc, elePtrTy, vecPtr,
                                                      ValueRange{indexVar});
      return replaceTOSValue(eleAddr);
    }
    if (typeName == "_Bit_reference" || typeName == "__bit_reference" ||
        typeName == "__bit_const_reference") {
      // For vector<bool>, on the kernel side this is represented as a sequence
      // of byte-sized boolean values (true and false). On the host side, C++ is
      // likely going to pack the booleans as bits in words.
      auto indexVar = popValue();
      auto svec = popValue();
      assert(isa<cc::StdvecType>(svec.getType()));
      auto i8Ty = builder.getI8Type();
      auto elePtrTy = cc::PointerType::get(i8Ty);
      auto eleArrTy = cc::PointerType::get(cc::ArrayType::get(i8Ty));
      auto vecPtr = builder.create<cc::StdvecDataOp>(loc, eleArrTy, svec);
      auto eleAddr = builder.create<cc::ComputePtrOp>(loc, elePtrTy, vecPtr,
                                                      ValueRange{indexVar});
      auto i1PtrTy = cc::PointerType::get(builder.getI1Type());
      auto i1Cast = builder.create<cudaq::cc::CastOp>(loc, i1PtrTy, eleAddr);
      return replaceTOSValue(i1Cast);
    }
    TODO_loc(loc, "unhandled operator call for quake conversion");
  }

  if (auto *func = dyn_cast_or_null<clang::FunctionDecl>(x->getCalleeDecl())) {
    // Lower <any>::operator()(...)
    if (isCallOperator(x)) {
      auto funcArity = func->getNumParams();
      SmallVector<Value> args = lastValues(funcArity);
      auto tos = popValue();
      auto tosTy = tos.getType();
      auto ptrTy = dyn_cast<cc::PointerType>(tosTy);
      bool isEntryKernel = hasTOSEntryKernel();
      if ((ptrTy && isa<cc::StructType>(ptrTy.getElementType())) ||
          isEntryKernel) {
        // The call operator has an object in the call position, so we want to
        // replace it with an indirect call to the func::ConstantOp.
        auto indirect = popValue();
        auto funcTy = cast<FunctionType>(indirect.getType());
        auto convertedArgs =
            convertKernelArgs(loc, 0, args, funcTy.getInputs(), x);
        auto call = builder.create<func::CallIndirectOp>(
            loc, funcTy.getResults(), indirect, convertedArgs);
        if (call.getResults().empty())
          return true;
        return pushValue(call.getResult(0));
      }
      auto indCallTy = [&]() -> cc::IndirectCallableType {
        if (ptrTy) {
          auto ty = dyn_cast<cc::IndirectCallableType>(ptrTy.getElementType());
          if (ty)
            return ty;
        }
        return dyn_cast<cc::IndirectCallableType>(tosTy);
      }();
      if (indCallTy) {
        [[maybe_unused]] auto discardedCallOp = popValue();
        auto funcTy = cast<FunctionType>(indCallTy.getSignature());
        auto call = builder.create<cc::CallIndirectCallableOp>(
            loc, funcTy.getResults(), tos, args);
        if (call.getResults().empty())
          return true;
        return pushValue(call.getResult(0));
      }
      auto callableTy = cast<cc::CallableType>(tosTy);
      auto callInd = builder.create<cc::CallCallableOp>(
          loc, callableTy.getSignature().getResults(), tos, args);
      if (callInd.getResults().empty()) {
        popValue();
        return true;
      }
      return replaceTOSValue(callInd.getResult(0));
    }

    // Lower cudaq::qudit<>::operator!()
    if (isInClassInNamespace(func, "qudit", "cudaq") &&
        isExclaimOperator(x->getOperator())) {
      auto qubit = popValue();
      negations.push_back(qubit);
      return replaceTOSValue(qubit);
    }
  }
  return true;
}

/// When traversing an expression such as `Kernel{}` or `Kernel()`, the object
/// may be passed to a function that needs a special callable object rather than
/// just the object. In order to make sure the call operator is already
/// declared, it is added here if needed to the module.
/// This method must only be called from a Traverse<Foo> method.
void QuakeBridgeVisitor::maybeAddCallOperationSignature(clang::Decl *x) {
  while (x) {
    if (auto *classDecl = dyn_cast<clang::CXXRecordDecl>(x)) {
      auto *callOperDecl = findCallOperator(classDecl);
      if (callOperDecl && isKernelEntryPoint(callOperDecl)) {
        auto loc = toLocation(callOperDecl);
        if (!TraverseType(callOperDecl->getType()))
          emitFatalError(loc, "expected type for call operator");
        auto kernelName = generateCudaqKernelName(callOperDecl);
        getOrAddFunc(loc, kernelName, peelPointerFromFunction(popType()));
      }
      return;
    }
    if (isa<clang::NamespaceDecl, clang::TranslationUnitDecl>(x))
      return;
    x = cast<clang::Decl>(x->getDeclContext()->getParent());
  }
}

bool QuakeBridgeVisitor::TraverseInitListExpr(clang::InitListExpr *x,
                                              DataRecursionQueue *) {
  if (x->isSyntacticForm()) {
    // The syntactic form is the surface level syntax as typed by the user. This
    // isn't really all that helpful during the lowering process. We want to
    // deal with the semantic form. See below.
    auto loc = toLocation(x);
    if (x->getNumInits() != 0)
      TODO_loc(loc, "initializer list containing elements");
    return true;
  }

  // If the initializer-list is possibly a Callable type, preemptively add the
  // signature of the call operation (`operator()`) to the Module if not
  // present.
  if (auto *ty = x->getType().getTypePtr())
    if (auto *tyDecl = ty->getAsRecordDecl())
      maybeAddCallOperationSignature(tyDecl);

  // Since an initializer-list can be empty (no objects), push the type on the
  // type stack. This will allow VisitInitListExpr, etc. to know what to do when
  // there are no values.
  [[maybe_unused]] auto typeStackDepth = typeStack.size();
  if (!TraverseType(x->getType()))
    return false;
  assert(typeStack.size() == typeStackDepth + 1 &&
         "expected a type for initializer-list");

  // Now visit the elements of the list, if any.
  for (auto *subStmt : x->children())
    if (!TraverseStmt(subStmt))
      return false;

  // And finish the post-order traversal.
  auto result = WalkUpFromInitListExpr(x);
  assert(typeStack.size() == typeStackDepth && "expected type to be consumed");
  return result;
}

bool QuakeBridgeVisitor::VisitInitListExpr(clang::InitListExpr *x) {
  auto loc = toLocation(x);
  std::int32_t size = x->getNumInits();
  auto initListTy = popType();
  if (size == 0) {
    // Nothing in the list. Just allocate the type.
    return pushValue(builder.create<cc::AllocaOp>(loc, initListTy));
  }

  // List has 1 or more members.
  if (size == 1 && isa<clang::MaterializeTemporaryExpr>(x->getInit(0)))
    if (auto alloc = peekValue().getDefiningOp<cudaq::cc::AllocaOp>())
      if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(initListTy))
        if (alloc.getElementType() == arrTy.getElementType())
          return true;
  auto last = lastValues(size);
  bool allRef = std::all_of(last.begin(), last.end(), [](auto v) {
    return isa<quake::RefType, quake::VeqType>(v.getType());
  });
  if (allRef && isa<quake::StruqType>(initListTy))
    return pushValue(builder.create<quake::MakeStruqOp>(loc, initListTy, last));

  if (allRef && !isa<cc::StructType>(initListTy)) {
    // Initializer list contains all quantum reference types. In this case we
    // want to create quake code to concatenate the references into a veq.
    if (size > 1) {
      auto veqTy = [&]() -> quake::VeqType {
        unsigned size = 0;
        for (auto v : last) {
          if (auto veqTy = dyn_cast<quake::VeqType>(v.getType())) {
            if (!veqTy.hasSpecifiedSize())
              return quake::VeqType::getUnsized(builder.getContext());
            size += veqTy.getSize();
          } else {
            ++size;
          }
        }
        return quake::VeqType::get(builder.getContext(), size);
      }();
      return pushValue(builder.create<quake::ConcatOp>(loc, veqTy, last));
    }
    // Pass initialization list with one member as a Ref.
    return pushValue(last[0]);
  }

  // These initializer expressions are not quantum references. In this case, we
  // allocate some memory for a variable and store the init list elements there.
  auto structTy = dyn_cast<cc::StructType>(initListTy);
  std::int32_t structMems = structTy ? structTy.getMembers().size() : 0;
  std::int32_t numEles = structMems ? size / structMems : size;
  // Generate the array size value.
  Value arrSize = builder.create<arith::ConstantIntOp>(loc, numEles, 64);

  // Allocate the required memory chunk.
  Type eleTy = [&]() {
    if (auto arrTy = dyn_cast<cc::ArrayType>(initListTy))
      return arrTy.getElementType();
    return initListTy;
  }();

  bool lastValuesAreConstant = [&]() {
    for (Value v : last)
      if (!opt::factory::maybeValueOfFloatConstant(v))
        return false;
    return true;
  }();

  if (initializerIsGlobal && lastValuesAreConstant) {
    static unsigned counter = 0;
    auto *ctx = builder.getContext();
    auto globalTy = cc::ArrayType::get(ctx, eleTy, size);
    SmallVector<Attribute> values;
    auto f64Ty = builder.getF64Type();
    for (Value v : last) {
      auto fp = opt::factory::maybeValueOfFloatConstant(v);
      values.push_back(FloatAttr::get(f64Ty, *fp));
    }
    // NB: Unfortunately, the LLVM-IR dialect doesn't lower DenseF64ArrayAttr to
    // LLVM IR without throwing errors.
    auto tensorTy = RankedTensorType::get(size, eleTy);
    auto f64Attr = DenseElementsAttr::get(tensorTy, values);
    // Create a unique name.
    std::string name = "__nvqpp__rodata_init_" + std::to_string(counter++);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(module.getBody());
      builder
          .create<cc::GlobalOp>(loc, globalTy, name, f64Attr,
                                /*constant=*/true, /*external=*/false)
          .setPrivate();
    }
    auto ptrTy = cc::PointerType::get(globalTy);
    auto globalInit = builder.create<cc::AddressOfOp>(loc, ptrTy, name);
    return pushValue(globalInit);
  }

  // If quantum, use value semantics with cc insert / extract value.
  if (isa<quake::StruqType>(eleTy))
    return pushValue(builder.create<quake::MakeStruqOp>(loc, eleTy, last));

  Value alloca = (numEles > 1)
                     ? builder.create<cc::AllocaOp>(loc, eleTy, arrSize)
                     : builder.create<cc::AllocaOp>(loc, eleTy);

  // Store the values in the allocated memory
  for (auto iter : llvm::enumerate(last)) {
    std::int32_t i = iter.index();
    auto v = iter.value();
    Value ptr;
    if (structMems) {
      if (numEles > 1) {
        auto ptrTy =
            cc::PointerType::get(structTy.getMembers()[i % structMems]);
        ptr = builder.create<cc::ComputePtrOp>(
            loc, ptrTy, alloca,
            ArrayRef<cc::ComputePtrArg>{i / structMems, i % structMems});
      } else {
        auto ptrTy = cc::PointerType::get(structTy.getMembers()[i]);
        ptr = builder.create<cc::ComputePtrOp>(loc, ptrTy, alloca,
                                               ArrayRef<cc::ComputePtrArg>{i});
      }
    } else {
      if (numEles > 1) {
        auto ptrTy = cc::PointerType::get(eleTy);
        ptr = builder.create<cc::ComputePtrOp>(loc, ptrTy, alloca,
                                               ArrayRef<cc::ComputePtrArg>{i});
      } else {
        auto arrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
        auto cast = builder.create<cc::CastOp>(loc, arrTy, alloca);
        auto ptrTy = cc::PointerType::get(eleTy);
        ptr = builder.create<cc::ComputePtrOp>(loc, ptrTy, cast,
                                               ArrayRef<cc::ComputePtrArg>{i});
      }
    }
    assert(ptr &&
           (v.getType() ==
            cast<cc::PointerType>(ptr.getType()).getElementType()) &&
           "value type must match pointer element type");
    builder.create<cc::StoreOp>(loc, v, ptr);
  }

  return pushValue(alloca);
}

bool QuakeBridgeVisitor::TraverseCXXConstructExpr(clang::CXXConstructExpr *x,
                                                  DataRecursionQueue *) {
  if (x->isElidable())
    return true;
  [[maybe_unused]] auto typeStackDepth = typeStack.size();
  bool saveInitializerIsGlobal = initializerIsGlobal;
  if (x->getConstructor()) {
    if (!TraverseType(x->getType()))
      return false;
    assert(typeStack.size() == typeStackDepth + 1);
    if (x->isStdInitListInitialization() && isa<quake::VeqType>(peekType()))
      initializerIsGlobal = true;
  }
  auto *ctor = x->getConstructor();
  // FIXME: this implicit code visit setting is a hack to only visit a default
  // argument value when constructing a complex value. We should always be able
  // to visit default arguments, but we currently trip over default allocators,
  // etc.
  bool saveVisitImplicitCode = visitImplicitCode;
  if (isInClassInNamespace(ctor, "complex", "std"))
    visitImplicitCode = true;
  auto result = Base::TraverseCXXConstructExpr(x);
  visitImplicitCode = saveVisitImplicitCode;
  initializerIsGlobal = saveInitializerIsGlobal;
  assert(typeStack.size() == typeStackDepth || raisedError);
  return result;
}

static bool isAllocatorQualType(const clang::QualType &ty) {
  if (auto recordType = dyn_cast<clang::RecordType>(ty.getTypePtr()))
    return recordType->getDecl()->getName() == "allocator";
  return false;
}

static bool isInitializerListQualType(const clang::QualType &ty) {
  if (auto recordType = dyn_cast<clang::RecordType>(ty.getTypePtr()))
    return recordType->getDecl()->getName() == "initializer_list";
  return false;
}

static Type getEleTyFromVectorCtor(Type ctorTy) {
  if (auto stdvecTy = dyn_cast<cc::StdvecType>(ctorTy))
    return stdvecTy.getElementType();
  return ctorTy;
}

bool QuakeBridgeVisitor::VisitCXXParenListInitExpr(
    clang::CXXParenListInitExpr *x) {
  auto ty = peekType();
  assert(ty && "type must be present");
  LLVM_DEBUG(llvm::dbgs() << "paren list type: " << ty << '\n');
  auto structTy = dyn_cast<quake::StruqType>(ty);
  if (!structTy)
    return true;
  auto loc = toLocation(x);
  auto last = lastValues(structTy.getMembers().size());
  return pushValue(builder.create<quake::MakeStruqOp>(loc, structTy, last));
}

bool QuakeBridgeVisitor::VisitCXXConstructExpr(clang::CXXConstructExpr *x) {
  auto loc = toLocation(x);
  auto *ctor = x->getConstructor();
  if (!ctor) {
    TODO_loc(loc, "C++ ctor (NULL)");
  }
  // The ctor type is the class for which the ctor is a member.
  auto ctorTy = popType();
  // FIXME: not every constructor has a name.
  std::string ctorName = ctor->getNameAsString();
  if (isInNamespace(ctor, "cudaq")) {
    if (x->getNumArgs() == 0) {
      if (ctorName == "qudit") {
        // This is a single qubit.
        assert(isa<quake::RefType>(ctorTy));
        return pushValue(builder.create<quake::AllocaOp>(loc));
      }
      // These classes have template arguments that may give a compile-time
      // constant size. qarray is the only one that requires it, however.
      if (ctorName == "qreg" || ctorName == "qarray" || ctorName == "qspan") {
        [[maybe_unused]] auto veqTy = cast<quake::VeqType>(ctorTy);
        assert(veqTy.hasSpecifiedSize());
        return pushValue(builder.create<quake::AllocaOp>(loc, ctorTy));
      }
      if (ctorName == "qvector") {
        // The default qvector ctor creates a veq of size 1.
        assert(isa<quake::VeqType>(ctorTy));
        auto veq1Ty = quake::VeqType::get(builder.getContext(), 1);
        return pushValue(builder.create<quake::AllocaOp>(loc, veq1Ty));
      }
    } else if (x->getNumArgs() == 1) {
      if (ctorName == "qreg") {
        // This is a cudaq::qreg(std::size_t).
        auto sizeVal = popValue();
        assert(isa<IntegerType>(sizeVal.getType()));
        return pushValue(builder.create<quake::AllocaOp>(
            loc, quake::VeqType::getUnsized(builder.getContext()), sizeVal));
      }

      if (ctorName == "state") {
        // cudaq::state ctor can be materialized when using local simulators and
        // converting raw data to state vectors. Use a runtime helper function
        // to perform the conversion.
        Value stdvec = popValue();
        auto stateTy = cudaq::cc::PointerType::get(
            quake::StateType::get(builder.getContext()));
        if (auto stdvecTy = dyn_cast<cudaq::cc::StdvecType>(stdvec.getType())) {
          auto dataTy = cudaq::cc::PointerType::get(stdvecTy.getElementType());
          Value data =
              builder.create<cudaq::cc::StdvecDataOp>(loc, dataTy, stdvec);
          auto i64Ty = builder.getI64Type();
          Value size =
              builder.create<cudaq::cc::StdvecSizeOp>(loc, i64Ty, stdvec);
          return pushValue(builder.create<quake::CreateStateOp>(
              loc, stateTy, ValueRange{data, size}));
        }
        if (auto alloc = stdvec.getDefiningOp<cudaq::cc::AllocaOp>()) {
          Value size = alloc.getSeqSize();
          return pushValue(builder.create<quake::CreateStateOp>(
              loc, stateTy, ValueRange{alloc, size}));
        }
        TODO_loc(loc, "unhandled state constructor");
        return false;
      }

      // lambda determines: is `t` a cudaq::state* ?
      auto isStateType = [&](Type t) {
        if (auto ptrTy = dyn_cast<cc::PointerType>(t))
          return isa<quake::StateType>(ptrTy.getElementType());
        return false;
      };

      if (ctorName == "qudit") {
        auto initials = popValue();
        if (isa<quake::StateType>(initials.getType()))
          if (auto load = initials.getDefiningOp<cudaq::cc::LoadOp>())
            initials = load.getPtrvalue();
        if (isStateType(initials.getType())) {
          Value alloca = builder.create<quake::AllocaOp>(loc);
          auto veq1Ty = quake::VeqType::get(builder.getContext(), 1);
          Value initSt = builder.create<quake::InitializeStateOp>(
              loc, veq1Ty, ValueRange{alloca, initials});
          if (auto initOp = initials.getDefiningOp<quake::CreateStateOp>())
            builder.create<quake::DeleteStateOp>(loc, initOp);
          return pushValue(builder.create<quake::ExtractRefOp>(loc, initSt, 0));
        }
        bool ok = false;
        if (auto ptrTy = dyn_cast<cc::PointerType>(initials.getType()))
          if (auto arrTy = dyn_cast<cc::ArrayType>(ptrTy.getElementType()))
            ok = isa<ComplexType>(arrTy.getElementType());
        if (!ok) {
          // Invalid initializer ignored, but emit an error.
          reportClangError(x, mangler, "invalid qudit initial value");
          return pushValue(builder.create<quake::AllocaOp>(loc));
        }
        auto *ctx = builder.getContext();
        auto veqTy = quake::VeqType::get(ctx, 1);
        auto alloc = builder.create<quake::AllocaOp>(loc, veqTy);
        auto init = builder.create<quake::InitializeStateOp>(loc, veqTy, alloc,
                                                             initials);
        return pushValue(builder.create<quake::ExtractRefOp>(loc, init, 0));
      }
      if (ctorName == "qvector") {
        auto initials = popValue();
        auto *ctx = builder.getContext();
        if (isa<IntegerType>(initials.getType())) {
          // This is the cudaq::qvector(std::size_t) ctor.
          return pushValue(builder.create<quake::AllocaOp>(
              loc, quake::VeqType::getUnsized(ctx), initials));
        }
        if (isa<quake::StateType>(initials.getType()))
          if (auto load = initials.getDefiningOp<cudaq::cc::LoadOp>())
            initials = load.getPtrvalue();
        if (isStateType(initials.getType())) {
          Value state = initials;
          auto i64Ty = builder.getI64Type();
          auto numQubits =
              builder.create<quake::GetNumberOfQubitsOp>(loc, i64Ty, state);
          auto veqTy = quake::VeqType::getUnsized(ctx);
          Value alloc = builder.create<quake::AllocaOp>(loc, veqTy, numQubits);
          Value initSt = builder.create<quake::InitializeStateOp>(loc, veqTy,
                                                                  alloc, state);
          if (auto initOp = initials.getDefiningOp<quake::CreateStateOp>())
            builder.create<quake::DeleteStateOp>(loc, initOp);
          return pushValue(initSt);
        }
        reportClangError(
            x, mangler,
            "internal error: could not determine the number of qubits");
        return false;
      }
      if ((ctorName == "qspan" || ctorName == "qview") &&
          isa<quake::VeqType>(peekValue().getType())) {
        // One of the qspan ctors, which effectively just makes a copy. Here we
        // omit making a copy and just forward the veq argument.
        assert(isa<quake::VeqType>(ctorTy));
        return true;
      }
    }
  } else if (isInNamespace(ctor, "std")) {
    bool isVectorOfQubitRefs = [&]() {
      if (auto *ctor = x->getConstructor()) {
        if (isInNamespace(ctor, "std") && ctor->getNameAsString() == "vector") {
          if (valueStack.empty())
            return false;
          Value v = peekValue();
          return v && isa<quake::VeqType>(v.getType());
        }
      }
      return false;
    }();
    if (isVectorOfQubitRefs)
      return true;
    if (ctorName == "complex") {
      Value imag = popValue();
      Value real = popValue();
      return pushValue(builder.create<complex::CreateOp>(
          loc, ComplexType::get(real.getType()), real, imag));
    }
    if (ctorName == "function") {
      // Are we converting a lambda expr to a std::function?
      auto backVal = peekValue();
      auto backTy = backVal.getType();
      if (auto ptrTy = dyn_cast<cc::PointerType>(backTy))
        backTy = ptrTy.getElementType();
      if (isa<cc::CallableType>(backTy)) {
        // Skip this constructor (for now).
        return true;
      }
      if (auto stTy = dyn_cast<cc::StructType>(backTy)) {
        if (!stTy.getMembers().empty()) {
          // TODO: We don't support a callable class with data members yet.
          TODO_loc(loc, "callable class with data members");
        }
        // Constructor generated as degenerate reference to call operator.
        auto *fromTy = x->getArg(0)->getType().getTypePtr();
        // FIXME: May need to peel off more than one layer of sugar?
        if (auto *elabTy = dyn_cast<clang::ElaboratedType>(fromTy))
          fromTy = elabTy->desugar().getTypePtr();
        auto *fromDecl = dyn_cast_or_null<clang::RecordType>(fromTy)->getDecl();
        if (!fromDecl)
          TODO_loc(loc, "recovering record type for a callable");
        auto *objDecl = dyn_cast_or_null<clang::CXXRecordDecl>(fromDecl);
        if (!objDecl)
          TODO_loc(loc, "recovering C++ declaration for callable");
        auto *callOperDecl = findCallOperator(objDecl);
        if (!callOperDecl) {
          auto &de = mangler->getASTContext().getDiagnostics();
          auto id = de.getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "std::function initializer must be a callable");
          de.Report(x->getBeginLoc(), id);
          return true;
        }
        auto kernelCallTy = cast<cc::CallableType>(ctorTy);
        auto kernelName = generateCudaqKernelName(callOperDecl);
        popValue(); // replace value at TOS.
        return pushValue(builder.create<cc::CreateLambdaOp>(
            loc, kernelCallTy, [&](OpBuilder &builder, Location loc) {
              auto args = builder.getBlock()->getArguments();
              auto call = builder.create<func::CallOp>(
                  loc, kernelCallTy.getSignature().getResults(), kernelName,
                  args);
              builder.create<cc::ReturnOp>(loc, call.getResults());
            }));
      }
    }
    if (ctorName == "reference_wrapper") {
      // The `reference_wrapper` class is used to guide the `qudit&` through a
      // container class (like `std::vector`). It is a NOP at the Quake level.
      [[maybe_unused]] auto tosTy = peekValue().getType();
      assert((isa<quake::RefType, quake::VeqType>(tosTy)));
      return true;
    }

    if (ctorName == "vector") {
      LLVM_DEBUG(llvm::dbgs() << "processing std::vector ctor\n");
      auto processVectorCtor = [&]() -> bool {
        // This is a std::vector constructor. Check if it is constructed from a
        // constant initializer list. In that case, we'll have a AllocaOp at the
        // top of the stack that allocates a ptr<array<T x C>>, where C is
        // constant / known.
        auto desugared = x->getArg(0)->getType().getCanonicalType();
        if (isInitializerListQualType(desugared)) {
          auto allocation = popValue();
          if (auto ptrTy = dyn_cast<cc::PointerType>(allocation.getType()))
            if (auto arrayTy = dyn_cast<cc::ArrayType>(ptrTy.getElementType()))
              if (auto definingOp = allocation.getDefiningOp<cc::AllocaOp>())
                return pushValue(builder.create<cc::StdvecInitOp>(
                    loc, cc::StdvecType::get(arrayTy.getElementType()),
                    allocation, definingOp.getSeqSize()));
        }

        // Next check if its created from a size integer. Let's do a check on
        // the first argument, make sure that when we peel off all the typedefs
        // that it is an integer.
        if (auto builtInType =
                dyn_cast<clang::BuiltinType>(desugared.getTypePtr()))
          if (builtInType->isInteger() &&
              isa<IntegerType>(peekValue().getType())) {
            // This is an integer argument, and the value on the stack is an
            // integer, so let's connect them up
            auto arrSize = popValue();
            auto eleTy = getEleTyFromVectorCtor(ctorTy);

            // Create stdvec init op without a buffer. Allocate the required
            // memory chunk.
            Type ty =
                (eleTy == builder.getI1Type()) ? builder.getI8Type() : eleTy;
            Value alloca = builder.create<cc::AllocaOp>(loc, ty, arrSize);

            // Create the stdvec_init op
            return pushValue(builder.create<cc::StdvecInitOp>(
                loc, cc::StdvecType::get(eleTy), alloca, arrSize));
          }
        return false;
      };

      // Check for vector constructors with 2 args. The first could be an
      // initializer_list or an integer, while the second should be an
      // allocator. This is the libstdc++ implementation. First, verify that the
      // second argument is the expected allocator before continuing.
      if (x->getNumArgs() == 2 &&
          isAllocatorQualType(x->getArg(1)->getType().getCanonicalType())) {
        if (processVectorCtor())
          return true;
      }

      // Check for vector constructors with 1 args. The argument could be an
      // initializer_list or an integer. This case happens when using libc++.
      if (x->getNumArgs() == 1) {
        if (processVectorCtor())
          return true;
      }

      // Disallow any default vector construction bc we don't want any calls
      // to the push_back member function, etc.
      if (ctor->isDefaultConstructor())
        reportClangError(ctor, mangler,
                         "Default std::vector<T> constructor within quantum "
                         "kernel is not allowed (cannot resize the vector).");

      if (ctor->isMoveConstructor()) {
        // Just use the !cc.stdvec<T> value at TOS.
        return true;
      }
    }
  }

  if (isa<quake::StruqType>(ctorTy)) {
    if (quake::isConstantQuantumRefType(ctorTy))
      return pushValue(builder.create<quake::AllocaOp>(loc, ctorTy));
    return true;
  }

  auto *parent = ctor->getParent();
  if (ctor->isCopyConstructor() && parent->isLambda()) {
    // Copy-ctor on a lambda. For now on the QPU device side, we do not make a
    // copy of a lambda. Any capture data will be marshalled at runtime and
    // passed as ordinary arguments via lambda lifting.
    return true;
  }

  if (ctor->isCopyOrMoveConstructor()) {
    // Just walk through copy constructors for quantum struct types.
    if (isa<quake::StruqType>(ctorTy))
      return true;
    if (parent->isPOD()) {
      // Copy or move constructor on a POD struct. The value stack should
      // contain the object to load the value from.
      auto fromStruct = popValue();
      assert(isa<cc::StructType>(ctorTy) && "POD must be a struct type");
      return pushValue(builder.create<cc::LoadOp>(loc, fromStruct));
    }
  }

  if (ctor->isCopyConstructor() && ctor->isTrivial() &&
      isa<cc::StructType>(ctorTy)) {
    auto copyObj = builder.create<cc::AllocaOp>(loc, ctorTy);
    auto fromStruct = popValue();
    auto fromVal = builder.create<cc::LoadOp>(loc, fromStruct);
    builder.create<cc::StoreOp>(loc, fromVal, copyObj);
    return pushValue(builder.create<cc::LoadOp>(loc, copyObj));
  }

  // TODO: remove this when we can handle ctors more generally.
  if (!ctor->isDefaultConstructor()) {
    LLVM_DEBUG(llvm::dbgs() << ctorName << " - unhandled ctor:\n"; x->dump());
    TODO_x(loc, x, mangler, "C++ constructor (non-default)");
  }

  // A C++ constructor lowers as:
  //
  // 1) A unique object must be created, so the type must have a minimum of
  //    one byte.
  // 2) Allocate a new object.
  // 3) If not POD, call the constructor passing the address of the allocation
  //    as `this`.
  auto mem = builder.create<cc::AllocaOp>(loc, ctorTy);

  // No constructor call needed for POD types
  if (parent->isPOD())
    return pushValue(mem);

  // FIXME: Using Ctor_Complete for mangled name generation blindly here. Is
  // there a programmatic way of determining which enum to use from the AST?
  auto mangledName =
      cxxMangledDeclName(clang::GlobalDecl{ctor, clang::Ctor_Complete});
  auto funcTy =
      FunctionType::get(builder.getContext(), TypeRange{mem.getType()}, {});
  auto func = getOrAddFunc(loc, mangledName, funcTy).first;
  // FIXME: The ctor may not be the default ctor. Get all the args.
  builder.create<func::CallOp>(loc, func, ValueRange{mem});
  return pushValue(mem);
}

bool QuakeBridgeVisitor::TraverseCXXDefaultArgExpr(clang::CXXDefaultArgExpr *x,
                                                   DataRecursionQueue *) {
  // Default std::allocator<T> arguments come up in classes like std::vector. We
  // don't want to traverse them.
  if (auto *decl = x->getExpr()->getType()->getAsRecordDecl())
    if (isInNamespace(decl, "std"))
      if (auto *id = decl->getIdentifier())
        if (id->getName() == "allocator")
          return true;
  return TraverseStmt(x->getExpr());
}

bool QuakeBridgeVisitor::TraverseDeclRefExpr(clang::DeclRefExpr *x,
                                             DataRecursionQueue *) {
  auto *decl = x->getDecl();
  if (auto *funcDecl = dyn_cast<clang::FunctionDecl>(decl)) {
    if (inRecType)
      return true;
    return TraverseFunctionDecl(funcDecl);
  }
  return WalkUpFromDeclRefExpr(x);
}

bool QuakeBridgeVisitor::VisitDeclRefExpr(clang::DeclRefExpr *x) {
  auto *decl = x->getDecl();
  assert(!isa<clang::FunctionDecl>(decl) &&
         "FunctionDecl should not reach here");
  if (!symbolTable.count(decl->getName())) {
    // This is a catastrophic error. This symbol is unknown and probably came
    // from a context that is inaccessible from this kernel.
    auto &de = astContext->getDiagnostics();
    const auto id =
        de.getCustomDiagID(clang::DiagnosticsEngine::Error,
                           "symbol is not accessible in this kernel");
    auto db = de.Report(x->getBeginLoc(), id);
    const auto range = x->getSourceRange();
    db.AddSourceRange(clang::CharSourceRange::getCharRange(range));
    raisedError = true;
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "decl ref: " << decl << '\n');
  pushValue(symbolTable.lookup(decl->getName()));
  return true;
}

bool QuakeBridgeVisitor::VisitStringLiteral(clang::StringLiteral *x) {
  auto strLitTy = cc::PointerType::get(cc::ArrayType::get(
      builder.getContext(), builder.getI8Type(), x->getString().size() + 1));
  return pushValue(builder.create<cc::CreateStringLiteralOp>(
      toLocation(x), strLitTy, builder.getStringAttr(x->getString())));
}

} // namespace cudaq::details
