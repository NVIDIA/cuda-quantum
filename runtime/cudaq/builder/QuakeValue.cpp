/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuakeValue.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <set>

using namespace mlir;

namespace cudaq {

/// @brief Pimpl Type wrapping an MLIR Value.
class QuakeValue::ValueHolder {
protected:
  /// @brief MLIR value
  Value value;

  /// @brief Set used to count or keep track of
  /// all unique vector-like extractions
  std::set<std::size_t> uniqueExtractions;

public:
  /// @brief The constructor
  ValueHolder(Value v) : value(v) {}
  ValueHolder(ValueHolder &) = default;
  ~ValueHolder() = default;

  /// @brief Whenever we encounter an extract on a
  /// StdVec QuakeValue, we want to record it here. This
  /// allows us to validate later that the number of runtime
  /// std::vector elements are correct.
  void addUniqueExtraction(std::size_t idx) {
    if (!isa<cc::StdvecType>(value.getType()))
      throw std::runtime_error(
          "Tracking unique extraction on non-stdvec type.");

    uniqueExtractions.insert(idx);
  }

  /// @brief Return the number of unique extractions
  /// this is how many elements any input vector needs
  /// to have present.
  std::size_t countUniqueExtractions() {
    if (uniqueExtractions.empty())
      return 0;
    // std set ordered by default, get last element
    return *uniqueExtractions.rbegin() + 1;
  }

  /// @brief Return the MLIR value
  Value asMLIR() { return value; }
};

QuakeValue::~QuakeValue() = default;
QuakeValue::QuakeValue(QuakeValue &) = default;
QuakeValue::QuakeValue(const QuakeValue &) = default;

void QuakeValue::dump() { value->asMLIR().dump(); }
void QuakeValue::dump(std::ostream &os) {
  std::string s;
  {
    llvm::raw_string_ostream os(s);
    value->asMLIR().print(os);
  }
  os << s;
}

QuakeValue::QuakeValue(QuakeValue &&) = default;
QuakeValue::QuakeValue(mlir::ImplicitLocOpBuilder &builder, double v)
    : opBuilder(builder) {
  llvm::APFloat d(v);
  value = std::make_shared<QuakeValue::ValueHolder>(
      arith::ConstantFloatOp::create(opBuilder, opBuilder.getF64Type(), d));
}

QuakeValue::QuakeValue(mlir::ImplicitLocOpBuilder &builder, Value v)
    : value(std::make_shared<QuakeValue::ValueHolder>(v)), opBuilder(builder) {}

bool QuakeValue::isStdVec() {
  return isa<cc::StdvecType>(value->asMLIR().getType());
}

std::size_t QuakeValue::getRequiredElements() {
  if (!isStdVec())
    throw std::runtime_error("Tracking unique extraction on non-stdvec type.");
  return value->countUniqueExtractions();
}

QuakeValue QuakeValue::operator[](const std::size_t idx) {
  Value vectorValue = value->asMLIR();
  Type type = vectorValue.getType();
  if (!isa<cc::StdvecType, quake::VeqType>(type)) {
    std::string typeName;
    {
      llvm::raw_string_ostream os(typeName);
      type.print(os);
    }

    throw std::runtime_error("This QuakeValue is not subscriptable (" +
                             typeName + ").");
  }

  Value indexVar = arith::ConstantIntOp::create(opBuilder, idx, 32);

  if (isa<quake::VeqType>(type)) {
    Value extractedQubit =
        quake::ExtractRefOp::create(opBuilder, vectorValue, indexVar);
    return QuakeValue(opBuilder, extractedQubit);
  }

  // must be a std vec type
  value->addUniqueExtraction(idx);

  Type eleTy =
      mlir::cast<cc::StdvecType>(vectorValue.getType()).getElementType();

  auto arrPtrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
  Value vecPtr = cc::StdvecDataOp::create(opBuilder, arrPtrTy, vectorValue);
  std::int32_t idx32 = static_cast<std::int32_t>(idx);
  auto elePtrTy = cc::PointerType::get(eleTy);
  Value eleAddr = cc::ComputePtrOp::create(
      opBuilder, elePtrTy, vecPtr, ArrayRef<cc::ComputePtrArg>{idx32});
  Value loaded = cc::LoadOp::create(opBuilder, eleAddr);
  return QuakeValue(opBuilder, loaded);
}

QuakeValue QuakeValue::operator[](const QuakeValue &idx) {
  Value vectorValue = value->asMLIR();
  Type type = vectorValue.getType();
  if (!isa<cc::StdvecType, quake::VeqType>(type)) {
    std::string typeName;
    {
      llvm::raw_string_ostream os(typeName);
      type.print(os);
    }

    throw std::runtime_error("This QuakeValue is not subscriptable (" +
                             typeName + ").");
  }

  Value indexVar = idx.getValue();

  if (isa<quake::VeqType>(type)) {
    Value extractedQubit =
        quake::ExtractRefOp::create(opBuilder, vectorValue, indexVar);
    return QuakeValue(opBuilder, extractedQubit);
  }

  // We are unable to check that the number of elements have
  // been passed in correctly.
  canValidateVectorNumElements = false;

  Type eleTy =
      mlir::cast<cc::StdvecType>(vectorValue.getType()).getElementType();
  auto arrEleTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
  Value vecPtr = cc::StdvecDataOp::create(opBuilder, arrEleTy, vectorValue);
  auto elePtrTy = cc::PointerType::get(eleTy);
  Value eleAddr = cc::ComputePtrOp::create(
      opBuilder, elePtrTy, vecPtr, ArrayRef<cc::ComputePtrArg>{indexVar});
  Value loaded = cc::LoadOp::create(opBuilder, eleAddr);
  return QuakeValue(opBuilder, loaded);
}

QuakeValue QuakeValue::size() {
  Value vectorValue = value->asMLIR();
  Type type = vectorValue.getType();
  if (!isa<cc::StdvecType, quake::VeqType>(type))
    throw std::runtime_error("This QuakeValue does not expose .size().");

  Type i64Ty = opBuilder.getI64Type();
  Value ret;
  if (isa<cc::StdvecType>(type))
    ret = cc::StdvecSizeOp::create(opBuilder, i64Ty, vectorValue);
  else
    ret = quake::VeqSizeOp::create(opBuilder, i64Ty, vectorValue);

  return QuakeValue(opBuilder, ret);
}

std::optional<std::size_t> QuakeValue::constantSize() {
  if (auto qvecTy = dyn_cast<quake::VeqType>(getValue().getType()))
    if (qvecTy.hasSpecifiedSize())
      return qvecTy.getSize();

  return std::nullopt;
}

QuakeValue QuakeValue::slice(const std::size_t startIdx,
                             const std::size_t count) {
  Value vectorValue = value->asMLIR();
  Type type = vectorValue.getType();
  if (!isa<cc::StdvecType, quake::VeqType>(type))
    throw std::runtime_error("This QuakeValue is not sliceable.");

  if (count == 0)
    throw std::runtime_error("QuakeValue::slice requesting slice of size 0.");

  Value startIdxValue =
      arith::ConstantIntOp::create(opBuilder, startIdx, 64);
  Value countValue = arith::ConstantIntOp::create(opBuilder, count, 64);
  if (auto veqType = mlir::dyn_cast_if_present<quake::VeqType>(type)) {
    auto veqSize = veqType.getSize();
    if (startIdx + count > veqSize)
      throw std::runtime_error("Invalid number of elements requested in slice, "
                               "must be less than size of array (" +
                               std::to_string(veqSize) + ").");

    auto one = arith::ConstantIntOp::create(opBuilder, 1, 64);
    Value offset =
        arith::AddIOp::create(opBuilder, startIdxValue, countValue);
    offset = arith::SubIOp::create(opBuilder, offset, one);
    auto sizedVecTy = quake::VeqType::get(opBuilder.getContext(), count);
    Value subVeq = quake::SubVeqOp::create(opBuilder, sizedVecTy, vectorValue,
                                            startIdxValue, offset);
    return QuakeValue(opBuilder, subVeq);
  }

  // must be a stdvec type
  auto svecTy = dyn_cast<cc::StdvecType>(vectorValue.getType());
  auto eleTy = svecTy.getElementType();
  assert(!isa<cc::ArrayType>(eleTy));
  Value vecPtr;
  Value offset;
  if (eleTy == opBuilder.getI1Type()) {
    // This is a workaround for when we go to LLVM. This workaround should
    // actually appear in CodeGen when lowering this to the LLVM-IR dialect.
    eleTy = opBuilder.getI8Type();
    auto ptrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
    vecPtr = cc::StdvecDataOp::create(opBuilder, ptrTy, vectorValue);
    auto bits = svecTy.getElementType().getIntOrFloatBitWidth();
    assert(bits > 0);
    auto scale = arith::ConstantIntOp::create(
        opBuilder, startIdxValue.getType(), (bits + 7) / 8);
    offset = arith::MulIOp::create(opBuilder, scale, startIdxValue);
  } else {
    auto ptrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
    vecPtr = cc::StdvecDataOp::create(opBuilder, ptrTy, vectorValue);
    offset = startIdxValue;
  }
  auto ptr = cc::ComputePtrOp::create(
      opBuilder, cudaq::cc::PointerType::get(eleTy), vecPtr,
      ArrayRef<cc::ComputePtrArg>{offset});
  Value subVeqInit = cc::StdvecInitOp::create(opBuilder, vectorValue.getType(),
                                               ptr, countValue);

  // If this is a slice, then we know we have
  // unique extraction on the elements of the slice,
  // which will be element startIdx, startIdx+1,...startIdx+count-1
  for (std::size_t i = startIdx; i < startIdx + count; i++)
    value->addUniqueExtraction(i);

  return QuakeValue(opBuilder, subVeqInit);
}

mlir::Value QuakeValue::getValue() const { return value->asMLIR(); }

QuakeValue QuakeValue::operator-() const {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only negate double/float QuakeValues.");

  Value negated = arith::NegFOp::create(opBuilder, v.getType(), v);
  return QuakeValue(opBuilder, negated);
}

QuakeValue QuakeValue::operator*(const double constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only multiply double/float QuakeValues.");

  llvm::APFloat d(constValue);
  Value constant =
      arith::ConstantFloatOp::create(opBuilder, opBuilder.getF64Type(), d);
  Value multiplied = arith::MulFOp::create(opBuilder, v.getType(), constant, v);
  return QuakeValue(opBuilder, multiplied);
}

QuakeValue QuakeValue::operator*(QuakeValue other) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only multiply double/float QuakeValues.");

  auto otherV = other.value->asMLIR();
  if (!otherV.getType().isIntOrFloat())
    throw std::runtime_error("Can only multiply double/float QuakeValues.");

  Value multiplied = arith::MulFOp::create(opBuilder, v.getType(), v, otherV);
  return QuakeValue(opBuilder, multiplied);
}

QuakeValue QuakeValue::operator/(const double constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only divide double/float QuakeValues.");

  llvm::APFloat d(constValue);
  Value constant =
      arith::ConstantFloatOp::create(opBuilder, opBuilder.getF64Type(), d);
  Value div = arith::DivFOp::create(opBuilder, v.getType(), v, constant);
  return QuakeValue(opBuilder, div);
}

QuakeValue QuakeValue::operator/(QuakeValue other) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only divide double/float QuakeValues.");

  auto otherV = other.value->asMLIR();
  if (!otherV.getType().isIntOrFloat())
    throw std::runtime_error("Can only divide double/float QuakeValues.");

  Value div = arith::DivFOp::create(opBuilder, v.getType(), v, otherV);
  return QuakeValue(opBuilder, div);
}

QuakeValue QuakeValue::operator+(const double constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only add double/float QuakeValues.");

  llvm::APFloat d(constValue);
  Value constant =
      arith::ConstantFloatOp::create(opBuilder, opBuilder.getF64Type(), d);
  Value added = arith::AddFOp::create(opBuilder, v.getType(), constant, v);
  return QuakeValue(opBuilder, added);
}

QuakeValue QuakeValue::operator+(const int constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrIndex())
    throw std::runtime_error("Can only add integral QuakeValues.");

  Value constant =
      arith::ConstantIntOp::create(opBuilder, v.getType(), constValue);
  Value added = arith::AddIOp::create(opBuilder, v.getType(), constant, v);
  return QuakeValue(opBuilder, added);
}

QuakeValue QuakeValue::operator+(QuakeValue other) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only add double/float QuakeValues.");

  auto otherV = other.value->asMLIR();
  if (!otherV.getType().isIntOrFloat())
    throw std::runtime_error("Can only add double/float QuakeValues.");

  Value added = arith::AddFOp::create(opBuilder, v.getType(), v, otherV);
  return QuakeValue(opBuilder, added);
}

QuakeValue QuakeValue::operator-(const double constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only subtract double/float QuakeValues.");

  llvm::APFloat d(constValue);
  Value constant =
      arith::ConstantFloatOp::create(opBuilder, opBuilder.getF64Type(), d);
  Value subtracted = arith::SubFOp::create(opBuilder, v.getType(), v, constant);
  return QuakeValue(opBuilder, subtracted);
}

QuakeValue QuakeValue::operator-(const int constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrIndex())
    throw std::runtime_error("Can only subtract double/float QuakeValues.");

  Value constant =
      arith::ConstantIntOp::create(opBuilder, v.getType(), constValue);

  Value subtracted = arith::SubIOp::create(opBuilder, v.getType(), v, constant);
  return QuakeValue(opBuilder, subtracted);
}

QuakeValue QuakeValue::operator-(QuakeValue other) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only subtract double/float QuakeValues.");

  auto otherV = other.value->asMLIR();
  if (!otherV.getType().isIntOrFloat())
    throw std::runtime_error("Can only subtract double/float QuakeValues.");

  Value subtracted = arith::SubFOp::create(opBuilder, v.getType(), v, otherV);
  return QuakeValue(opBuilder, subtracted);
}

QuakeValue QuakeValue::inverse() const {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only inverse double/float QuakeValues.");
  Value constantOne = arith::ConstantFloatOp::create(
      opBuilder, opBuilder.getF64Type(), llvm::APFloat(1.0));
  Value inv = arith::DivFOp::create(opBuilder, v.getType(), constantOne, v);
  return QuakeValue(opBuilder, inv);
}
} // namespace cudaq
