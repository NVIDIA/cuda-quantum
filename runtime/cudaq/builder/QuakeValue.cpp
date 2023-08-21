/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
    if (!value.getType().isa<cc::StdvecType>())
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
      opBuilder.create<arith::ConstantFloatOp>(d, opBuilder.getF64Type()));
}

QuakeValue::QuakeValue(mlir::ImplicitLocOpBuilder &builder, Value v)
    : value(std::make_shared<QuakeValue::ValueHolder>(v)), opBuilder(builder) {}

bool QuakeValue::isStdVec() {
  return value->asMLIR().getType().isa<cc::StdvecType>();
}

std::size_t QuakeValue::getRequiredElements() {
  if (!value->asMLIR().getType().isa<cc::StdvecType>())
    throw std::runtime_error("Tracking unique extraction on non-stdvec type.");
  return value->countUniqueExtractions();
}

QuakeValue QuakeValue::operator[](const std::size_t idx) {
  auto iter = extractedFromIndex.find(idx);
  if (iter != extractedFromIndex.end())
    return iter->second;

  Value vectorValue = value->asMLIR();
  Type type = vectorValue.getType();
  if (!type.isa<cc::StdvecType, quake::VeqType>()) {
    std::string typeName;
    {
      llvm::raw_string_ostream os(typeName);
      type.print(os);
    }

    throw std::runtime_error("This QuakeValue is not subscriptable (" +
                             typeName + ").");
  }

  Value indexVar = opBuilder.create<arith::ConstantIntOp>(idx, 32);

  if (type.isa<quake::VeqType>()) {
    Value extractedQubit =
        opBuilder.create<quake::ExtractRefOp>(vectorValue, indexVar);
    auto ret = extractedFromIndex.emplace(
        std::make_pair(idx, QuakeValue(opBuilder, extractedQubit)));
    return ret.first->second;
  }

  // must be a std vec type
  value->addUniqueExtraction(idx);

  Type eleTy = vectorValue.getType().cast<cc::StdvecType>().getElementType();

  auto arrPtrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
  Value vecPtr = opBuilder.create<cc::StdvecDataOp>(arrPtrTy, vectorValue);
  Type elePtrTy = cc::PointerType::get(eleTy);
  std::int32_t idx32 = static_cast<std::int32_t>(idx);
  Value eleAddr = opBuilder.create<cc::ComputePtrOp>(
      elePtrTy, vecPtr, ArrayRef<cc::ComputePtrArg>{idx32});
  Value loaded = opBuilder.create<cc::LoadOp>(eleAddr);
  auto ret = extractedFromIndex.emplace(
      std::make_pair(idx, QuakeValue(opBuilder, loaded)));
  return ret.first->second;
}

QuakeValue QuakeValue::operator[](const QuakeValue &idx) {
  auto opaquePtr = idx.getValue().getAsOpaquePointer();
  auto iter = extractedFromValue.find(opaquePtr);
  if (iter != extractedFromValue.end())
    return iter->second;

  Value vectorValue = value->asMLIR();
  Type type = vectorValue.getType();
  if (!type.isa<cc::StdvecType, quake::VeqType>()) {
    std::string typeName;
    {
      llvm::raw_string_ostream os(typeName);
      type.print(os);
    }

    throw std::runtime_error("This QuakeValue is not subscriptable (" +
                             typeName + ").");
  }

  Value indexVar = idx.getValue();

  if (type.isa<quake::VeqType>()) {
    Value extractedQubit =
        opBuilder.create<quake::ExtractRefOp>(vectorValue, indexVar);
    auto ret = extractedFromValue.emplace(
        std::make_pair(opaquePtr, QuakeValue(opBuilder, extractedQubit)));
    return ret.first->second;
  }

  if (indexVar.getType().isa<IndexType>())
    indexVar =
        opBuilder.create<arith::IndexCastOp>(opBuilder.getI64Type(), indexVar);

  // We are unable to check that the number of elements have
  // been passed in correctly.
  canValidateVectorNumElements = false;

  Type eleTy = vectorValue.getType().cast<cc::StdvecType>().getElementType();

  Type elePtrTy = cc::PointerType::get(eleTy);
  Value vecPtr = opBuilder.create<cc::StdvecDataOp>(elePtrTy, vectorValue);
  Value eleAddr = opBuilder.create<cc::ComputePtrOp>(
      elePtrTy, vecPtr, ArrayRef<cc::ComputePtrArg>{indexVar});
  Value loaded = opBuilder.create<cc::LoadOp>(eleAddr);
  auto ret = extractedFromValue.emplace(
      std::make_pair(opaquePtr, QuakeValue(opBuilder, loaded)));
  return ret.first->second;
}

QuakeValue QuakeValue::size() {
  Value vectorValue = value->asMLIR();
  Type type = vectorValue.getType();
  if (!type.isa<cc::StdvecType, quake::VeqType>())
    throw std::runtime_error("This QuakeValue does not expose .size().");

  Type i64Ty = opBuilder.getI64Type();
  Value ret;
  if (type.isa<cc::StdvecType>())
    ret = opBuilder.create<cc::StdvecSizeOp>(i64Ty, vectorValue);
  else
    ret = opBuilder.create<quake::VeqSizeOp>(i64Ty, vectorValue);

  return QuakeValue(opBuilder, ret);
}

std::optional<std::size_t> QuakeValue::constantSize() {
  if (auto qvecTy = dyn_cast<quake::VeqType>(getValue().getType()))
    return qvecTy.getSize();

  return std::nullopt;
}

QuakeValue QuakeValue::slice(const std::size_t startIdx,
                             const std::size_t count) {
  Value vectorValue = value->asMLIR();
  Type type = vectorValue.getType();
  if (!type.isa<cc::StdvecType, quake::VeqType>())
    throw std::runtime_error("This QuakeValue is not sliceable.");

  if (count == 0)
    throw std::runtime_error("QuakeValue::slice requesting slice of size 0.");

  Value startIdxValue = opBuilder.create<arith::ConstantIntOp>(startIdx, 64);
  Value countValue = opBuilder.create<arith::ConstantIntOp>(count, 64);
  if (auto veqType = type.dyn_cast_or_null<quake::VeqType>()) {
    auto veqSize = veqType.getSize();
    if (startIdx + count > veqSize)
      throw std::runtime_error("Invalid number of elements requested in slice, "
                               "must be less than size of array (" +
                               std::to_string(veqSize) + ").");

    auto one = opBuilder.create<arith::ConstantIntOp>(1, 64);
    Value offset = opBuilder.create<arith::AddIOp>(startIdxValue, countValue);
    offset = opBuilder.create<arith::SubIOp>(offset, one);
    auto sizedVecTy = quake::VeqType::get(opBuilder.getContext(), count);
    Value subVeq = opBuilder.create<quake::SubVeqOp>(sizedVecTy, vectorValue,
                                                     startIdxValue, offset);
    return QuakeValue(opBuilder, subVeq);
  }

  // must be a stdvec type
  auto svecTy = dyn_cast<cc::StdvecType>(vectorValue.getType());
  auto eleTy = svecTy.getElementType();
  assert(!isa<cc::ArrayType>(eleTy));
  Type ptrTy;
  Value vecPtr;
  Value offset;
  if (eleTy == opBuilder.getI1Type()) {
    // This is a workaround for when we go to LLVM. This workaround should
    // actually appear in CodeGen when lowering this to the LLVM-IR dialect.
    auto newEleTy = cc::ArrayType::get(opBuilder.getI8Type());
    ptrTy = cc::PointerType::get(newEleTy);
    vecPtr = opBuilder.create<cc::StdvecDataOp>(ptrTy, vectorValue);
    auto bits = svecTy.getElementType().getIntOrFloatBitWidth();
    assert(bits > 0);
    auto scale = opBuilder.create<arith::ConstantIntOp>(
        (bits + 7) / 8, startIdxValue.getType());
    offset = opBuilder.create<arith::MulIOp>(scale, startIdxValue);
  } else {
    ptrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
    vecPtr = opBuilder.create<cc::StdvecDataOp>(ptrTy, vectorValue);
    offset = startIdxValue;
  }
  auto ptr = opBuilder.create<cc::ComputePtrOp>(
      ptrTy, vecPtr, ArrayRef<cc::ComputePtrArg>{offset});
  Value subVeqInit = opBuilder.create<cc::StdvecInitOp>(vectorValue.getType(),
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

  Value negated = opBuilder.create<arith::NegFOp>(v.getType(), v);
  return QuakeValue(opBuilder, negated);
}

QuakeValue QuakeValue::operator*(const double constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only multiply double/float QuakeValues.");

  llvm::APFloat d(constValue);
  Value constant =
      opBuilder.create<arith::ConstantFloatOp>(d, opBuilder.getF64Type());
  Value multiplied = opBuilder.create<arith::MulFOp>(v.getType(), constant, v);
  return QuakeValue(opBuilder, multiplied);
}

QuakeValue QuakeValue::operator*(QuakeValue other) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only multiply double/float QuakeValues.");

  auto otherV = other.value->asMLIR();
  if (!otherV.getType().isIntOrFloat())
    throw std::runtime_error("Can only multiply double/float QuakeValues.");

  Value multiplied = opBuilder.create<arith::MulFOp>(v.getType(), v, otherV);
  return QuakeValue(opBuilder, multiplied);
}

QuakeValue QuakeValue::operator+(const double constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only add double/float QuakeValues.");

  llvm::APFloat d(constValue);
  Value constant =
      opBuilder.create<arith::ConstantFloatOp>(d, opBuilder.getF64Type());
  Value added = opBuilder.create<arith::AddFOp>(v.getType(), constant, v);
  return QuakeValue(opBuilder, added);
}

QuakeValue QuakeValue::operator+(const int constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrIndex())
    throw std::runtime_error("Can only add int/index QuakeValues.");

  Value constant;
  if (isa<IndexType>(v.getType())) {
    constant = opBuilder.create<arith::ConstantIndexOp>(constValue);
  } else {
    constant = opBuilder.create<arith::ConstantIntOp>(constValue, v.getType());
  }
  Value added = opBuilder.create<arith::AddIOp>(v.getType(), constant, v);
  return QuakeValue(opBuilder, added);
}

QuakeValue QuakeValue::operator+(QuakeValue other) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only add double/float QuakeValues.");

  auto otherV = other.value->asMLIR();
  if (!otherV.getType().isIntOrFloat())
    throw std::runtime_error("Can only add double/float QuakeValues.");

  Value added = opBuilder.create<arith::AddFOp>(v.getType(), v, otherV);
  return QuakeValue(opBuilder, added);
}

QuakeValue QuakeValue::operator-(const double constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only subtract double/float QuakeValues.");

  llvm::APFloat d(constValue);
  Value constant =
      opBuilder.create<arith::ConstantFloatOp>(d, opBuilder.getF64Type());
  Value subtracted = opBuilder.create<arith::SubFOp>(v.getType(), v, constant);
  return QuakeValue(opBuilder, subtracted);
}

QuakeValue QuakeValue::operator-(const int constValue) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrIndex())
    throw std::runtime_error("Can only subtract double/float QuakeValues.");

  Value constant;
  if (isa<IndexType>(v.getType())) {
    constant = opBuilder.create<arith::ConstantIndexOp>(constValue);
  } else {
    constant = opBuilder.create<arith::ConstantIntOp>(constValue, v.getType());
  }

  Value subtracted = opBuilder.create<arith::SubIOp>(v.getType(), v, constant);
  return QuakeValue(opBuilder, subtracted);
}

QuakeValue QuakeValue::operator-(QuakeValue other) {
  auto v = value->asMLIR();
  if (!v.getType().isIntOrFloat())
    throw std::runtime_error("Can only subtract double/float QuakeValues.");

  auto otherV = other.value->asMLIR();
  if (!otherV.getType().isIntOrFloat())
    throw std::runtime_error("Can only subtract double/float QuakeValues.");

  Value subtracted = opBuilder.create<arith::SubFOp>(v.getType(), v, otherV);
  return QuakeValue(opBuilder, subtracted);
}
} // namespace cudaq
