/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cudaq {
namespace cc {
class LoopOp;
class PointerType;
class StructType;
} // namespace cc

namespace opt {

template <typename T>
  requires std::integral<T>
T convertBitsToBytes(T bits) {
  return (bits + 7) / 8;
}

namespace factory {

constexpr const char targetTripleAttrName[] = "llvm.triple";
constexpr const char targetDataLayoutAttrName[] = "llvm.data_layout";

//===----------------------------------------------------------------------===//
// Type builders
//===----------------------------------------------------------------------===//

/// Return the LLVM-IR dialect void type.
inline mlir::Type getVoidType(mlir::MLIRContext *ctx) {
  return mlir::LLVM::LLVMVoidType::get(ctx);
}

inline mlir::Type getCharType(mlir::MLIRContext *ctx) {
  return mlir::IntegerType::get(ctx, /*bits=*/8);
}

/// Return the LLVM-IR dialect `ptr` type.
inline mlir::Type getPointerType(mlir::MLIRContext *ctx) {
  return mlir::LLVM::LLVMPointerType::get(getCharType(ctx));
}

/// The type of a dynamic buffer as returned via the runtime.
cudaq::cc::StructType getDynamicBufferType(mlir::MLIRContext *ctx);

/// Extract the element type of a `sret` return result.
mlir::Type getSRetElementType(mlir::FunctionType funcTy);

/// Do not use this yet. Opaque pointers are all or nothing.
inline mlir::Type getOpaquePointerType(mlir::MLIRContext *ctx) {
  return mlir::LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/0);
}

/// Return the LLVM-IR dialect type: `ty*`.
inline mlir::Type getPointerType(mlir::Type ty) {
  return mlir::LLVM::LLVMPointerType::get(ty);
}

/// Get the Quake type translation of a `cudaq::state` type.
cc::StructType getCudaqStateType(mlir::MLIRContext *ctx);

cudaq::cc::PointerType getIndexedObjectType(mlir::Type eleTy);

mlir::Type genArgumentBufferType(mlir::Type ty);

/// Build an LLVM struct type with all the arguments and then all the results.
/// If the type is a std::vector, then add an i64 to the struct for the
/// length. The actual data values will be appended to the end of the
/// dynamically sized struct.
///
/// A kernel signature of
/// ```c++
/// i32_t operator() (i16_t, std::vector<double>, double);
/// ```
/// will generate the LLVM struct
/// ```llvm
/// { i16, i64, double, i32 }
/// ```
/// where the values of the vector argument are pass-by-value and appended to
/// the end of the struct as a sequence of \i n double values.
cudaq::cc::StructType buildInvokeStructType(mlir::FunctionType funcTy);

/// Return the LLVM-IR dialect type: `[length x i8]`.
inline mlir::Type getStringType(mlir::MLIRContext *ctx, std::size_t length) {
  return mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(ctx, 8), length);
}

/// Return the QPU-side version of a `std::vector<T>` when lowered to a plain
/// old C `struct`. Currently, the QPU-side struct is `{ T*, i64 }` where the
/// fields are the buffer pointer and a length (in number of elements). The size
/// of each element (which shall be a basic numeric type) is inferred from
/// \p eleTy (`T`).
inline mlir::LLVM::LLVMStructType stdVectorImplType(mlir::Type eleTy) {
  auto *ctx = eleTy.getContext();
  // Map stdvec<complex<T>> to stdvec<struct<T,T>>
  if (auto cTy = dyn_cast<mlir::ComplexType>(eleTy)) {
    llvm::SmallVector<mlir::Type> types = {cTy.getElementType(),
                                           cTy.getElementType()};
    eleTy = mlir::LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  auto elePtrTy = cudaq::opt::factory::getPointerType(eleTy);
  auto i64Ty = mlir::IntegerType::get(ctx, 64);
  llvm::SmallVector<mlir::Type> eleTys = {elePtrTy, i64Ty};
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, eleTys);
}

// Host side types for std::string and std::vector

cudaq::cc::StructType stlStringType(mlir::MLIRContext *ctx);
cudaq::cc::StructType stlVectorType(mlir::Type eleTy);

//===----------------------------------------------------------------------===//
// Constant builders
//===----------------------------------------------------------------------===//

/// Generate an LLVM IR dialect constant with type `i32` for a specific value.
inline mlir::LLVM::ConstantOp genLlvmI32Constant(mlir::Location loc,
                                                 mlir::OpBuilder &builder,
                                                 std::int32_t val) {
  auto idx = builder.getI32IntegerAttr(val);
  auto i32Ty = builder.getI32Type();
  return builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, idx);
}

inline mlir::LLVM::ConstantOp genLlvmI64Constant(mlir::Location loc,
                                                 mlir::OpBuilder &builder,
                                                 std::int64_t val) {
  auto idx = builder.getI64IntegerAttr(val);
  auto i64Ty = builder.getI64Type();
  return builder.create<mlir::LLVM::ConstantOp>(loc, i64Ty, idx);
}

inline mlir::Value createFloatConstant(mlir::Location loc,
                                       mlir::OpBuilder &builder,
                                       llvm::APFloat value,
                                       mlir::FloatType type) {
  return builder.create<mlir::arith::ConstantFloatOp>(loc, value, type);
}

inline mlir::Value createFloatConstant(mlir::Location loc,
                                       mlir::OpBuilder &builder, double value,
                                       mlir::FloatType type) {
  return createFloatConstant(loc, builder, llvm::APFloat(value), type);
}

inline mlir::Value createF64Constant(mlir::Location loc,
                                     mlir::OpBuilder &builder, double value) {
  return createFloatConstant(loc, builder, value, builder.getF64Type());
}

/// Return the integer value if \p v is an integer constant.
std::optional<std::uint64_t> maybeValueOfIntConstant(mlir::Value v);

/// Return the floating point value if \p v is a floating-point constant.
std::optional<double> maybeValueOfFloatConstant(mlir::Value v);

//===----------------------------------------------------------------------===//

inline mlir::Block *addEntryBlock(mlir::LLVM::GlobalOp initVar) {
  auto *entry = new mlir::Block;
  initVar.getRegion().push_back(entry);
  return entry;
}

/// Return an i64 array where element `k` is `N` if the
/// operand `k` is `veq<N>` and 0 otherwise.
mlir::Value packIsArrayAndLengthArray(mlir::Location loc,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      mlir::ModuleOp parentModule,
                                      mlir::Value numOperands,
                                      mlir::ValueRange operands);
mlir::FlatSymbolRefAttr
createLLVMFunctionSymbol(mlir::StringRef name, mlir::Type retType,
                         mlir::ArrayRef<mlir::Type> inArgTypes,
                         mlir::ModuleOp module, bool isVar = false);

mlir::func::FuncOp createFunction(mlir::StringRef name,
                                  mlir::ArrayRef<mlir::Type> retTypes,
                                  mlir::ArrayRef<mlir::Type> inArgTypes,
                                  mlir::ModuleOp module);

void createGlobalCtorCall(mlir::ModuleOp mod, mlir::FlatSymbolRefAttr ctor);

/// Builds a simple invariant loop. A simple invariant loop is a loop that is
/// guaranteed to execute the body of the loop \p totalIterations times. Early
/// exits are not allowed. This builder threads the loop control value, which
/// will be returned as the value \p totalIterations when the loop exits.
cc::LoopOp
createInvariantLoop(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value totalIterations,
                    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                            mlir::Region &, mlir::Block &)>
                        bodyBuilder);

/// Builds a monotonic loop. A monotonic loop is a loop that is guaranteed to
/// execute the body of the loop from \p start to (but not including) \p stop
/// stepping by \p step times. Exceptional conditions will cause the loop body
/// to execute 0 times. Early exits are not allowed. This builder threads the
/// loop control value, which will be returned as the value \p stop (or the next
/// value near \p stop) when the loop exits.
cc::LoopOp
createMonotonicLoop(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value start, mlir::Value stop, mlir::Value step,
                    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                            mlir::Region &, mlir::Block &)>
                        bodyBuilder);

bool hasHiddenSRet(mlir::FunctionType funcTy);

/// Convert the function type \p funcTy to a signature compatible with the code
/// on the host side. This will add hidden arguments, such as the `this`
/// pointer, convert some results to `sret` pointers, etc.
mlir::FunctionType toHostSideFuncType(mlir::FunctionType funcTy,
                                      bool addThisPtr, mlir::ModuleOp module);

// Return `true` if the given type corresponds to a standard vector type
// according to our convention.
// The convention is a `ptr<struct<ptr<T>, ptr<T>, ptr<T>>>`.
bool isStdVecArg(mlir::Type type);

bool isX86_64(mlir::ModuleOp);
bool isAArch64(mlir::ModuleOp);

/// A small structure may be passed as two arguments on the host side. (e.g., on
/// the X86-64 ABI.) If \p ty is not a `struct`, this returns `false`.
bool structUsesTwoArguments(mlir::Type ty);

} // namespace factory
} // namespace opt
} // namespace cudaq
