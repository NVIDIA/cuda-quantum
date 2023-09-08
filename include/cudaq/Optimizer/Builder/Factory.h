/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

namespace cudaq {
namespace cc {
class LoopOp;
}

namespace opt::factory {

/// Return the LLVM-IR dialect void type.
inline mlir::Type getVoidType(mlir::MLIRContext *ctx) {
  return mlir::LLVM::LLVMVoidType::get(ctx);
}

inline mlir::Type getCharType(mlir::MLIRContext *ctx) {
  return mlir::IntegerType::get(ctx, /*bits=*/8);
}

/// Return the LLVM-IR dialect ptr type.
inline mlir::Type getPointerType(mlir::MLIRContext *ctx) {
  return mlir::LLVM::LLVMPointerType::get(getCharType(ctx));
}

/// Return the LLVM-IR dialect type: `ty*`.
inline mlir::Type getPointerType(mlir::Type ty) {
  return mlir::LLVM::LLVMPointerType::get(ty);
}

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
  auto elePtrTy = cudaq::opt::factory::getPointerType(eleTy);
  auto i64Ty = mlir::IntegerType::get(ctx, 64);
  llvm::SmallVector<mlir::Type> eleTys = {elePtrTy, i64Ty};
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, eleTys);
}

/// Generate an LLVM IR dialect constant with type `i32` for a specific value.
inline mlir::LLVM::ConstantOp
genI32Constant(mlir::Location loc, mlir::OpBuilder &builder, std::int32_t val) {
  auto idx = builder.getI32IntegerAttr(val);
  auto i32Ty = builder.getIntegerType(32);
  return builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, idx);
}

inline mlir::Block *addEntryBlock(mlir::LLVM::GlobalOp initVar) {
  auto *entry = new mlir::Block;
  initVar.getRegion().push_back(entry);
  return entry;
}

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

bool hasHiddenSRet(mlir::FunctionType funcTy);

/// Convert the function type \p funcTy to a signature compatible with the code
/// on the CPU side. This will add hidden arguments, such as the `this` pointer,
/// convert some results to `sret` pointers, etc.
mlir::FunctionType toCpuSideFuncType(mlir::FunctionType funcTy,
                                     bool addThisPtr);

/// @brief Return true if the given type corresponds to a
/// std-vector type according to our convention. The convention
/// is a ptr<struct<ptr<T>, ptr<T>, ptr<T>>>.
bool isStdVecArg(mlir::Type type);

} // namespace opt::factory
} // namespace cudaq
