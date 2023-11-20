/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/Factory.h"

namespace cudaq {

static constexpr const char llvmMemCopyIntrinsic[] =
    "llvm.memcpy.p0i8.p0i8.i64";
static constexpr const char setCudaqRangeVector[] = "__nvqpp_CudaqRangeInit";
static constexpr const char stdvecBoolCtorFromInitList[] =
    "__nvqpp_initializer_list_to_vector_bool";

/// Builder for lowering the clang AST to an IR for CUDA Quantum. Lowering
/// includes the transformation of both quantum and classical computation.
/// Different features of the CUDA Quantum programming model are lowered into
/// different dialects of MLIR. This builder makes heavy use of the Quake
/// (QUAntum Kernel Execution) and CC (Classical Computation) dialects.
///
/// This builder also allows for the inclusion of predefined intrinsics into
/// the ModuleOp on demand. Intrinsics exist in a map accessed by a symbol name.
class IRBuilder : public mlir::OpBuilder {
public:
  using OpBuilder::OpBuilder;

  mlir::LLVM::ConstantOp genLlvmI32Constant(mlir::Location loc,
                                            std::int32_t val) {
    return opt::factory::genLlvmI32Constant(loc, *this, val);
  }

  /// Create a global for a C-style string. (A pointer to a NUL terminated
  /// sequence of bytes.) \p cstring must have the NUL character appended \b
  /// prior to calling this builder function.
  mlir::LLVM::GlobalOp genCStringLiteral(mlir::Location loc,
                                         mlir::ModuleOp module,
                                         llvm::StringRef cstring);

  /// Helper function to create a C-style string from a string that is not
  /// already NUL terminated. No checks are made, so if the string already is
  /// NUL terminated, a second NUL is appended.
  mlir::LLVM::GlobalOp genCStringLiteralAppendNul(mlir::Location loc,
                                                  mlir::ModuleOp module,
                                                  llvm::StringRef cstring) {
    auto buffer = cstring.str();
    buffer += '\0';
    return genCStringLiteral(loc, module, buffer);
  }

  /// Load an intrinsic into \p module. The intrinsic to load has name \p name.
  /// This will automatically load any intrinsics that \p name depends upon.
  /// Return `failure()` when \p name is not in the table of known intrinsics.
  mlir::LogicalResult loadIntrinsic(mlir::ModuleOp module,
                                    llvm::StringRef name);

  std::string hashStringByContent(llvm::StringRef sref);

  static IRBuilder atBlockEnd(mlir::Block *block) {
    return IRBuilder(block, block->end(), nullptr);
  }

  static IRBuilder atBlockTerminator(mlir::Block *block) {
    auto *terminator = block->getTerminator();
    assert(terminator && "block has no terminator");
    return IRBuilder(block, mlir::Block::iterator(terminator), nullptr);
  }
};

} // namespace cudaq
