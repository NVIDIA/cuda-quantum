/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/Tools/mlir-translate/Translation.h"
#include <memory>

namespace mlir {
class MLIRContext;
class ExecutionEngine;
class ModuleOp;
} // namespace mlir

namespace llvm {
class Module;
}

namespace cudaq {
/// @brief Initialize MLIR with CUDA Quantum dialects and return the
/// MLIRContext.
std::unique_ptr<mlir::MLIRContext> initializeMLIR();
/// @brief Given an LLVM Module, set its target triple corresponding to the
/// current host machine.
bool setupTargetTriple(llvm::Module *);

/// @brief Run the LLVM PassManager.
void optimizeLLVM(llvm::Module *);

/// @brief Lower ModuleOp to a full QIR LLVMIR representation
/// and return an ExecutionEngine pointer for JIT function pointer
/// execution. Clients are responsible for deleting this pointer.
mlir::ExecutionEngine *createQIRJITEngine(mlir::ModuleOp &moduleOp);

class Translation {
public:
  Translation() = default;
  Translation(mlir::TranslateFromMLIRFunction function,
              llvm::StringRef description)
      : function(std::move(function)), description(description) {}

  /// Return the description of this translation.
  llvm::StringRef getDescription() const { return description; }

  /// Invoke the translation function with the given input and output streams.
  mlir::LogicalResult operator()(mlir::Operation *op,
                                 llvm::raw_ostream &output) const {
    return function(op, output);
  }

private:
  /// The underlying translation function.
  mlir::TranslateFromMLIRFunction function;

  /// The description of the translation.
  llvm::StringRef description;
};

cudaq::Translation &getTranslation(llvm::StringRef name);

struct TranslateFromMLIRRegistration {
  TranslateFromMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const mlir::TranslateFromMLIRFunction &function);
};

} // namespace cudaq
