/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
/// @brief Function to lower MLIR to target
/// @param op MLIR operation
/// @param output Output stream
/// @param additionalPasses Additional passes to run at the end
/// @param printIR Print IR to `stderr`
/// @param printIntermediateMLIR Print IR in between each pass
/// @param printStats Print pass statistics
using TranslateFromMLIRFunction = std::function<mlir::LogicalResult(
    mlir::Operation *, llvm::raw_string_ostream &, const std::string &, bool,
    bool, bool)>;
using TranslateFromMLIRFunctionExtended = std::function<mlir::LogicalResult(
    mlir::Operation *, const std::string &, llvm::raw_string_ostream &,
    const std::string &, bool, bool, bool)>;

/// @brief Initialize MLIR with CUDA-Q dialects and create an internal MLIR
/// context
void initializeMLIR();

/// @brief Retrieve the context created by initializeMLIR()
mlir::MLIRContext *getMLIRContext();

/// @brief Create a new context and transfer the ownership. To be used to avoid
/// overcrowding the current MLIR context with temporary modules.
std::unique_ptr<mlir::MLIRContext> getOwningMLIRContext();

/// @brief Given an LLVM Module, set its target triple corresponding to the
/// current host machine.
bool setupTargetTriple(llvm::Module *);

/// @brief Run the LLVM PassManager.
void optimizeLLVM(llvm::Module *);

/// @brief Lower ModuleOp to a full QIR LLVMIR representation
/// and return an ExecutionEngine pointer for JIT function pointer
/// execution. Clients are responsible for deleting this pointer.
mlir::ExecutionEngine *createQIRJITEngine(mlir::ModuleOp &moduleOp,
                                          llvm::StringRef convertTo);

class Translation {
public:
  Translation() = default;
  Translation(TranslateFromMLIRFunction function, llvm::StringRef description)
      : function(std::move(function)), description(description) {}
  Translation(TranslateFromMLIRFunctionExtended f, llvm::StringRef description)
      : ext_function(std::move(f)), description(description) {}

  /// Return the description of this translation.
  llvm::StringRef getDescription() const { return description; }

  /// Invoke the translation function with the given input and output streams.
  mlir::LogicalResult operator()(mlir::Operation *op,
                                 llvm::raw_string_ostream &output,
                                 const std::string &additionalPasses,
                                 bool printIR, bool printIntermediateMLIR,
                                 bool printStats) const {
    if (function.has_value())
      return (*function)(op, output, additionalPasses, printIR,
                         printIntermediateMLIR, printStats);
    return mlir::failure();
  }

  /// Translation into QIR \e requires the use of the transport triple to
  /// specify: the profile, the version, and any profile options.
  mlir::LogicalResult operator()(mlir::Operation *op,
                                 const std::string &transport,
                                 llvm::raw_string_ostream &output,
                                 const std::string &additionalPasses,
                                 bool printIR, bool printIntermediateMLIR,
                                 bool printStats) const {
    if (ext_function.has_value())
      return (*ext_function)(op, transport, output, additionalPasses, printIR,
                             printIntermediateMLIR, printStats);
    return mlir::failure();
  }

private:
  /// The underlying translation function.
  std::optional<TranslateFromMLIRFunction> function;
  std::optional<TranslateFromMLIRFunctionExtended> ext_function;

  /// The description of the translation.
  llvm::StringRef description;
};

cudaq::Translation &getTranslation(llvm::StringRef name);

struct TranslateFromMLIRRegistration {
  TranslateFromMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const cudaq::TranslateFromMLIRFunction &function);
  TranslateFromMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const cudaq::TranslateFromMLIRFunctionExtended &function);
};

/// This is misnamed. This function returns the name of the .thunk unmarshalling
/// function.
std::optional<std::string>
getEntryPointName(mlir::OwningOpRef<mlir::ModuleOp> &module);

} // namespace cudaq
