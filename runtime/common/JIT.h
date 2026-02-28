/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "CompiledKernel.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace llvm {
class StringRef;
namespace orc {
class LLJIT;
}
} // namespace llvm

namespace mlir {
class ExecutionEngine;
class ModuleOp;
} // namespace mlir

namespace cudaq {

/// Util to create a wrapped kernel defined by LLVM IR with serialized
/// arguments.
// Note: We don't use `mlir::ExecutionEngine` to skip unnecessary
// `packFunctionArguments` (slow for raw LLVM IR containing many functions from
// included headers).
std::tuple<std::unique_ptr<llvm::orc::LLJIT>, std::function<void()>>
createWrappedKernel(std::string_view llvmIr, const std::string &kernelName,
                    void *args, std::uint64_t argsSize);

/// JitEngine is a type-erased class that is wrapping an mlir::ExecutionEngine
/// without introducing any link time dependency on MLIR for the client of the
/// class. Memory management for of the mlir::ExecutionEngine is handled
/// internally.
class JitEngine {
public:
  JitEngine(std::unique_ptr<mlir::ExecutionEngine>);
  void run(const std::string &kernelName) const;
  void (*lookupRawNameOrFail(const std::string &kernelName) const)();
  std::size_t getKey() const;

private:
  class Impl;
  std::shared_ptr<Impl> impl;
};

/// Lower ModuleOp to QIR/LLVM IR and create a JIT execution engine.
JitEngine createQIRJITEngine(mlir::ModuleOp &moduleOp,
                             llvm::StringRef convertTo);

CompiledKernel createCompiledKernel(JitEngine engine, std::string kernelName,
                                    bool hasResult);

} // namespace cudaq
