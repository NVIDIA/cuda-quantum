/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/RecordLogParser.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/algorithms/run.h"
#include "cudaq/simulators.h"
#include "nvqir/CircuitSimulator.h"
#include "llvm/IR/DataLayout.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include <mlir/IR/BuiltinOps.h>
#include <string>
#include <utility>
#include <vector>

namespace cudaq {
namespace details {
/// Extracts data layout information from MLIR modules
class LayoutExtractor {
public:
  static std::pair<std::size_t, std::vector<std::size_t>>
  extractLayout(const std::string &, const std::string &);

private:
  static mlir::MLIRContext *createContext();
};
} // namespace details
} // namespace cudaq

using namespace llvm;
using namespace mlir;

mlir::MLIRContext *cudaq::details::LayoutExtractor::createContext() {
  mlir::DialectRegistry registry;
  cudaq::opt::registerCodeGenDialect(registry);
  cudaq::registerAllDialects(registry);
  auto context = new mlir::MLIRContext(registry);
  context->loadAllAvailableDialects();
  mlir::registerLLVMDialectTranslation(*context);
  return context;
}

std::pair<std::size_t, std::vector<std::size_t>>
cudaq::details::LayoutExtractor::extractLayout(const std::string &kernelName,
                                               const std::string &quakeCode) {
  std::unique_ptr<mlir::MLIRContext> mlirContext(createContext());
  auto m_module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(quakeCode), mlirContext.get());
  if (!m_module)
    throw std::runtime_error("module cannot be parsed");
  mlir::func::FuncOp kernelFunc = m_module->lookupSymbol<func::FuncOp>(
      cudaq::runtime::cudaqGenPrefixName + kernelName);
  if (!kernelFunc)
    throw std::runtime_error("Could not find " + kernelName +
                             " function in the module.");
  // Extract layout information from the function's return type
  std::size_t totalSize = 0;
  std::vector<std::size_t> fieldOffsets;
  if (kernelFunc.getNumResults() > 0) {
    mlir::Type returnType = kernelFunc.getResultTypes()[0];
    auto mod = kernelFunc->getParentOfType<ModuleOp>();
    StringRef dataLayoutSpec = "";
    if (auto attr = mod->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
      dataLayoutSpec = cast<StringAttr>(attr);
    auto dataLayout = llvm::DataLayout(dataLayoutSpec);
    cudaq::info("Data Layout: {}", dataLayout.getStringRepresentation());
    llvm::LLVMContext context;
    LLVMTypeConverter converter(kernelFunc.getContext());
    cudaq::opt::initializeTypeConversions(converter);
    // Handle structure types
    if (auto structType = mlir::dyn_cast<cudaq::cc::StructType>(returnType)) {
      auto llvmDialectTy = converter.convertType(structType);
      LLVM::TypeToLLVMIRTranslator translator(context);
      auto *llvmStructTy =
          cast<llvm::StructType>(translator.translateType(llvmDialectTy));
      auto *layout = dataLayout.getStructLayout(llvmStructTy);
      totalSize = layout->getSizeInBytes();
      std::size_t numElements = structType.getMembers().size();
      for (std::size_t i = 0; i < numElements; ++i)
        fieldOffsets.emplace_back(layout->getElementOffset(i));
    } else
      totalSize = cudaq::opt::getDataSize(dataLayout, returnType);
  }
  return {totalSize, fieldOffsets};
}

cudaq::details::RunResultSpan cudaq::details::runTheKernel(
    std::function<void()> &&kernel, quantum_platform &platform,
    const std::string &kernel_name, std::size_t shots) {
  ScopedTraceWithContext(cudaq::TIMING_RUN, "runTheKernel");
  // 1. Clear the outputLog.
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();

  circuitSimulator->outputLog.clear();

  // 2. Launch the kernel on the QPU.
  if (platform.get_remote_capabilities().isRemoteSimulator ||
      platform.is_emulated() || platform.is_remote()) {
    // In a remote simulator execution/hardware emulation environment, set the
    // `run` context name and number of iterations (shots)
    auto ctx = std::make_unique<cudaq::ExecutionContext>("run", shots);
    platform.set_exec_ctx(ctx.get());
    // Launch the kernel a single time to post the 'run' request to the remote
    // server or emulation executor.
    kernel();
    platform.reset_exec_ctx();
    // Retrieve the result output log.
    // FIXME: this currently assumes all the shots are good.
    std::string remoteOutputLog(ctx->invocationResultBuffer.begin(),
                                ctx->invocationResultBuffer.end());
    circuitSimulator->outputLog.swap(remoteOutputLog);
  } else {
    auto ctx = std::make_unique<cudaq::ExecutionContext>("run", 1);
    for (std::size_t i = 0; i < shots; ++i) {
      // Set the execution context since as noise model is attached to this
      // context.
      platform.set_exec_ctx(ctx.get());
      kernel();
      // Reset the context to flush qubit deallocation.
      platform.reset_exec_ctx();
    }
  }

  // 3a. Get the data layout information
  std::pair<std::size_t, std::vector<std::size_t>> layoutInfo = {0, {}};
  auto quakeCode =
      cudaq::get_quake_by_name(kernel_name, /*throwException=*/false);
  if (!quakeCode.empty())
    layoutInfo =
        cudaq::details::LayoutExtractor::extractLayout(kernel_name, quakeCode);

  // 3b. Pass the outputLog to the parser (target-specific?)
  cudaq::RecordLogParser parser(layoutInfo);
  parser.parse(circuitSimulator->outputLog);

  // 4. Get the buffer and length of buffer (in bytes) from the parser.
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);

  // 5. Clear the outputLog (?)
  circuitSimulator->outputLog.clear();

  // 6. Pass the span back as a RunResultSpan. NB: it is the responsibility of
  // the caller to free the buffer.
  return {buffer, bufferSize};
}
