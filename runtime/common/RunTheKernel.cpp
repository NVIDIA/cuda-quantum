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
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/InitAllDialects.h"
#include "cudaq/algorithms/run.h"
#include "cudaq/simulators.h"
#include "nvqir/CircuitSimulator.h"
#include "llvm/IR/DataLayout.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

static std::once_flag initializeContextOnce;

// Create a scratch context for parsing a kernel's MLIR content and cache it.
static MLIRContext *scratchContext;

static void initializeScratchContext() {
  DialectRegistry registry;
  cudaq::registerAllDialects(registry);
  scratchContext = new MLIRContext(registry);
  scratchContext->loadAllAvailableDialects();
}

static std::pair<std::size_t, std::vector<std::size_t>>
extractLayout(const std::string &kernelName, const std::string &quakeCode) {
  if (!scratchContext)
    throw std::runtime_error("must have a scratch context");
  auto m_module =
      parseSourceString<ModuleOp>(StringRef(quakeCode), scratchContext);
  if (!m_module)
    throw std::runtime_error("module cannot be parsed");
  func::FuncOp kernelFunc = m_module->lookupSymbol<func::FuncOp>(
      cudaq::runtime::cudaqGenPrefixName + kernelName);
  if (!kernelFunc)
    throw std::runtime_error("Could not find " + kernelName +
                             " function in the module.");
  if (kernelFunc.getNumResults() == 0)
    return {0, {}};

  // Extract layout information from the function's return type.
  Type returnTy = kernelFunc.getResultTypes()[0];
  auto mod = kernelFunc->getParentOfType<ModuleOp>();
  auto attr = mod->getAttr(cudaq::opt::factory::targetDataLayoutAttrName);
  if (!attr)
    throw std::runtime_error("module is malformed. missing data layout.");
  StringRef dataLayoutSpec = cast<StringAttr>(attr);
  auto dataLayout = llvm::DataLayout(dataLayoutSpec);
  CUDAQ_INFO("Data Layout: {}", dataLayout.getStringRepresentation());
  llvm::LLVMContext context;
  LLVMTypeConverter converter(kernelFunc.getContext());
  cudaq::opt::initializeTypeConversions(converter);
  auto structTy = dyn_cast<cudaq::cc::StructType>(returnTy);
  if (!structTy) {
    std::size_t totalSize = cudaq::opt::getDataSize(dataLayout, returnTy);
    if (totalSize == 0)
      throw std::runtime_error("size of result must not be 0.");
    return {totalSize, {}};
  }

  // Handle structure types
  auto llvmDialectTy = converter.convertType(structTy);
  LLVM::TypeToLLVMIRTranslator translator(context);
  auto *llvmStructTy =
      cast<llvm::StructType>(translator.translateType(llvmDialectTy));
  auto *layout = dataLayout.getStructLayout(llvmStructTy);
  std::size_t totalSize = layout->getSizeInBytes();
  std::size_t numElements = structTy.getMembers().size();
  std::vector<std::size_t> fieldOffsets;
  for (std::size_t i = 0; i < numElements; ++i)
    fieldOffsets.emplace_back(layout->getElementOffset(i));
  return {totalSize, fieldOffsets};
}

cudaq::details::RunResultSpan cudaq::details::runTheKernel(
    std::function<void()> &&kernel, quantum_platform &platform,
    const std::string &kernel_name, const std::string &original_name,
    std::size_t shots, std::size_t qpu_id) {
  ScopedTraceWithContext(cudaq::TIMING_RUN, "runTheKernel");
  // 1. Clear the outputLog.
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  circuitSimulator->outputLog.clear();

  std::call_once(initializeContextOnce, []() { initializeScratchContext(); });

  // Some platforms do not support run yet, emit error.
  if (!platform.get_codegen_config().outputLog)
    throw std::runtime_error("`run` is not yet supported on this target.");

  // 2. Launch the kernel on the QPU.
  if (platform.is_remote() || platform.is_emulated() ||
      platform.get_remote_capabilities().isRemoteSimulator) {
    // In a remote simulator execution or hardware emulation environment, set
    // the `run` context name and number of iterations (shots)
    auto ctx = std::make_unique<cudaq::ExecutionContext>("run", shots, qpu_id);
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
    auto ctx = std::make_unique<cudaq::ExecutionContext>("run", 1, qpu_id);
    for (std::size_t i = 0; i < shots; ++i) {
      // Set the execution context since as noise model is attached to this
      // context.
      platform.set_exec_ctx(ctx.get());
      kernel();
      // Reset the context to flush qubit deallocation.
      platform.reset_exec_ctx();
    }
  }

  // 3a. Get the data layout information. Use the original kernel, since it has
  // the information while the kernel being called dropped it on the floor.
  std::pair<std::size_t, std::vector<std::size_t>> layoutInfo = {0, {}};
  auto quakeCode =
      cudaq::get_quake_by_name(original_name, /*throwException=*/false);
  if (!quakeCode.empty())
    layoutInfo = extractLayout(original_name, quakeCode);

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
