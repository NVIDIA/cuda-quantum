/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "KernelJIT.h"
#include "NvidiaPlatformHelper.h"
#include "TargetBackend.h"
#include "common/Logger.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nvqpp_config.h"
#include "rpc/server.h"
#include "rpc/this_handler.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include <charconv>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace cudaq {
/// Flag that stops the server and exits the qpud process
static std::atomic<bool> _stopServer = false;

/// Storage for loaded Kernels
static std::unordered_map<std::string, Kernel> loadedThunkSymbols;

/// Storage for the JIT engine used for each kernel
std::unordered_map<std::string, std::unique_ptr<KernelJIT>> jitStorage;

/// Pointer to the targeted QPUD backend
std::unique_ptr<TargetBackend> backend = nullptr;

/// Pointer to the global MLIR context
std::unique_ptr<mlir::MLIRContext> mlirContext;

/// @brief Respond to the client with an error and
/// return from the calling function.
template <typename RetType>
RetType returnWithError(const std::string &errorMsg) {
  rpc::this_handler().respond_error(errorMsg);
  if constexpr (std::is_same_v<void, RetType>)
    return;
  else
    return RetType{};
}

/// @brief Utility function that will let us wrap our
/// backend invocations in a try-catch to better report errors to the client.
template <typename Functor, typename R = typename std::invoke_result_t<Functor>>
auto backendInvokeHandleErrors(Functor &&functor, const std::string &errorMsg) {
  try {
    return functor();
  } catch (std::exception &e) {
    auto msg = errorMsg + " " + std::string(e.what());
    return returnWithError<R>(msg);
  }
}

/// @brief Stop the server.
void stopServer() { _stopServer = true; }

/// @brief Reset the backend to the given target backend
/// @param backend
void setTargetBackend(const std::string &backend) {
  cudaq::info("Setting qpud backend to {}", backend);
  std::string mutableName = backend, subBackend = "";
  auto split = cudaq::split(backend, ':');
  if (split.size() > 1) {
    mutableName = split[0];
    subBackend = split[1];
  }

  // Set the backend, check that it is valid
  cudaq::backend = cudaq::registry::get<cudaq::TargetBackend>(mutableName);
  if (!cudaq::backend)
    returnWithError<void>("Invalid target backend. (" + backend + ")");

  // Set the sub backend if we have one
  if (!subBackend.empty())
    cudaq::backend->setSpecificBackend(subBackend);

  return;
}

bool getIsSimulator() { return cudaq::backend->isSimulator(); }
bool getSupportsConditionalFeedback() {
  return cudaq::backend->supportsConditionalFeedback();
}

/// @brief If it has not been loaded, JIT the provided quakeCode to LLVM.
/// @param kernelName
/// @param quakeCode
void loadQuakeCode(const std::string &kernelName, const std::string &quakeCode,
                   const std::vector<std::string> &extraLibraries) {
  cudaq::ScopedTrace trace("qpud::loadQuakeCode", kernelName, extraLibraries);

  if (!cudaq::backend->isInitialized())
    cudaq::backend->initialize();

  // Will need to JIT the quakeCode to LLVM only
  // Load as MLIR Module, run the PassManager to lower to LLVM Dialect
  // Translate to LLVM Module and use MLIR ExecutionEngine
  // add to loadedThunkSymbols
  if (!loadedThunkSymbols.count(kernelName)) {
    // Get the LLVM Module from the backend compile phase
    // Default will lower quake to QIR llvm, others may
    // lower to the QIR base profile.
    auto llvmModule = cudaq::backend->compile(*mlirContext.get(), quakeCode);
    if (!llvmModule)
      returnWithError<void>(
          "[qpud::loadQuake] Failed to lower quake code to LLVM IR: " +
          kernelName);

    std::string qirCode;
    llvm::raw_string_ostream os(qirCode);
    llvmModule->print(os, nullptr);
    os.flush();
    // Create and store the KernelJIT instance, get pointer to it
    auto result = jitStorage.insert(
        {kernelName, cantFail(KernelJIT::Create(extraLibraries))});
    auto kernelJIT = result.first->second.get();

    // Add the LLVM Module, Get the KERNEL.thunk function pointer
    cantFail(kernelJIT->addModule(std::move(llvmModule)),
             "Could not load the llvm::Module for thunk JIT.");

    // Ensure we have the thunk symbol
    std::string symbolName = kernelName + ".thunk";
    if (quakeCode.find(symbolName) == std::string::npos) {
      return returnWithError<void>(
          symbolName +
          " symbol not available. Please compile with --enable-mlir.");
    }

    // Apple for some reason prepends a "_"
#if defined(__APPLE__) && defined(__MACH__)
    symbolName = "_" + symbolName;
#endif

    // Get the thunk symbol
    auto symbol =
        cantFail(kernelJIT->lookup(symbolName), "Could not find the symbol");
    auto *thunkFunctor =
        reinterpret_cast<DynamicResult (*)(void *, bool)>(symbol.getAddress());

    // Store the thunk pointer.
    loadedThunkSymbols.insert(
        {kernelName, Kernel(thunkFunctor, kernelName, qirCode, quakeCode)});
  }
}

/// @brief Direct the server to execute the kernel with given name and provided
/// runtime arguments.
/// @param kernelName name of the kernel to execute
/// @param args vector<uint8_t> representation of the void* kernelArgs.
/// @return The modified kernelArgs (the return value is in there)
std::vector<uint8_t> executeKernel(const std::string &kernelName,
                                   std::vector<uint8_t> args) {
  cudaq::ScopedTrace trace("qpud::executeKernel", kernelName);

  if (!cudaq::backend->isInitialized())
    cudaq::backend->initialize();

  auto f_iter = loadedThunkSymbols.find(kernelName);
  if (f_iter == loadedThunkSymbols.end())
    returnWithError<std::vector<uint8_t>>(
        "[qpud::base_exec] Invalid CUDA Quantum kernel name: " + kernelName);

  auto function = f_iter->second;
  auto raw_args = static_cast<void *>(args.data());

  return backendInvokeHandleErrors(
      [&]() -> std::vector<uint8_t> {
        auto res = cudaq::backend->baseExecute(function, raw_args,
                                               /*isClientServer=*/true);
        if (!res.ptr)
          return args;
        return {&res.ptr[0], &res.ptr[res.len]};
      },
      "Error in base execute.");
}

/// @brief Sample the state generated by the kernel with given name
/// @param kernelName name of the kernel to execute
/// @param shots
/// @param args vector<uint8_t> representation of the void* kernelArgs.
/// @return The modified kernelArgs (the return value is in there)
std::vector<std::size_t> sampleKernel(const std::string &kernelName,
                                      std::size_t shots,
                                      std::vector<uint8_t> args) {
  cudaq::ScopedTrace trace("qpud::sampleKernel", kernelName, shots);

  if (!cudaq::backend->isInitialized())
    cudaq::backend->initialize();

  auto f_iter = loadedThunkSymbols.find(kernelName);
  if (f_iter == loadedThunkSymbols.end())
    returnWithError<std::vector<std::size_t>>(
        "[qpud::sample] Invalid CUDA Quantum kernel name: " + kernelName);

  auto function = f_iter->second;
  auto raw_args = static_cast<void *>(args.data());
  return backendInvokeHandleErrors(
      [&]() { return backend->sample(function, shots, raw_args); },
      "Error in sample.");
}

/// @brief Observe the state generated by the kernel with the given spin
/// operator
/// @param kernelName name of the kernel to execute
/// @param spin_op_data The operator representation <kernel|H|kernel, H.
/// @param args vector<uint8_t> representation of the void* kernelArgs.
/// @return
std::tuple<double, std::vector<std::size_t>>
observeKernel(const std::string &kernelName, std::vector<double> spin_op_data,
              const std::size_t shots, std::vector<uint8_t> &args) {
  cudaq::ScopedTrace trace("qpud::observeKernel", kernelName, shots);
  if (!cudaq::backend->isInitialized())
    cudaq::backend->initialize();

  auto f_iter = loadedThunkSymbols.find(kernelName);
  if (f_iter == loadedThunkSymbols.end())
    returnWithError<double>("[qpud::observe] Invalid CUDA Quantum kernel name: " +
                            kernelName);

  auto function = f_iter->second;
  auto raw_args = static_cast<void *>(args.data());
  return backendInvokeHandleErrors(
      [&]() {
        return backend->observe(function, spin_op_data, shots, raw_args);
      },
      "Error in observe.");
}

/// @brief Observe the state generated by the kernel with the given spin
/// operator, but immediately return with the Job ID information.
/// @param kernelName name of the kernel to execute
/// @param spin_op_data The operator representation <kernel|H|kernel, H.
/// @param args vector<uint8_t> representation of the void* kernelArgs.
/// @return
std::tuple<std::vector<std::string>, std::vector<std::string>>
observeKernelDetach(const std::string &kernelName,
                    std::vector<double> spin_op_data, const std::size_t shots,
                    std::vector<uint8_t> &args) {
  cudaq::ScopedTrace trace("qpud::observeKernelDetach", kernelName, shots);

  if (!cudaq::backend->isInitialized())
    backend->initialize();

  auto f_iter = loadedThunkSymbols.find(kernelName);
  if (f_iter == loadedThunkSymbols.end())
    returnWithError<double>("[qpud::observe] Invalid CUDA Quantum kernel name: " +
                            kernelName);

  auto function = f_iter->second;
  auto raw_args = static_cast<void *>(args.data());
  return backendInvokeHandleErrors(
      [&]() {
        return backend->observeDetach(function, spin_op_data, shots, raw_args);
      },
      "Error in detached observe.");
}

/// @brief Produce the result from a detached observe job
/// @param jobId
/// @return
std::tuple<double, std::vector<std::size_t>>
observeKernelFromJobId(const std::string &jobId) {
  cudaq::ScopedTrace trace("qpud::observeKernelFromJobId", jobId);

  if (!cudaq::backend->isInitialized())
    backend->initialize();
  return backendInvokeHandleErrors(
      [&]() { return backend->observeFromJobId(jobId); },
      "Error in observe from Job ID.");
}

/// @brief Sample the given kernel asynchronously, detach and return the job
/// information
std::tuple<std::string, std::string>
sampleKernelDetach(const std::string &kernelName, const std::size_t shots,
                   std::vector<uint8_t> &args) {
  cudaq::ScopedTrace trace("qpud::sampleKernelDetach", kernelName, shots);
  if (!cudaq::backend->isInitialized())
    backend->initialize();

  auto f_iter = loadedThunkSymbols.find(kernelName);
  if (f_iter == loadedThunkSymbols.end())
    returnWithError<double>("[qpud::sampleDetach] Invalid CUDA Quantum kernel name: " +
                            kernelName);

  auto function = f_iter->second;
  auto raw_args = static_cast<void *>(args.data());
  return backendInvokeHandleErrors(
      [&]() { return backend->sampleDetach(function, shots, raw_args); },
      "Error in detached sample.");
}

/// @brief Produce the result from a detached observe job
/// @param jobId
/// @return
std::vector<std::size_t> sampleKernelFromJobId(const std::string &jobId) {
  cudaq::ScopedTrace trace("qpud::sampleKernelFromJobId", jobId);

  if (!cudaq::backend->isInitialized())
    backend->initialize();
  return backendInvokeHandleErrors(
      [&]() { return backend->sampleFromJobId(jobId); },
      "Error in sample from Job ID.");
}

} // namespace cudaq

int main(int argc, char **argv) {
  int qpu_id = 0, port = 8888;
  std::vector<std::string> args(&argv[0], &argv[0] + argc);
  for (std::size_t i = 0; i < args.size(); i++) {
    if (args[i] == "--qpu") {
      if (i == args.size() - 1) {
        llvm::errs() << "--qpu specified but no qpu id provided.\n";
        return -1;
      }
      std::string arg = args[i + 1];
      auto [ptr, ec] =
          std::from_chars(arg.data(), arg.data() + arg.size(), qpu_id);
      if (ec == std::errc::invalid_argument) {
        llvm::errs() << "[qpud] Invalid QPU ID (" << arg
                     << "). Provide an integer [0,N_QPUS).\n";
        return -1;
      }
    }

    if (args[i] == "--port") {
      if (i == args.size() - 1) {
        llvm::errs() << "--port specified but no port provided.\n";
        return -1;
      }
      std::string arg = args[i + 1];
      auto [ptr, ec] =
          std::from_chars(arg.data(), arg.data() + arg.size(), port);
      if (ec == std::errc::invalid_argument) {
        llvm::errs() << "[qpud] Invalid Port (" << arg << ").\n";
        return -1;
      }
    }
  }

  // One time initialization of LLVM
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerAllPasses();

  // One time initialization of MLIR
  mlir::DialectRegistry registry;
  registry.insert<cudaq::cc::CCDialect, quake::QuakeDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                  mlir::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect>();
  cudaq::mlirContext = std::make_unique<mlir::MLIRContext>(registry);
  cudaq::mlirContext->loadAllAvailableDialects();
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*cudaq::mlirContext.get());

  // Create the Default Target Backend.
  cudaq::backend = cudaq::registry::get<cudaq::TargetBackend>("default");

  // Create the server and bind the functions
  std::unique_ptr<rpc::server> server;
  try {
    server = std::make_unique<rpc::server>(port);
    server->bind("loadQuakeCode", &cudaq::loadQuakeCode);
    server->bind("executeKernel", &cudaq::executeKernel);
    server->bind("sampleKernel", &cudaq::sampleKernel);
    server->bind("sampleKernelDetach", &cudaq::sampleKernelDetach);
    server->bind("sampleKernelFromJobId", &cudaq::sampleKernelFromJobId);
    server->bind("observeKernel", &cudaq::observeKernel);
    server->bind("observeKernelFromJobId", &cudaq::observeKernelFromJobId);
    server->bind("observeKernelDetach", &::cudaq::observeKernelDetach);
    server->bind("setTargetBackend", &cudaq::setTargetBackend);
    server->bind("getIsSimulator", &cudaq::getIsSimulator);
    server->bind("getSupportsConditionalFeedback",
                 &cudaq::getSupportsConditionalFeedback);
    server->bind("stopServer", &cudaq::stopServer);

    cudaq::NvidiaPlatformHelper helper;
    helper.setQPU(qpu_id);

    server->async_run();
  } catch (std::exception &e) {
    printf("%s\n", e.what());
    return -1;
  }

  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (cudaq::_stopServer)
      break;
  }
  return 0;
}
