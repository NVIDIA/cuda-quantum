/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "llvm/Support/Error.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <fmt/core.h>
#include <pybind11/stl.h>

#include "JITExecutionCache.h"
#include "utils/OpaqueArguments.h"

namespace py = pybind11;
using namespace mlir;

namespace cudaq {
static std::unique_ptr<JITExecutionCache> jitCache;

std::tuple<ExecutionEngine *, void *, std::size_t>
jitAndCreateArgs(const std::string &name, MlirModule module,
                 cudaq::OpaqueArguments &runtimeArgs,
                 const std::vector<std::string> &names, Type returnType) {
  auto mod = unwrap(module);
  auto cloned = mod.clone();
  auto context = cloned.getContext();

  // Have we JIT compiled this before?
  auto hash = llvm::hash_code{0};
  mod.walk([&hash](Operation *op) {
    hash = llvm::hash_combine(hash, OperationEquivalence::computeHash(op));
  });
  auto hashKey = static_cast<size_t>(hash);

  ExecutionEngine *jit = nullptr;
  if (jitCache->hasJITEngine(hashKey))
    jit = jitCache->getJITEngine(hashKey);
  else {

    PassManager pm(context);
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createPySynthCallableBlockArgs(names));
    pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader(/*genAsQuake=*/true));
    pm.addPass(cudaq::opt::createGenerateKernelExecution());
    pm.addPass(cudaq::opt::createLambdaLiftingPass());
    cudaq::opt::addPipelineToQIR<>(pm);
    if (failed(pm.run(cloned)))
      throw std::runtime_error(
          "cudaq::builder failed to JIT compile the Quake representation.");

    // The "fast" instruction selection compilation algorithm is actually very
    // slow for large quantum circuits. Disable that here. Revisit this
    // decision by testing large UCCSD circuits if jitCodeGenOptLevel is changed
    // in the future. Also note that llvm::TargetMachine::setFastIsel() and
    // setO0WantsFastISel() do not retain their values in our current version of
    // LLVM. This use of LLVM command line parameters could be changed if the
    // LLVM JIT ever supports the TargetMachine options in the future.
    const char *argv[] = {"", "-fast-isel=0", nullptr};
    llvm::cl::ParseCommandLineOptions(2, argv);

    ExecutionEngineOptions opts;
    opts.enableGDBNotificationListener = false;
    opts.enablePerfNotificationListener = false;
    opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
    opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
    SmallVector<StringRef, 4> sharedLibs;
    opts.llvmModuleBuilder =
        [](Operation *module,
           llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
      llvmContext.setOpaquePointers(false);
      auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
      if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return nullptr;
      }
      ExecutionEngine::setupTargetTriple(llvmModule.get());
      return llvmModule;
    };

    auto jitOrError = ExecutionEngine::create(cloned, opts);
    assert(!!jitOrError);

    auto uniqueJit = std::move(jitOrError.get());
    jit = uniqueJit.release();
    jitCache->cache(hashKey, jit);
  }

  // We need to append the return type to the OpaqueArguments here
  // so that we get a spot in the `rawArgs` memory for the
  // altLaunchKernel function to dump the result
  if (!isa<NoneType>(returnType)) {
    if (returnType.isInteger(64)) {
      py::args returnVal = py::make_tuple(py::int_(0));
      packArgs(runtimeArgs, returnVal);
    } else if (returnType.isInteger(1)) {
      py::args returnVal = py::make_tuple(py::bool_(0));
      packArgs(runtimeArgs, returnVal);
    } else if (isa<FloatType>(returnType)) {
      py::args returnVal = py::make_tuple(py::float_(0.0));
      packArgs(runtimeArgs, returnVal);
    } else {
      std::string msg;
      {
        llvm::raw_string_ostream os(msg);
        returnType.print(os);
      }
      throw std::runtime_error(
          "Unsupported CUDA Quantum kernel return type - " + msg + ".\n");
    }
  }

  void *rawArgs = nullptr;
  std::size_t size = 0;
  if (runtimeArgs.size()) {
    auto expectedPtr = jit->lookup(name + ".argsCreator");
    if (!expectedPtr) {
      throw std::runtime_error(
          "cudaq::builder failed to get argsCreator function.");
    }
    auto argsCreator =
        reinterpret_cast<std::size_t (*)(void **, void **)>(*expectedPtr);
    rawArgs = nullptr;
    size = argsCreator(runtimeArgs.data(), &rawArgs);
  }
  return std::make_tuple(jit, rawArgs, size);
}

std::tuple<void *, std::size_t>
pyAltLaunchKernelBase(const std::string &name, MlirModule module,
                      Type returnType, cudaq::OpaqueArguments &runtimeArgs,
                      const std::vector<std::string> &names) {
  auto [jit, rawArgs, size] =
      jitAndCreateArgs(name, module, runtimeArgs, names, returnType);

  auto mod = unwrap(module);
  auto thunkName = name + ".thunk";
  auto thunkPtr = jit->lookup(thunkName);
  if (!thunkPtr)
    throw std::runtime_error("cudaq::builder failed to get thunk function");

  auto thunk = reinterpret_cast<void (*)(void *)>(*thunkPtr);

  std::string properName = name;

  // Need to first invoke the init_func()
  auto kernelInitFunc = properName + ".init_func";
  auto initFuncPtr = jit->lookup(kernelInitFunc);
  if (!initFuncPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get kernelReg function.");
  }
  auto kernelInit = reinterpret_cast<void (*)()>(*initFuncPtr);
  kernelInit();

  // Need to first invoke the kernelRegFunc()
  auto kernelRegFunc = properName + ".kernelRegFunc";
  auto regFuncPtr = jit->lookup(kernelRegFunc);
  if (!regFuncPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get kernelReg function.");
  }
  auto kernelReg = reinterpret_cast<void (*)()>(*regFuncPtr);
  kernelReg();

  auto &platform = cudaq::get_platform();
  if (platform.is_remote() || platform.is_emulated()) {
    struct ArgWrapper {
      ModuleOp mod;
      std::vector<std::string> callableNames;
      void *rawArgs = nullptr;
    };
    auto *wrapper = new ArgWrapper{mod, names, rawArgs};
    cudaq::altLaunchKernel(name.c_str(), thunk,
                           reinterpret_cast<void *>(wrapper), size, 0);
    delete wrapper;
  } else
    cudaq::altLaunchKernel(name.c_str(), thunk, rawArgs, size, 0);

  return std::make_tuple(rawArgs, size);
}

void pyAltLaunchKernel(const std::string &name, MlirModule module,
                       cudaq::OpaqueArguments &runtimeArgs,
                       const std::vector<std::string> &names) {
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());
  auto [rawArgs, size] =
      pyAltLaunchKernelBase(name, module, noneType, runtimeArgs, names);
  std::free(rawArgs);
}

py::object pyAltLaunchKernelR(const std::string &name, MlirModule module,
                              MlirType returnType,
                              cudaq::OpaqueArguments &runtimeArgs,
                              const std::vector<std::string> &names) {
  auto [rawArgs, size] = pyAltLaunchKernelBase(name, module, unwrap(returnType),
                                               runtimeArgs, names);
  auto unwrapped = unwrap(returnType);
  if (unwrapped.isInteger(64)) {
    std::size_t concrete;
    // Here we know the return type should be at
    // the last 8 bytes of memory
    // FIXME revisit this calculation when we support returning vectors
    std::memcpy(&concrete, ((char *)rawArgs) + size - 8, 8);
    std::free(rawArgs);
    return py::int_(concrete);
  } else if (unwrapped.isInteger(1)) {
    bool concrete = false;
    std::memcpy(&concrete, ((char *)rawArgs) + size - 1, 1);
    return py::bool_(concrete);
  } else if (isa<FloatType>(unwrapped)) {
    double concrete;
    std::memcpy(&concrete, ((char *)rawArgs) + size - 8, 8);
    std::free(rawArgs);
    return py::float_(concrete);
  }

  std::free(rawArgs);
  unwrapped.dump();
  throw std::runtime_error("Invalid return type for pyAltLaunchKernel.");
}

MlirModule synthesizeKernel(const std::string &name, MlirModule module,
                            cudaq::OpaqueArguments &runtimeArgs) {
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());

  auto [jit, rawArgs, size] =
      jitAndCreateArgs(name, module, runtimeArgs, {}, noneType);
  auto cloned = unwrap(module).clone();
  auto context = cloned.getContext();
  registerLLVMDialectTranslation(*context);

  PassManager pm(context);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createQuakeSynthesizer(name, rawArgs));
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLoopNormalize());
  pm.addPass(cudaq::opt::createLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");
  std::free(rawArgs);
  return wrap(cloned);
}

std::string getQIRLL(const std::string &name, MlirModule module,
                     cudaq::OpaqueArguments &runtimeArgs,
                     std::string &profile) {
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());

  auto [jit, rawArgs, size] =
      jitAndCreateArgs(name, module, runtimeArgs, {}, noneType);
  auto cloned = unwrap(module).clone();
  auto context = cloned.getContext();

  PassManager pm(context);
  if (profile.empty())
    cudaq::opt::addPipelineToQIR<>(pm);
  else
    cudaq::opt::addPipelineToQIR<true>(pm, profile);
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");
  std::free(rawArgs);

  llvm::LLVMContext llvmContext;
  llvmContext.setOpaquePointers(false);
  auto llvmModule = translateModuleToLLVMIR(cloned, llvmContext);
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get()))
    throw std::runtime_error("Failed to optimize LLVM IR ");

  std::string str;
  {
    llvm::raw_string_ostream os(str);
    llvmModule->print(os, nullptr);
  }
  return str;
}

void bindAltLaunchKernel(py::module &mod) {
  jitCache = std::make_unique<JITExecutionCache>();

  auto callableArgHandler = [](cudaq::OpaqueArguments &argData,
                               py::object &arg) {
    if (py::hasattr(arg, "module")) {
      // Just give it some dummy data that will not be used.
      // We synthesize away all callables, the block argument
      // remains but it is not used, so just give argsCreator
      // something, and we'll make sure its cleaned up.
      long *ourAllocatedArg = new long();
      argData.emplace_back(ourAllocatedArg,
                           [](void *ptr) { delete static_cast<long *>(ptr); });
      return true;
    }
    return false;
  };

  mod.def(
      "pyAltLaunchKernel",
      [&](const std::string &kernelName, MlirModule module,
          py::args runtimeArgs, std::vector<std::string> callable_names) {
        cudaq::OpaqueArguments args;
        cudaq::packArgs(args, runtimeArgs, callableArgHandler);
        pyAltLaunchKernel(kernelName, module, args, callable_names);
      },
      py::arg("kernelName"), py::arg("module"), py::kw_only(),
      py::arg("callable_names") = std::vector<std::string>{}, "DOC STRING");
  mod.def(
      "pyAltLaunchKernelR",
      [&](const std::string &kernelName, MlirModule module, MlirType returnType,
          py::args runtimeArgs, std::vector<std::string> callable_names) {
        cudaq::OpaqueArguments args;
        cudaq::packArgs(args, runtimeArgs, callableArgHandler);
        return pyAltLaunchKernelR(kernelName, module, returnType, args,
                                  callable_names);
      },
      py::arg("kernelName"), py::arg("module"), py::arg("returnType"),
      py::kw_only(), py::arg("callable_names") = std::vector<std::string>{},
      "DOC STRING");

  mod.def("synthesize", [](py::object kernel, py::args runtimeArgs) {
    MlirModule module = kernel.attr("module").cast<MlirModule>();
    auto name = kernel.attr("name").cast<std::string>();
    cudaq::OpaqueArguments args;
    cudaq::packArgs(args, runtimeArgs);
    return synthesizeKernel(name, module, args);
  });

  mod.def(
      "get_qir",
      [](py::object kernel, py::args runtimeArgs, std::string profile) {
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        MlirModule module = kernel.attr("module").cast<MlirModule>();
        auto name = kernel.attr("name").cast<std::string>();
        cudaq::OpaqueArguments args;
        cudaq::packArgs(args, runtimeArgs);
        return getQIRLL(name, module, args, profile);
      },
      py::arg("kernel"), py::kw_only(), py::arg("profile") = "");
}
} // namespace cudaq