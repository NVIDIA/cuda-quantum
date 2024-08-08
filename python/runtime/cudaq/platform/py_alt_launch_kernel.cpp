/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "JITExecutionCache.h"
#include "common/ArgumentWrapper.h"
#include "common/Environment.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "utils/OpaqueArguments.h"
#include "utils/PyTypes.h"

#include "llvm/Support/Error.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <fmt/core.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir;

namespace cudaq {
// TODO: unify with the definition in GenKernelExec.cpp
static constexpr std::int32_t NoResultOffset =
    std::numeric_limits<std::int32_t>::max();
static std::unique_ptr<JITExecutionCache> jitCache;

struct PyStateVectorData {
  void *data = nullptr;
  simulation_precision precision = simulation_precision::fp32;
  std::string kernelName;
};
using PyStateVectorStorage = std::map<std::string, PyStateVectorData>;

struct PyStateData {
  cudaq::state data;
  std::string kernelName;
};
using PyStateStorage = std::map<std::string, PyStateData>;

static std::unique_ptr<PyStateVectorStorage> stateStorage =
    std::make_unique<PyStateVectorStorage>();

static std::unique_ptr<PyStateStorage> cudaqStateStorage =
    std::make_unique<PyStateStorage>();

std::tuple<ExecutionEngine *, void *, std::size_t, std::int32_t>
jitAndCreateArgs(const std::string &name, MlirModule module,
                 cudaq::OpaqueArguments &runtimeArgs,
                 const std::vector<std::string> &names, Type returnType,
                 std::size_t startingArgIdx = 0) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "jitAndCreateArgs", name);
  auto mod = unwrap(module);

  // Do not cache the JIT if we are running with startingArgIdx > 0 because a)
  // we won't be executing right after JIT-ing, and b) we might get called later
  // this with startingArgIdx == 0, and we need that JIT to be performed and
  // cached.
  const bool allowCache = startingArgIdx == 0;

  // Have we JIT compiled this before?
  auto hash = llvm::hash_code{0};
  mod.walk([&hash](Operation *op) {
    hash = llvm::hash_combine(hash, OperationEquivalence::computeHash(op));
  });
  auto hashKey = static_cast<size_t>(hash);

  ExecutionEngine *jit = nullptr;
  if (allowCache && jitCache->hasJITEngine(hashKey)) {
    jit = jitCache->getJITEngine(hashKey);
  } else {
    ScopedTraceWithContext(cudaq::TIMING_JIT,
                           "jitAndCreateArgs - execute passes", name);

    auto cloned = mod.clone();
    auto context = cloned.getContext();
    PassManager pm(context);
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createPySynthCallableBlockArgs(names));
    pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader(/*genAsQuake=*/true));
    pm.addPass(cudaq::opt::createGenerateKernelExecution(
        {.startingArgIdx = startingArgIdx}));
    pm.addPass(cudaq::opt::createLambdaLiftingPass());
    cudaq::opt::addPipelineConvertToQIR(pm);

    DefaultTimingManager tm;
    tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
    auto timingScope = tm.getRootScope(); // starts the timer
    pm.enableTiming(timingScope);         // do this right before pm.run
    if (failed(pm.run(cloned)))
      throw std::runtime_error(
          "cudaq::builder failed to JIT compile the Quake representation.");
    timingScope.stop();

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
    if (allowCache)
      jitCache->cache(hashKey, jit);
  }

  // We need to append the return type to the OpaqueArguments here
  // so that we get a spot in the `rawArgs` memory for the
  // altLaunchKernel function to dump the result
  if (!isa<NoneType>(returnType))
    TypeSwitch<Type, void>(returnType)
        .Case([&](IntegerType type) {
          if (type.getIntOrFloatBitWidth() == 1) {
            bool *ourAllocatedArg = new bool();
            *ourAllocatedArg = 0;
            runtimeArgs.emplace_back(ourAllocatedArg, [](void *ptr) {
              delete static_cast<bool *>(ptr);
            });
            return;
          }

          long *ourAllocatedArg = new long();
          *ourAllocatedArg = 0;
          runtimeArgs.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<long *>(ptr);
          });
        })
        .Case([&](ComplexType type) {
          Py_complex *ourAllocatedArg = new Py_complex();
          ourAllocatedArg->real = 0.0;
          ourAllocatedArg->imag = 0.0;
          runtimeArgs.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<Py_complex *>(ptr);
          });
        })
        .Case([&](Float64Type type) {
          double *ourAllocatedArg = new double();
          *ourAllocatedArg = 0.;
          runtimeArgs.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<double *>(ptr);
          });
        })
        .Case([&](Float32Type type) {
          float *ourAllocatedArg = new float();
          *ourAllocatedArg = 0.;
          runtimeArgs.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<float *>(ptr);
          });
        })
        .Default([](Type ty) {
          std::string msg;
          {
            llvm::raw_string_ostream os(msg);
            ty.print(os);
          }
          throw std::runtime_error("Unsupported CUDA-Q kernel return type - " +
                                   msg + ".\n");
        });

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

  std::int32_t returnOffset = 0;
  if (runtimeArgs.size()) {
    auto expectedPtr = jit->lookup(name + ".returnOffset");
    if (!expectedPtr) {
      throw std::runtime_error(
          "cudaq::builder failed to get returnOffset function.");
    }
    auto returnOffsetCalculator =
        reinterpret_cast<std::int64_t (*)()>(*expectedPtr);
    returnOffset = (std::int32_t)returnOffsetCalculator();
    if (returnOffset == NoResultOffset) {
      returnOffset = 0;
    }
  }
  return {jit, rawArgs, size, returnOffset};
}

std::tuple<void *, std::size_t, std::int32_t>
pyAltLaunchKernelBase(const std::string &name, MlirModule module,
                      Type returnType, cudaq::OpaqueArguments &runtimeArgs,
                      const std::vector<std::string> &names,
                      std::size_t startingArgIdx = 0) {
  // Do not allow kernel execution if we are running with startingArgIdx > 0.
  // This is used in remote VQE execution.
  const bool launch = startingArgIdx == 0;

  auto [jit, rawArgs, size, returnOffset] = jitAndCreateArgs(
      name, module, runtimeArgs, names, returnType, startingArgIdx);

  auto mod = unwrap(module);
  auto thunkName = name + ".thunk";
  auto thunkPtr = jit->lookup(thunkName);
  if (!thunkPtr)
    throw std::runtime_error("cudaq::builder failed to get thunk function");

  auto thunk = reinterpret_cast<void (*)(void *)>(*thunkPtr);

  std::string properName = name;

  // If we have any state vector data, we need to extract the function pointer
  // to set that data, and then set it.
  for (auto &[stateHash, stateData] : *stateStorage) {
    if (stateData.kernelName != name)
      continue;

    auto setStateFPtr = jit->lookup("nvqpp.set.state." + stateHash);
    if (auto error = setStateFPtr.takeError()) {
      auto message = "python alt_launch_kernel failed to get set state "
                     "function for kernel: " +
                     name;
      llvm::logAllUnhandledErrors(std::move(error), llvm::errs(), message);
      throw std::runtime_error(message);
    }

    if (stateData.precision == simulation_precision::fp64) {
      auto setStateFunc =
          reinterpret_cast<void (*)(std::complex<double> *)>(*setStateFPtr);
      setStateFunc(reinterpret_cast<std::complex<double> *>(stateData.data));
      continue;
    }

    auto setStateFunc =
        reinterpret_cast<void (*)(std::complex<float> *)>(*setStateFPtr);
    setStateFunc(reinterpret_cast<std::complex<float> *>(stateData.data));
  }

  // If we have any cudaq state data, we need to extract the function pointer
  // to set that data, and then set it.
  for (auto &[stateHash, stateData] : *cudaqStateStorage) {
    if (stateData.kernelName != name)
      continue;

    auto setStateFPtr = jit->lookup("nvqpp.set.cudaq.state." + stateHash);
    if (auto error = setStateFPtr.takeError()) {
      auto message = "python alt_launch_kernel failed to get set cudaq state "
                     "function for kernel: " +
                     name;
      llvm::logAllUnhandledErrors(std::move(error), llvm::errs(), message);
      throw std::runtime_error(message);
    }

    auto setStateFunc =
        reinterpret_cast<void (*)(cudaq::state *)>(*setStateFPtr);
    setStateFunc(&stateData.data);
  }

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

  if (launch) {
    auto &platform = cudaq::get_platform();
    if (platform.is_remote() || platform.is_emulated()) {
      auto *wrapper = new cudaq::ArgWrapper{mod, names, rawArgs};
      cudaq::altLaunchKernel(name.c_str(), thunk,
                             reinterpret_cast<void *>(wrapper), size,
                             (uint64_t)returnOffset);
      delete wrapper;
    } else
      cudaq::altLaunchKernel(name.c_str(), thunk, rawArgs, size,
                             (uint64_t)returnOffset);
  }

  return std::make_tuple(rawArgs, size, returnOffset);
}

cudaq::KernelArgsHolder
pyCreateNativeKernel(const std::string &name, MlirModule module,
                     cudaq::OpaqueArguments &runtimeArgs) {
  auto [jit, rawArgs, size, returnOffset] =
      jitAndCreateArgs(name, module, runtimeArgs, {},
                       mlir::NoneType::get(unwrap(module).getContext()));

  auto thunkName = name + ".thunk";
  auto thunkPtr = jit->lookup(thunkName);
  if (!thunkPtr)
    throw std::runtime_error("Failed to get thunk function");
  const std::string properName = name;
  // If we have any state vector data, we need to extract the function pointer
  // to set that data, and then set it.
  for (auto &[stateHash, svdata] : *stateStorage) {
    if (svdata.kernelName != name)
      continue;
    auto setStateFPtr = jit->lookup("nvqpp.set.state." + stateHash);
    if (!setStateFPtr)
      throw std::runtime_error(
          "python CreateNativeKernel failed to get set state function.");

    if (svdata.precision == simulation_precision::fp64) {
      auto setStateFunc =
          reinterpret_cast<void (*)(std::complex<double> *)>(*setStateFPtr);
      setStateFunc(reinterpret_cast<std::complex<double> *>(svdata.data));
      continue;
    }

    auto setStateFunc =
        reinterpret_cast<void (*)(std::complex<float> *)>(*setStateFPtr);
    setStateFunc(reinterpret_cast<std::complex<float> *>(svdata.data));
  }

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
  cudaq::ArgWrapper wrapper{unwrap(module), {}, rawArgs};
  return cudaq::KernelArgsHolder(wrapper, size, returnOffset);
}

void pyAltLaunchKernel(const std::string &name, MlirModule module,
                       cudaq::OpaqueArguments &runtimeArgs,
                       const std::vector<std::string> &names) {
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());
  auto [rawArgs, size, returnOffset] =
      pyAltLaunchKernelBase(name, module, noneType, runtimeArgs, names);
  std::free(rawArgs);
}

/// @brief Serialize \p runtimeArgs into a flat buffer starting at
/// \p startingArgIdx (0-based). This does not execute the kernel. This is
/// useful for VQE applications when you want to serialize the constant
/// parameters that are not being optimized. The caller is responsible for
/// executing `std::free()` on the return value.
void *pyGetKernelArgs(const std::string &name, MlirModule module,
                      cudaq::OpaqueArguments &runtimeArgs,
                      const std::vector<std::string> &names,
                      std::size_t startingArgIdx) {
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());
  auto [rawArgs, size, returnOffset] = pyAltLaunchKernelBase(
      name, module, noneType, runtimeArgs, names, startingArgIdx);
  return rawArgs;
}

inline unsigned int byteSize(mlir::Type ty) {
  if (isa<ComplexType>(ty)) {
    auto eleTy = cast<ComplexType>(ty).getElementType();
    return 2 * cudaq::opt::convertBitsToBytes(eleTy.getIntOrFloatBitWidth());
  }
  return cudaq::opt::convertBitsToBytes(ty.getIntOrFloatBitWidth());
}

template <typename T>
py::object readPyObject(mlir::Type ty, char *arg) {
  unsigned int bytes = byteSize(ty);
  if (sizeof(T) != bytes) {
    ty.dump();
    throw std::runtime_error(
        "Error reading return value of type (reading bytes: " +
        std::to_string(sizeof(T)) +
        ", bytes available to read: " + std::to_string(bytes) + ")");
  }
  T concrete;
  std::memcpy(&concrete, arg, bytes);
  return py_ext::convert<T>(concrete);
}

py::object pyAltLaunchKernelR(const std::string &name, MlirModule module,
                              MlirType returnType,
                              cudaq::OpaqueArguments &runtimeArgs,
                              const std::vector<std::string> &names) {
  auto [rawArgs, size, returnOffset] = pyAltLaunchKernelBase(
      name, module, unwrap(returnType), runtimeArgs, names);

  auto unwrapped = unwrap(returnType);
  auto rawReturn = ((char *)rawArgs) + returnOffset;

  // Extract the return value from the rawReturn pointer.
  py::object returnValue =
      llvm::TypeSwitch<mlir::Type, py::object>(unwrapped)
          .Case([&](IntegerType ty) -> py::object {
            if (ty.getIntOrFloatBitWidth() == 1) {
              return readPyObject<bool>(ty, rawReturn);
            }
            return readPyObject<long>(ty, rawReturn);
          })
          .Case([&](ComplexType ty) -> py::object {
            auto eleTy = ty.getElementType();
            return llvm::TypeSwitch<mlir::Type, py::object>(eleTy)
                .Case([&](Float64Type eTy) -> py::object {
                  return readPyObject<std::complex<double>>(ty, rawReturn);
                })
                .Case([&](Float32Type eTy) -> py::object {
                  return readPyObject<std::complex<float>>(ty, rawReturn);
                })
                .Default([](Type eTy) -> py::object {
                  eTy.dump();
                  throw std::runtime_error(
                      "Invalid float element type for return "
                      "complex type for pyAltLaunchKernel.");
                });
          })
          .Case([&](Float64Type ty) -> py::object {
            return readPyObject<double>(ty, rawReturn);
          })
          .Case([&](Float32Type ty) -> py::object {
            return readPyObject<float>(ty, rawReturn);
          })
          .Default([](Type ty) -> py::object {
            ty.dump();
            throw std::runtime_error(
                "Invalid return type for pyAltLaunchKernel.");
          });

  std::free(rawArgs);
  return returnValue;
}

MlirModule synthesizeKernel(const std::string &name, MlirModule module,
                            cudaq::OpaqueArguments &runtimeArgs) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "synthesizeKernel", name);
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());

  auto [jit, rawArgs, size, returnOffset] =
      jitAndCreateArgs(name, module, runtimeArgs, {}, noneType);
  auto cloned = unwrap(module).clone();
  auto context = cloned.getContext();
  registerLLVMDialectTranslation(*context);

  // Get additional debug values
  auto disableMLIRthreading = getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", false);
  auto enablePrintMLIREachPass =
      getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);

  PassManager pm(context);
  pm.addPass(cudaq::opt::createQuakeSynthesizer(name, rawArgs, 0, true));
  pm.addPass(createCanonicalizerPass());

  // Run state preparation for quantum devices only.
  // Simulators have direct implementation of state initialization
  // in their runtime.
  auto &platform = cudaq::get_platform();
  if (!platform.is_simulator() || platform.is_emulated()) {
    pm.addPass(cudaq::opt::createConstPropComplex());
    pm.addPass(cudaq::opt::createLiftArrayAlloc());
    pm.addPass(cudaq::opt::createStatePreparation());
  }
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLoopNormalize());
  pm.addPass(cudaq::opt::createLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  DefaultTimingManager tm;
  tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
  auto timingScope = tm.getRootScope(); // starts the timer
  pm.enableTiming(timingScope);         // do this right before pm.run
  if (disableMLIRthreading || enablePrintMLIREachPass)
    context->disableMultithreading();
  if (enablePrintMLIREachPass)
    pm.enableIRPrinting();
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");
  timingScope.stop();
  std::free(rawArgs);
  return wrap(cloned);
}

std::string getQIR(const std::string &name, MlirModule module,
                   cudaq::OpaqueArguments &runtimeArgs,
                   const std::string &profile) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getQIR", name);
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());

  auto [jit, rawArgs, size, returnOffset] =
      jitAndCreateArgs(name, module, runtimeArgs, {}, noneType);
  auto cloned = unwrap(module).clone();
  auto context = cloned.getContext();

  PassManager pm(context);
  pm.addPass(cudaq::opt::createLambdaLiftingPass());
  if (profile.empty())
    cudaq::opt::addPipelineConvertToQIR(pm);
  else
    cudaq::opt::addPipelineConvertToQIR(pm, profile);
  DefaultTimingManager tm;
  tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
  auto timingScope = tm.getRootScope(); // starts the timer
  pm.enableTiming(timingScope);         // do this right before pm.run
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "getQIR failed to JIT compile the Quake representation.");
  timingScope.stop();
  std::free(rawArgs);

  llvm::LLVMContext llvmContext;
  llvmContext.setOpaquePointers(false);
  auto llvmModule = translateModuleToLLVMIR(cloned, llvmContext);
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get()))
    throw std::runtime_error("getQIR Failed to optimize LLVM IR ");

  std::string str;
  {
    llvm::raw_string_ostream os(str);
    llvmModule->print(os, nullptr);
  }
  return str;
}

std::string getASM(const std::string &name, MlirModule module,
                   cudaq::OpaqueArguments &runtimeArgs) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getASM", name);
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());

  auto [jit, rawArgs, size, returnOffset] =
      jitAndCreateArgs(name, module, runtimeArgs, {}, noneType);
  auto cloned = unwrap(module).clone();
  auto context = cloned.getContext();

  PassManager pm(context);
  pm.addPass(cudaq::opt::createLambdaLiftingPass());
  cudaq::opt::addPipelineTranslateToOpenQASM(pm);

  if (failed(pm.run(cloned)))
    throw std::runtime_error("getASM: code generation failed.");
  std::free(rawArgs);

  std::string str;
  llvm::raw_string_ostream os(str);
  if (failed(cudaq::translateToOpenQASM(cloned, os)))
    throw std::runtime_error("getASM: failed to translate to OpenQASM.");
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
        auto kernelFunc = getKernelFuncOp(module, kernelName);

        cudaq::OpaqueArguments args;
        cudaq::packArgs(args, runtimeArgs, kernelFunc, callableArgHandler);
        pyAltLaunchKernel(kernelName, module, args, callable_names);
      },
      py::arg("kernelName"), py::arg("module"), py::kw_only(),
      py::arg("callable_names") = std::vector<std::string>{}, "DOC STRING");

  mod.def(
      "pyAltLaunchKernelR",
      [&](const std::string &kernelName, MlirModule module, MlirType returnType,
          py::args runtimeArgs, std::vector<std::string> callable_names) {
        auto kernelFunc = getKernelFuncOp(module, kernelName);

        cudaq::OpaqueArguments args;
        cudaq::packArgs(args, runtimeArgs, kernelFunc, callableArgHandler);
        return pyAltLaunchKernelR(kernelName, module, returnType, args,
                                  callable_names);
      },
      py::arg("kernelName"), py::arg("module"), py::arg("returnType"),
      py::kw_only(), py::arg("callable_names") = std::vector<std::string>{},
      "DOC STRING");

  mod.def("synthesize", [](py::object kernel, py::args runtimeArgs) {
    MlirModule module = kernel.attr("module").cast<MlirModule>();
    auto name = kernel.attr("name").cast<std::string>();
    auto kernelFuncOp = getKernelFuncOp(module, name);
    cudaq::OpaqueArguments args;
    cudaq::packArgs(args, runtimeArgs, kernelFuncOp,
                    [](OpaqueArguments &, py::object &) { return false; });
    return synthesizeKernel(name, module, args);
  });

  mod.def(
      "get_qir",
      [](py::object kernel, std::string profile) {
        PyErr_WarnEx(PyExc_DeprecationWarning,
                     "to_qir()/get_qir() is deprecated, use translate() "
                     "with `format=\"qir\"`.",
                     1);

        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        MlirModule module = kernel.attr("module").cast<MlirModule>();
        auto name = kernel.attr("name").cast<std::string>();
        cudaq::OpaqueArguments args;
        return getQIR(name, module, args, profile);
      },
      py::arg("kernel"), py::kw_only(), py::arg("profile") = "");

  mod.def(
      "storePointerToStateData",
      [](const std::string &name, const std::string &hash, py::buffer data,
         simulation_precision precision) {
        auto ptr = data.request().ptr;
        stateStorage->insert({hash, PyStateVectorData{ptr, precision, name}});
      },
      "Store qalloc state initialization array data.");

  mod.def(
      "deletePointersToStateData",
      [](const std::vector<std::string> &hashes) {
        for (auto iter = stateStorage->cbegin(); iter != stateStorage->end();) {
          if (std::find(hashes.begin(), hashes.end(), iter->first) !=
              hashes.end()) {
            stateStorage->erase(iter++);
            continue;
          }
          iter++;
        }
      },
      "Remove our pointers to the qalloc array data.");

  mod.def(
      "storePointerToCudaqState",
      [](const std::string &name, const std::string &hash, py::object data) {
        auto state = data.cast<cudaq::state>();
        cudaqStateStorage->insert({hash, PyStateData{state, name}});
      },
      "Store qalloc state initialization states.");

  mod.def(
      "deletePointersToCudaqState",
      [](const std::vector<std::string> &hashes) {
        for (auto iter = cudaqStateStorage->cbegin();
             iter != cudaqStateStorage->end();) {
          if (std::find(hashes.begin(), hashes.end(), iter->first) !=
              hashes.end()) {
            cudaqStateStorage->erase(iter++);
            continue;
          }
          iter++;
        }
      },
      "Remove our pointers to the cudaq states.");
}
} // namespace cudaq
