/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "JITExecutionCache.h"
#include "common/ArgumentWrapper.h"
#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "utils/OpaqueArguments.h"

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
static std::unique_ptr<JITExecutionCache> jitCache;

struct PyStateVectorData {
  void *data = nullptr;
  simulation_precision precision = simulation_precision::fp32;
  std::string kernelName;
};
using PyStateVectorStorage = std::map<std::string, PyStateVectorData>;

static std::unique_ptr<PyStateVectorStorage> stateStorage =
    std::make_unique<PyStateVectorStorage>();

std::tuple<ExecutionEngine *, void *, std::size_t>
jitAndCreateArgs(const std::string &name, MlirModule module,
                 cudaq::OpaqueArguments &runtimeArgs,
                 const std::vector<std::string> &names, Type returnType) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "jitAndCreateArgs", name);
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
    ScopedTraceWithContext(cudaq::TIMING_JIT,
                           "jitAndCreateArgs - execute passes", name);

    PassManager pm(context);
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createPySynthCallableBlockArgs(names));
    pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader(/*genAsQuake=*/true));
    pm.addPass(cudaq::opt::createGenerateKernelExecution());
    pm.addPass(cudaq::opt::createLambdaLiftingPass());
    cudaq::opt::addPipelineToQIR<>(pm);

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
        .Case([&](Float64Type type) {
          double *ourAllocatedArg = new double();
          *ourAllocatedArg = 0.;
          runtimeArgs.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<double *>(ptr);
          });
        })
        .Default([](Type ty) {
          std::string msg;
          {
            llvm::raw_string_ostream os(msg);
            ty.print(os);
          }
          throw std::runtime_error(
              "Unsupported CUDA Quantum kernel return type - " + msg + ".\n");
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

  // If we have any state vector data, we need to extract the function pointer
  // to set that data, and then set it.
  for (auto &[stateHash, svdata] : *stateStorage) {
    if (svdata.kernelName != name)
      continue;
    auto setStateFPtr = jit->lookup("nvqpp.set.state." + stateHash);
    if (!setStateFPtr)
      throw std::runtime_error(
          "python alt_launch_kernel failed to get set state function.");

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

  auto &platform = cudaq::get_platform();
  if (platform.is_remote() || platform.is_emulated()) {
    auto *wrapper = new cudaq::ArgWrapper{mod, names, rawArgs};
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

  // We first need to compute the offset for the return value.
  // We'll loop through all the arguments and increment the
  // offset for the argument type. Then we'll be at our return type location.
  auto returnOffset = [&]() {
    std::size_t offset = 0;
    auto kernelFunc = getKernelFuncOp(module, name);
    for (auto argType : kernelFunc.getArgumentTypes())
      llvm::TypeSwitch<mlir::Type, void>(argType)
          .Case([&](IntegerType ty) {
            if (ty.getIntOrFloatBitWidth() == 1) {
              offset += 1;
              return;
            }

            offset += 8;
            return;
          })
          .Case([&](cc::StdvecType ty) { offset += 8; })
          .Case([&](Float64Type ty) { offset += 8; })
          .Default([](Type) {});

    return offset;
  }();

  // Extract the return value from the rawArgs pointer.
  return llvm::TypeSwitch<mlir::Type, py::object>(unwrapped)
      .Case([&](IntegerType ty) -> py::object {
        if (ty.getIntOrFloatBitWidth() == 1) {
          bool concrete = false;
          std::memcpy(&concrete, ((char *)rawArgs) + returnOffset, 1);
          std::free(rawArgs);
          return py::bool_(concrete);
        }
        std::size_t concrete;
        std::memcpy(&concrete, ((char *)rawArgs) + returnOffset, 8);
        std::free(rawArgs);
        return py::int_(concrete);
      })
      .Case([&](Float64Type ty) -> py::object {
        double concrete;
        std::memcpy(&concrete, ((char *)rawArgs) + returnOffset, 8);
        std::free(rawArgs);
        return py::float_(concrete);
      })
      .Default([](Type ty) -> py::object {
        ty.dump();
        throw std::runtime_error("Invalid return type for pyAltLaunchKernel.");
      });
}

MlirModule synthesizeKernel(const std::string &name, MlirModule module,
                            cudaq::OpaqueArguments &runtimeArgs) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "synthesizeKernel", name);
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
  DefaultTimingManager tm;
  tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
  auto timingScope = tm.getRootScope(); // starts the timer
  pm.enableTiming(timingScope);         // do this right before pm.run
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");
  timingScope.stop();
  std::free(rawArgs);
  return wrap(cloned);
}

std::string getQIRLL(const std::string &name, MlirModule module,
                     cudaq::OpaqueArguments &runtimeArgs,
                     std::string &profile) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getQIRLL", name);
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
  DefaultTimingManager tm;
  tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
  auto timingScope = tm.getRootScope(); // starts the timer
  pm.enableTiming(timingScope);         // do this right before pm.run
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");
  timingScope.stop();
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
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();
        MlirModule module = kernel.attr("module").cast<MlirModule>();
        auto name = kernel.attr("name").cast<std::string>();
        cudaq::OpaqueArguments args;
        return getQIRLL(name, module, args, profile);
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
}
} // namespace cudaq