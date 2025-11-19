/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_alt_launch_kernel.h"
#include "JITExecutionCache.h"
#include "common/AnalogHamiltonian.h"
#include "common/ArgumentConversion.h"
#include "common/ArgumentWrapper.h"
#include "common/Environment.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/OptUtils.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "runtime/cudaq/algorithms/py_utils.h"
#include "utils/OpaqueArguments.h"
#include "utils/PyTypes.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include <fmt/core.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir;

// TODO: unify with the definition in GenKernelExec.cpp
static constexpr std::int32_t NoResultOffset =
    std::numeric_limits<std::int32_t>::max();

static std::unique_ptr<cudaq::JITExecutionCache> jitCache;

static std::function<std::string()> getTransportLayer = []() -> std::string {
  throw std::runtime_error("binding for kernel launch is incomplete");
};

namespace cudaq {

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

static std::string createDataLayout() {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target)
    throw std::runtime_error("Cannot create target");

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  if (!machine)
    throw std::runtime_error("Cannot create target machine");

  return machine->createDataLayout().getStringRepresentation();
}

void setDataLayout(MlirModule module) {
  auto mod = unwrap(module);
  if (!mod->hasAttr(opt::factory::targetDataLayoutAttrName)) {
    auto dataLayout = createDataLayout();
    mod->setAttr(opt::factory::targetDataLayoutAttrName,
                 StringAttr::get(mod->getContext(), dataLayout));
  }
}

std::function<bool(OpaqueArguments &argData, py::object &arg)>
getCallableArgHandler() {
  return [](cudaq::OpaqueArguments &argData, py::object &arg) {
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
}

/// @brief Create a new OpaqueArguments pointer and pack the python arguments
/// in it. Clients must delete the memory.
OpaqueArguments *
toOpaqueArgs(py::args &args, MlirModule mod, const std::string &name,
             const std::optional<
                 std::function<bool(OpaqueArguments &argData, py::object &arg)>>
                 &optionalBackupHandler) {
  auto kernelFunc = getKernelFuncOp(mod, name);
  auto *argData = new cudaq::OpaqueArguments();
  args = simplifiedValidateInputArguments(args);
  setDataLayout(mod);
  auto backupHandler = optionalBackupHandler.value_or(
      [](OpaqueArguments &, py::object &) { return false; });
  cudaq::packArgs(*argData, args, kernelFunc, backupHandler);
  return argData;
}

ExecutionEngine *jitKernel(const std::string &name, MlirModule module,
                           const std::vector<std::string> &names,
                           std::size_t startingArgIdx = 0) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "jitKernel", name);
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
    ScopedTraceWithContext(cudaq::TIMING_JIT, "jitKernel - execute passes",
                           name);

    auto cloned = mod.clone();
    auto context = cloned.getContext();
    PassManager pm(context);
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createPySynthCallableBlockArgs(
        SmallVector<StringRef>(names.begin(), names.end())));
    pm.addPass(cudaq::opt::createLambdaLiftingPass());
    pm.addPass(cudaq::opt::createGenerateKernelExecution(
        {.startingArgIdx = startingArgIdx}));
    pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader({.jitTime = true}));
    pm.addPass(cudaq::opt::createReturnToOutputLog());
    pm.addPass(cudaq::opt::createLambdaLiftingPass());
    pm.addPass(cudaq::opt::createDistributedDeviceCall());
    std::string tl = getTransportLayer();
    auto tlPair = StringRef(tl).split(':');
    if (tlPair.first != "qir") {
      // FIXME: this code path has numerous bugs for anything not full QIR, so
      // do an end-around for now and pretend it was full QIR.
      if (tlPair.second.empty())
        tl = "qir:0.1";
      else
        tl = "qir:" + tlPair.second.str();
    }
    cudaq::opt::addAOTPipelineConvertToQIR(pm, tl);
    pm.addPass(createSymbolDCEPass());

    auto enablePrintMLIREachPass =
        getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);

    if (enablePrintMLIREachPass) {
      cloned.getContext()->disableMultithreading();
      pm.enableIRPrinting();
    }

    std::string error_msg;
    mlir::DiagnosticEngine &engine = context->getDiagEngine();
    auto handlerId = engine.registerHandler(
        [&error_msg](mlir::Diagnostic &diag) -> mlir::LogicalResult {
          if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
            error_msg += diag.str();
            return mlir::failure(false);
          }
          return mlir::failure();
        });

    DefaultTimingManager tm;
    tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
    auto timingScope = tm.getRootScope(); // starts the timer
    pm.enableTiming(timingScope);         // do this right before pm.run

    if (failed(pm.run(cloned))) {
      engine.eraseHandler(handlerId);
      throw std::runtime_error(
          "failed to JIT compile the Quake representation\n" + error_msg);
    }
    timingScope.stop();
    engine.eraseHandler(handlerId);

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

  return jit;
}

std::tuple<ExecutionEngine *, void *, std::size_t, std::int32_t>
jitAndCreateArgs(const std::string &name, MlirModule module,
                 cudaq::OpaqueArguments &runtimeArgs,
                 const std::vector<std::string> &names, Type returnType,
                 std::size_t startingArgIdx = 0) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "jitAndCreateArgs", name);
  auto jit = jitKernel(name, module, names, startingArgIdx);

  // We need to append the return type to the OpaqueArguments here
  // so that we get a spot in the `rawArgs` memory for the
  // altLaunchKernel function to dump the result
  if (returnType && !isa<NoneType>(returnType))
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
        .Case([&](cudaq::cc::StdvecType ty) {
          // Vector is a span: `{ data, length }`.
          struct vec {
            char *data;
            std::size_t length;
          };
          vec *ourAllocatedArg = new vec{nullptr, 0};
          runtimeArgs.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<vec *>(ptr);
          });
        })
        .Case([&](cudaq::cc::StructType ty) {
          auto funcOp = getKernelFuncOp(module, name);
          auto [size, offsets] = getTargetLayout(funcOp, ty);
          auto memberTys = ty.getMembers();
          for (auto mTy : memberTys) {
            if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(mTy))
              throw std::runtime_error("return values with dynamically sized "
                                       "element types are not yet supported");
          }
          auto ourAllocatedArg = std::malloc(size);
          runtimeArgs.emplace_back(ourAllocatedArg,
                                   [](void *ptr) { std::free(ptr); });
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

/// @brief Set captured data into globals read by the kernel.
///
/// Kernel compilation prepares the state storage as following:
/// - creates globals to hold captured vector and state data
/// - adds code to the kernel that reads the data from globals
/// - creates setter functions to store values into the globals
/// - saves unique setter hashes and the captured data into state storage
///
/// Now we can use the setters to store captured data into the globals.
void storeCapturedData(ExecutionEngine *jit, const std::string &kernelName) {
  auto &platform = cudaq::get_platform();
  // If we have any state vector data, we need to extract the function pointer
  // to set that data, and then set it.
  for (auto &[stateHash, stateData] : *stateStorage) {
    if (stateData.kernelName != kernelName)
      continue;

    // Ignore stale kernel state data.
    auto setStateFPtr = jit->lookup("nvqpp.set.state." + stateHash);
    if (auto error = setStateFPtr.takeError()) {
      llvm::logAllUnhandledErrors(std::move(error), llvm::nulls());
      continue;
    }

    if (platform.is_remote() || platform.is_emulated())
      throw std::runtime_error("captured vectors are not supported on quantum "
                               "hardware or remote simulators");

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
    if (stateData.kernelName != kernelName)
      continue;

    // Ignore stale kernel state data.
    auto setStateFPtr = jit->lookup("nvqpp.set.cudaq.state." + stateHash);
    if (auto error = setStateFPtr.takeError()) {
      llvm::logAllUnhandledErrors(std::move(error), llvm::nulls());
      continue;
    }

    if (platform.is_remote() || platform.is_emulated())
      throw std::runtime_error("captured states are not supported on quantum "
                               "hardware or remote simulators");

    auto setStateFunc =
        reinterpret_cast<void (*)(cudaq::state *)>(*setStateFPtr);
    setStateFunc(&stateData.data);
  }
}

// NB: this is a backdoor marshaling of `std::vector<bool>` and it must be kept
// in synch with `__nvqpp_vector_bool_to_initializer_list()`.
static bool marshalRuntimeArgs(mlir::IntegerType i1Ty,
                               cudaq::OpaqueArguments &newArgs,
                               const std::vector<void *> &origArgs,
                               mlir::TypeRange inputTys) {
  for (auto [ptr, ty] : llvm::zip(origArgs, inputTys)) {
    if (auto vecTy = mlir::dyn_cast<cc::StdvecType>(ty)) {
      if (vecTy.getElementType() == i1Ty) {
        auto *vbp = reinterpret_cast<std::vector<bool> *>(ptr);
        // Add a deleter for this new allocation.
        auto *initList = new std::vector<char>;
        for (bool b : *vbp)
          initList->emplace_back(static_cast<char>(b));
        newArgs.emplace_back(initList, [](void *p) {
          delete static_cast<std::vector<char> *>(p);
        });
        continue;
      }
      if (mlir::isa<cc::StdvecType>(vecTy.getElementType())) {
        // Can't handle recursive lists, so punt for now.
        return false;
      }
    }
    // NB: do _not_ delete copied pointers as they are deleted elsewhere!
    newArgs.emplace_back(ptr, [](void *) {});
  }
  if (origArgs.size() > inputTys.size()) {
    // Apparently this happens for quantinuum local emulation tests? FIXME! This
    // seems like a serious bug.
    for (auto [i, ptr] : llvm::enumerate(origArgs)) {
      if (i < inputTys.size())
        continue;
      // Make copies of the residual so things stay tilted in a good direction.
      newArgs.emplace_back(ptr, [](void *) {});
    }
    // Buckle up, we're going to call the streamlined launch here.
  }
  return true;
}

void pyLaunchKernel(const std::string &name, KernelThunkType thunk,
                    mlir::ModuleOp mod, cudaq::OpaqueArguments &runtimeArgs,
                    void *rawArgs, std::size_t size, std::uint32_t returnOffset,
                    const std::vector<std::string> &names) {
  auto &platform = cudaq::get_platform();
  auto isRemoteSimulator = platform.get_remote_capabilities().isRemoteSimulator;
  auto isQuantumDevice =
      !isRemoteSimulator && (platform.is_remote() || platform.is_emulated());

  if (isRemoteSimulator) {
    // Remote simulator - use altLaunchKernel to support returning values.
    // TODO: after cudaq::run support this should be merged with the quantum
    // device case.
    std::unique_ptr<cudaq::ArgWrapper> wrapper(
        new cudaq::ArgWrapper{mod, names, rawArgs});
    auto dynamicResult = cudaq::altLaunchKernel(
        name.c_str(), thunk, reinterpret_cast<void *>(wrapper.get()), size,
        returnOffset);
    if (dynamicResult.data_buffer || dynamicResult.size)
      throw std::runtime_error("not implemented: support dynamic results");
  } else if (isQuantumDevice) {
    // Quantum devices or their emulation - we can use streamlinedLaunchKernel
    // as quantum platform do not support direct returns.
    auto fn = mod.lookupSymbol<mlir::func::FuncOp>(runtime::cudaqGenPrefixName +
                                                   name);
    if (!fn)
      throw std::runtime_error("cannot find kernel " + name);
    OpaqueArguments marshaledArgs;
    bool ok = marshalRuntimeArgs(mlir::IntegerType::get(mod.getContext(), 1),
                                 marshaledArgs, runtimeArgs.getArgs(),
                                 fn.getFunctionType().getInputs());
    if (ok) {
      auto dynamicResult =
          cudaq::streamlinedLaunchKernel(name.c_str(), marshaledArgs.getArgs());
      if (dynamicResult.data_buffer || dynamicResult.size)
        throw std::runtime_error("not implemented: support dynamic results");
    } else {
      // Backdoor approach to marshaling the arguments failed, so use the
      // compiler generated code.
      auto dynamicResult = cudaq::altLaunchKernel(name.c_str(), thunk, rawArgs,
                                                  size, returnOffset);
      if (dynamicResult.data_buffer || dynamicResult.size)
        throw std::runtime_error("not implemented: support dynamic results");
    }
  } else {
    // Local simulator - use altLaunchKernel with the thunk function.
    auto dynamicResult = cudaq::altLaunchKernel(name.c_str(), thunk, rawArgs,
                                                size, returnOffset);
    if (dynamicResult.data_buffer || dynamicResult.size)
      throw std::runtime_error("not implemented: support dynamic results");
  }
}

std::tuple<void *, std::size_t, std::int32_t, KernelThunkType>
pyAltLaunchKernelBase(const std::string &name, MlirModule module,
                      Type returnType, cudaq::OpaqueArguments &runtimeArgs,
                      const std::vector<std::string> &names,
                      std::size_t startingArgIdx, bool launch) {
  // Do not allow kernel execution if we are running with startingArgIdx > 0.
  // This is used in remote VQE execution.
  launch = launch && (startingArgIdx == 0);

  auto [jit, rawArgs, size, returnOffset] = jitAndCreateArgs(
      name, module, runtimeArgs, names, returnType, startingArgIdx);

  auto mod = unwrap(module);
  auto thunkName = name + ".thunk";
  auto thunkPtr = jit->lookup(thunkName);
  if (!thunkPtr)
    throw std::runtime_error("cudaq::builder failed to get thunk function");

  auto thunk = reinterpret_cast<KernelThunkType>(*thunkPtr);

  std::string properName = name;

  // Store captured vectors and states into globals read by the kernel.
  storeCapturedData(jit, name);

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

  if (launch)
    pyLaunchKernel(name, thunk, mod, runtimeArgs, rawArgs, size, returnOffset,
                   names);

  return std::make_tuple(rawArgs, size, returnOffset, thunk);
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

  // Store captured vectors and states into globals read by the kernel.
  storeCapturedData(jit, name);

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
  auto [rawArgs, size, returnOffset, thunk] =
      pyAltLaunchKernelBase(name, module, noneType, runtimeArgs, names);
  std::free(rawArgs);
}

void pyAltLaunchAnalogKernel(const std::string &name,
                             std::string &programArgs) {
  if (name.find(cudaq::runtime::cudaqAHKPrefixName) != 0)
    throw std::runtime_error("Unexpected type of kernel.");
  auto dynamicResult = cudaq::altLaunchKernel(
      name.c_str(), KernelThunkType(nullptr),
      (void *)(const_cast<char *>(programArgs.c_str())), 0, 0);
  if (dynamicResult.data_buffer || dynamicResult.size)
    throw std::runtime_error("Not implemented: support dynamic results");
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
  auto [rawArgs, size, returnOffset, thunk] = pyAltLaunchKernelBase(
      name, module, noneType, runtimeArgs, names, startingArgIdx);
  return rawArgs;
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

/// @brief Convert raw return of kernel to python object.
py::object convertResult(mlir::ModuleOp module, mlir::func::FuncOp kernelFuncOp,
                         mlir::Type ty, char *data) {
  auto isRunContext = module->hasAttr(runtime::enableCudaqRun);

  return llvm::TypeSwitch<mlir::Type, py::object>(ty)
      .Case([&](IntegerType ty) -> py::object {
        if (ty.getIntOrFloatBitWidth() == 1)
          return readPyObject<bool>(ty, data);
        if (ty.getIntOrFloatBitWidth() == 8)
          return readPyObject<std::int8_t>(ty, data);
        if (ty.getIntOrFloatBitWidth() == 16)
          return readPyObject<std::int16_t>(ty, data);
        if (ty.getIntOrFloatBitWidth() == 32)
          return readPyObject<std::int32_t>(ty, data);
        return readPyObject<std::int64_t>(ty, data);
      })
      .Case([&](mlir::ComplexType ty) -> py::object {
        auto eleTy = ty.getElementType();
        return llvm::TypeSwitch<mlir::Type, py::object>(eleTy)
            .Case([&](mlir::Float64Type eTy) -> py::object {
              return readPyObject<std::complex<double>>(ty, data);
            })
            .Case([&](mlir::Float32Type eTy) -> py::object {
              return readPyObject<std::complex<float>>(ty, data);
            })
            .Default([](mlir::Type eTy) -> py::object {
              eTy.dump();
              throw std::runtime_error(
                  "Unsupported float element type for complex type return.");
            });
      })
      .Case([&](Float64Type ty) -> py::object {
        return readPyObject<double>(ty, data);
      })
      .Case([&](Float32Type ty) -> py::object {
        return readPyObject<float>(ty, data);
      })
      .Case([&](cudaq::cc::StdvecType ty) -> py::object {
        if (isRunContext) {
          // cudaq.run return.
          auto eleTy = ty.getElementType();
          auto eleByteSize = byteSize(eleTy);

          // Vector of booleans has a special layout.
          // Read the vector and create a list of booleans.
          if (eleTy.getIntOrFloatBitWidth() == 1) {
            auto v = reinterpret_cast<std::vector<bool> *>(data);
            py::list list;
            for (auto const bit : *v)
              list.append(py::bool_(bit));
            return list;
          }

          // Vector is a triple of pointers: `{ begin, end, end }`.
          // Read `begin` and `end` pointers from the buffer.
          struct vec {
            char *begin;
            char *end;
            char *end2;
          };
          auto v = reinterpret_cast<vec *>(data);

          // Read vector elements.
          py::list list;
          for (char *i = v->begin; i < v->end; i += eleByteSize)
            list.append(convertResult(module, kernelFuncOp, eleTy, i));
          return list;
        }

        // Direct call return.
        auto eleTy = ty.getElementType();
        auto eleByteSize = byteSize(eleTy);

        // Vector is a span: `{ data, length }`.
        // Read `data` and `length` from the buffer.
        struct vec {
          char *data;
          std::size_t length;
        };
        auto v = reinterpret_cast<vec *>(data);

        // Read vector elements.
        py::list list;
        std::size_t byteLength = v->length * eleByteSize;
        for (std::size_t i = 0; i < byteLength; i += eleByteSize)
          list.append(convertResult(module, kernelFuncOp, eleTy, v->data + i));
        return list;
      })
      .Case([&](cudaq::cc::StructType ty) -> py::object {
        auto name = ty.getName().str();
        // Handle tuples.
        if (name == "tuple") {
          auto [size, offsets] = getTargetLayout(kernelFuncOp, ty);
          auto memberTys = ty.getMembers();
          py::list list;
          for (std::size_t i = 0; i < offsets.size(); i++) {
            auto eleTy = memberTys[i];
            if (!eleTy.isIntOrFloat()) {
              // TODO: support nested aggregate types.
              eleTy.dump();
              throw std::runtime_error(
                  "Unsupported element type in struct type.");
            }
            list.append(
                convertResult(module, kernelFuncOp, eleTy, data + offsets[i]));
          }
          return py::tuple(list);
        }

        // Handle data class objects.
        if (!DataClassRegistry::isRegisteredClass(name))
          throw std::runtime_error("Dataclass is not registered: " + name);

        // Find class information.
        auto [cls, attributes] = DataClassRegistry::getClassAttributes(name);

        // Collect field names.
        std::vector<py::str> fieldNames;
        for (const auto &[attr_name, unused] : attributes)
          fieldNames.emplace_back(py::str(attr_name));

        // Read field values and create the constructor `kwargs`
        auto [size, offsets] = getTargetLayout(kernelFuncOp, ty);
        auto memberTys = ty.getMembers();
        py::dict kwargs;
        for (std::size_t i = 0; i < offsets.size(); i++) {
          auto eleTy = memberTys[i];
          if (!eleTy.isIntOrFloat()) {
            // TODO: support nested aggregate types.
            eleTy.dump();
            throw std::runtime_error(
                "Unsupported element type in struct type.");
          }
          if (i < fieldNames.size())
            kwargs[fieldNames[i]] =
                convertResult(module, kernelFuncOp, eleTy, data + offsets[i]);
          else
            throw std::runtime_error("Field name and value mismatch when "
                                     "returning an object of dataclass " +
                                     name);
        }

        // Create python object of class `cls` with the collected args.
        return cls(**kwargs);
      })
      .Default([](Type ty) -> py::object {
        ty.dump();
        throw std::runtime_error("Unsupported return type.");
      });
}

py::object pyAltLaunchKernelR(const std::string &name, MlirModule module,
                              MlirType returnType,
                              cudaq::OpaqueArguments &runtimeArgs,
                              const std::vector<std::string> &names) {
  auto mod = unwrap(module);
  auto returnTy = unwrap(returnType);

  auto [rawArgs, size, returnOffset, thunk] =
      pyAltLaunchKernelBase(name, module, returnTy, runtimeArgs, names);

  auto rawReturn = ((char *)rawArgs) + returnOffset;
  auto funcOp = cudaq::getKernelFuncOp(module, name);

  auto returnValue = convertResult(mod, funcOp, returnTy, rawReturn);
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

  auto &platform = cudaq::get_platform();
  auto isRemoteSimulator = platform.get_remote_capabilities().isRemoteSimulator;
  auto isLocalSimulator = platform.is_simulator() && !platform.is_emulated();
  auto isSimulator = isLocalSimulator || isRemoteSimulator;

  cudaq::opt::ArgumentConverter argCon(name, unwrap(module));
  argCon.gen(runtimeArgs.getArgs());

  // Store kernel and substitution strings on the stack.
  // We pass string references to the `createArgumentSynthesisPass`.
  mlir::SmallVector<std::string> kernels;
  mlir::SmallVector<std::string> substs;
  for (auto *kInfo : argCon.getKernelSubstitutions()) {
    std::string kernName =
        cudaq::runtime::cudaqGenPrefixName + kInfo->getKernelName().str();
    kernels.emplace_back(kernName);
    std::string substBuff;
    llvm::raw_string_ostream ss(substBuff);
    ss << kInfo->getSubstitutionModule();
    substs.emplace_back(substBuff);
  }

  // Collect references for the argument synthesis.
  mlir::SmallVector<mlir::StringRef> kernelRefs{kernels.begin(), kernels.end()};
  mlir::SmallVector<mlir::StringRef> substRefs{substs.begin(), substs.end()};

  PassManager pm(context);
  pm.addPass(opt::createArgumentSynthesisPass(kernelRefs, substRefs));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(opt::createDeleteStates());
  pm.addNestedPass<mlir::func::FuncOp>(opt::createReplaceStateWithKernel());
  pm.addPass(mlir::createSymbolDCEPass());

  // Run state preparation for quantum devices (or their emulation) only.
  // Simulators have direct implementation of state initialization
  // in their runtime.
  if (!isSimulator) {
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createConstantPropagation());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createLiftArrayAlloc());
    pm.addPass(cudaq::opt::createGlobalizeArrayValues());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createStatePreparation());
  }
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopUnroll());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());
  if (disableMLIRthreading || enablePrintMLIREachPass)
    context->disableMultithreading();
  if (enablePrintMLIREachPass)
    pm.enableIRPrinting();

  std::string error_msg;
  mlir::DiagnosticEngine &engine = context->getDiagEngine();
  auto handlerId = engine.registerHandler(
      [&error_msg](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
          error_msg += diag.str();
          return mlir::failure(false);
        }
        return mlir::failure();
      });

  DefaultTimingManager tm;
  tm.setEnabled(cudaq::isTimingTagEnabled(cudaq::TIMING_JIT_PASSES));
  auto timingScope = tm.getRootScope(); // starts the timer
  pm.enableTiming(timingScope);         // do this right before pm.run

  if (failed(pm.run(cloned))) {
    engine.eraseHandler(handlerId);
    throw std::runtime_error(
        "failed to JIT compile the Quake representation\n" + error_msg);
  }
  timingScope.stop();
  engine.eraseHandler(handlerId);
  std::free(rawArgs);
  return wrap(cloned);
}

std::string getQIR(const std::string &name, MlirModule module,
                   cudaq::OpaqueArguments &runtimeArgs,
                   const std::string &profile_) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getQIR", name);
  auto noneType = mlir::NoneType::get(unwrap(module).getContext());

  auto [jit, rawArgs, size, returnOffset] =
      jitAndCreateArgs(name, module, runtimeArgs, {}, noneType);
  auto cloned = unwrap(module).clone();
  auto context = cloned.getContext();

  PassManager pm(context);
  pm.addPass(cudaq::opt::createLambdaLiftingPass());
  pm.addPass(cudaq::opt::createReturnToOutputLog());
  std::string profile{profile_};
  if (profile.empty())
    profile = "qir:0.1";
  cudaq::opt::addAOTPipelineConvertToQIR(pm, profile);
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
  auto optPipeline = cudaq::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get()))
    throw std::runtime_error("getQIR Failed to optimize LLVM IR ");

  std::string str;
  llvm::raw_string_ostream os(str);
  llvmModule->print(os, nullptr);
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

  // Get additional debug values
  auto disableMLIRthreading = getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", false);
  auto enablePrintMLIREachPass =
      getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);

  PassManager pm(context);
  pm.addPass(cudaq::opt::createLambdaLiftingPass());
  // Run most of the passes from hardware pipelines.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopUnroll());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLiftArrayAlloc());
  pm.addPass(cudaq::opt::createGlobalizeArrayValues());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createStatePreparation());
  pm.addPass(cudaq::opt::createGetConcreteMatrix());
  pm.addPass(cudaq::opt::createUnitarySynthesis());
  pm.addPass(cudaq::opt::createApplySpecialization());
  cudaq::opt::addAggressiveInlining(pm);
  pm.addPass(createSymbolDCEPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(
      cudaq::opt::createMultiControlDecompositionPass());
  pm.addPass(cudaq::opt::createDecompositionPass(
      {.enabledPatterns = {"SToR1", "TToR1", "R1ToU3", "U3ToRotations",
                           "CHToCX", "CCZToCX", "CRzToCX", "CRyToCX", "CRxToCX",
                           "CR1ToCX", "CCZToCX", "RxAdjToRx", "RyAdjToRy",
                           "RzAdjToRz"}}));
  pm.addPass(cudaq::opt::createQuakeToCCPrep());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandControlVeqs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
  cudaq::opt::addPipelineTranslateToOpenQASM(pm);

  if (disableMLIRthreading || enablePrintMLIREachPass)
    context->disableMultithreading();
  if (enablePrintMLIREachPass)
    pm.enableIRPrinting();
  if (failed(pm.run(cloned)))
    throw std::runtime_error("getASM: code generation failed.");
  std::free(rawArgs);

  std::string str;
  llvm::raw_string_ostream os(str);
  if (failed(cudaq::translateToOpenQASM(cloned, os)))
    throw std::runtime_error("getASM: failed to translate to OpenQASM.");
  return str;
}

std::vector<std::string> getCallableNames(py::object &kernel, py::args &args) {
  // Handle callable arguments, if any, similar to `PyKernelDecorator.__call__`,
  // so that the callable arguments are properly packed for `pyAltLaunchKernel`
  // as if it's launched from Python.
  std::vector<std::string> callableNames;
  for (std::size_t i = 0; i < args.size(); ++i) {
    auto arg = args[i];
    // If this is a `PyKernelDecorator` callable:
    if (py::hasattr(arg, "__call__") && py::hasattr(arg, "module") &&
        py::hasattr(arg, "name")) {
      if (py::hasattr(arg, "compile"))
        arg.attr("compile")();

      if (py::hasattr(kernel, "processCallableArg"))
        kernel.attr("processCallableArg")(arg);
      callableNames.push_back(arg.attr("name").cast<std::string>());
    }
  }
  return callableNames;
}

void bindAltLaunchKernel(py::module &mod,
                         std::function<std::string()> &&getTL) {
  jitCache = std::make_unique<JITExecutionCache>();
  getTransportLayer = std::move(getTL);

  mod.def(
      "pyAltLaunchKernel",
      [&](const std::string &kernelName, MlirModule module,
          py::args runtimeArgs, std::vector<std::string> callable_names) {
        auto kernelFunc = getKernelFuncOp(module, kernelName);

        cudaq::OpaqueArguments args;
        setDataLayout(module);
        cudaq::packArgs(args, runtimeArgs, kernelFunc, getCallableArgHandler());
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
        setDataLayout(module);
        cudaq::packArgs(args, runtimeArgs, kernelFunc, getCallableArgHandler());
        return pyAltLaunchKernelR(kernelName, module, returnType, args,
                                  callable_names);
      },
      py::arg("kernelName"), py::arg("module"), py::arg("returnType"),
      py::kw_only(), py::arg("callable_names") = std::vector<std::string>{},
      "DOC STRING");

  mod.def(
      "pyAltLaunchAnalogKernel",
      [&](const std::string &name, std::string &programArgs) {
        return pyAltLaunchAnalogKernel(name, programArgs);
      },
      py::arg("name"), py::arg("programArgs"),
      "Launch an analog Hamiltonian simulation kernel with given JSON "
      "payload.");

  mod.def("synthesize", [](py::object kernel, py::args runtimeArgs) {
    MlirModule module = kernel.attr("module").cast<MlirModule>();
    auto name = kernel.attr("name").cast<std::string>();
    auto kernelFuncOp = getKernelFuncOp(module, name);
    cudaq::OpaqueArguments args;
    setDataLayout(module);
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
        for (auto iter = stateStorage->cbegin();
             iter != stateStorage->cend();) {
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
             iter != cudaqStateStorage->cend();) {
          if (std::find(hashes.begin(), hashes.end(), iter->first) !=
              hashes.end()) {
            cudaqStateStorage->erase(iter++);
            continue;
          }
          iter++;
        }
      },
      "Remove our pointers to the cudaq states.");

  mod.def(
      "mergeExternalMLIR",
      [](MlirModule modA, const std::string &modBStr) {
        auto ctx = unwrap(modA).getContext();
        auto moduleB = parseSourceString<ModuleOp>(modBStr, ctx);
        auto moduleA = unwrap(modA).clone();
        moduleB->walk([&moduleA](func::FuncOp op) {
          if (!moduleA.lookupSymbol<func::FuncOp>(op.getName()))
            moduleA.push_back(op.clone());
          return WalkResult::advance();
        });
        return wrap(moduleA);
      },
      "Merge the two Modules into a single Module.");

  mod.def(
      "synthPyCallable",
      [](MlirModule modA, const std::vector<std::string> &funcNames) {
        auto m = unwrap(modA);
        auto context = m.getContext();
        PassManager pm(context);
        pm.addNestedPass<func::FuncOp>(
            cudaq::opt::createPySynthCallableBlockArgs(
                SmallVector<StringRef>(funcNames.begin(), funcNames.end()),
                true));
        if (failed(pm.run(m)))
          throw std::runtime_error(
              "cudaq::jit failed to remove callable block arguments.");

        // fix up the mangled name map
        DictionaryAttr attr;
        m.walk([&](func::FuncOp op) {
          if (op->hasAttrOfType<UnitAttr>("cudaq-entrypoint")) {
            auto strAttr = StringAttr::get(
                context, op.getName().str() + "_PyKernelEntryPointRewrite");
            attr = DictionaryAttr::get(
                context, {NamedAttribute(StringAttr::get(context, op.getName()),
                                         strAttr)});
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (attr)
          m->setAttr("quake.mangled_name_map", attr);
      },
      "Synthesize away the callable block argument from the entrypoint in "
      "`modA` with the `FuncOp` of given name.");

  mod.def(
      "jitAndGetFunctionPointer",
      [](MlirModule mod, const std::string &funcName) {
        auto jit = jitKernel(funcName, mod, {});
        auto funcPtr = jit->lookup(funcName);
        if (!funcPtr) {
          throw std::runtime_error(
              "cudaq::builder failed to get kernelReg function.");
        }

        return py::capsule(*funcPtr);
      },
      "JIT compile and return the C function pointer for the FuncOp of given "
      "name.");
}
} // namespace cudaq
