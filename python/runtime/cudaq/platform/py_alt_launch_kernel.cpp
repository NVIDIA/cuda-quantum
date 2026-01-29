/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/OptUtils.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "runtime/cudaq/algorithms/py_utils.h"
#include "utils/LinkedLibraryHolder.h"
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
#include "mlir/IR/Builders.h"
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

static std::unique_ptr<cudaq::JITExecutionCache> jitCache;

static std::function<std::string()> getTransportLayer = []() -> std::string {
  throw std::runtime_error("binding for kernel launch is incomplete");
};

namespace {
struct PyStateVectorData {
  void *data = nullptr;
  cudaq::simulation_precision precision = cudaq::simulation_precision::fp32;
  std::string kernelName;
};
} // namespace
using PyStateVectorStorage = std::map<std::string, PyStateVectorData>;

namespace {
struct PyStateData {
  cudaq::state data;
  std::string kernelName;
};
} // namespace
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

std::size_t cudaq::byteSize(Type ty) {
  if (isa<ComplexType>(ty)) {
    auto eleTy = cast<ComplexType>(ty).getElementType();
    return 2 * opt::convertBitsToBytes(eleTy.getIntOrFloatBitWidth());
  }
  if (ty.isIntOrFloat())
    return opt::convertBitsToBytes(ty.getIntOrFloatBitWidth());

  ty.dump();
  throw std::runtime_error("Expected a complex, floating, or integral type");
}

void cudaq::setDataLayout(MlirModule module) {
  auto mod = unwrap(module);
  if (mod->hasAttr(cudaq::opt::factory::targetDataLayoutAttrName))
    return;
  auto dataLayout = createDataLayout();
  mod->setAttr(cudaq::opt::factory::targetDataLayoutAttrName,
               StringAttr::get(mod->getContext(), dataLayout));
}

//===----------------------------------------------------------------------===//
// The section is the implementation of functions declared in OpaqueArguments.h
//===----------------------------------------------------------------------===//

py::args cudaq::simplifiedValidateInputArguments(py::args &args) {
  py::args processed = py::tuple(args.size());
  for (std::size_t i = 0; i < args.size(); ++i) {
    auto arg = args[i];
    // Check if it has tolist, so it might be a 1d buffer (array / numpy
    // ndarray)
    if (py::hasattr(args[i], "tolist")) {
      // This is a valid ndarray if it has tolist and shape
      if (!py::hasattr(args[i], "shape"))
        throw std::runtime_error(
            "Invalid input argument type, could not get shape of array.");

      // This is an ndarray with tolist() and shape attributes
      // get the shape and check its size
      auto shape = args[i].attr("shape").cast<py::tuple>();
      if (shape.size() != 1)
        throw std::runtime_error("Cannot pass ndarray with shape != (N,).");

      arg = args[i].attr("tolist")();
    } else if (py::isinstance<py::str>(arg)) {
      arg = py::cast<std::string>(arg);
    } else if (py::isinstance<py::list>(arg)) {
      py::list arg_list = py::cast<py::list>(arg);
      const bool all_strings = [&]() {
        for (auto &item : arg_list)
          if (!py::isinstance<py::str>(item))
            return false;
        return true;
      }();
      if (all_strings) {
        std::vector<cudaq::pauli_word> pw_list;
        pw_list.reserve(arg_list.size());
        for (auto &item : arg_list)
          pw_list.emplace_back(py::cast<std::string>(item));
        arg = std::move(pw_list);
      }
    }

    processed[i] = arg;
  }

  return processed;
}

std::pair<std::size_t, std::vector<std::size_t>>
cudaq::getTargetLayout(mlir::ModuleOp mod, cudaq::cc::StructType structTy) {
  mlir::StringRef dataLayoutSpec = "";
  if (auto attr = mod->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
    dataLayoutSpec = mlir::cast<mlir::StringAttr>(attr);
  else
    throw std::runtime_error("No data layout attribute is set on the module.");

  auto dataLayout = llvm::DataLayout(dataLayoutSpec);
  // Convert bufferTy to llvm.
  llvm::LLVMContext context;
  mlir::LLVMTypeConverter converter(structTy.getContext());
  cudaq::opt::initializeTypeConversions(converter);
  auto llvmDialectTy = converter.convertType(structTy);
  mlir::LLVM::TypeToLLVMIRTranslator translator(context);
  auto *llvmStructTy =
      mlir::cast<llvm::StructType>(translator.translateType(llvmDialectTy));
  auto *layout = dataLayout.getStructLayout(llvmStructTy);
  auto strSize = layout->getSizeInBytes();
  std::vector<std::size_t> fieldOffsets;
  for (std::size_t i = 0, I = structTy.getMembers().size(); i != I; ++i)
    fieldOffsets.emplace_back(layout->getElementOffset(i));
  return {strSize, fieldOffsets};
}

void cudaq::handleStructMemberVariable(void *data, std::size_t offset,
                                       mlir::Type memberType,
                                       py::object value) {
  auto appendValue = [](void *data, auto &&value, std::size_t offset) {
    std::memcpy(((char *)data) + offset, &value,
                sizeof(std::remove_cvref_t<decltype(value)>));
  };
  llvm::TypeSwitch<mlir::Type, void>(memberType)
      .Case([&](mlir::IntegerType ty) {
        if (ty.isInteger(1)) {
          appendValue(data, (bool)value.cast<py::bool_>(), offset);
          return;
        }
        appendValue(data, (std::int64_t)value.cast<py::int_>(), offset);
      })
      .Case([&](mlir::Float64Type ty) {
        appendValue(data, (double)value.cast<py::float_>(), offset);
      })
      .Case([&](cudaq::cc::StdvecType ty) {
        auto appendVectorValue = []<typename T>(py::object value, void *data,
                                                std::size_t offset, T) {
          auto asList = value.cast<py::list>();
          std::vector<double> *values = new std::vector<double>(asList.size());
          for (std::size_t i = 0; auto &v : asList)
            (*values)[i++] = v.cast<double>();

          std::memcpy(((char *)data) + offset, values, 16);
        };

        mlir::TypeSwitch<mlir::Type, void>(ty.getElementType())
            .Case([&](mlir::IntegerType type) {
              if (type.isInteger(1)) {
                appendVectorValue(value, data, offset, char());
                return;
              }
              appendVectorValue(value, data, offset, std::size_t());
            })
            .Case([&](mlir::FloatType type) {
              if (type.isF32()) {
                appendVectorValue(value, data, offset, float());
                return;
              }
              appendVectorValue(value, data, offset, double());
            });
      })
      .Default([&](mlir::Type ty) {
        ty.dump();
        throw std::runtime_error(
            "Type not supported for custom struct in kernel.");
      });
}

void *cudaq::handleVectorElements(mlir::Type eleTy, py::list list) {
  auto appendValue = []<typename T>(py::list list, auto &&converter) -> void * {
    std::vector<T> *values = new std::vector<T>(list.size());
    for (std::size_t i = 0; auto &v : list) {
      auto converted = converter(v, i);
      (*values)[i++] = converted;
    }
    return values;
  };

  return llvm::TypeSwitch<mlir::Type, void *>(eleTy)
      .Case([&](mlir::IntegerType ty) {
        if (ty.getIntOrFloatBitWidth() == 1)
          return appendValue.template operator()<char>(
              list, [](py::handle v, std::size_t i) {
                checkListElementType<py::bool_>(v, i);
                return v.cast<bool>();
              });
        if (ty.getIntOrFloatBitWidth() == 8)
          return appendValue.template operator()<std::int8_t>(
              list, [](py::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return v.cast<std::int8_t>();
              });
        if (ty.getIntOrFloatBitWidth() == 16)
          return appendValue.template operator()<std::int16_t>(
              list, [](py::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return v.cast<std::int16_t>();
              });
        if (ty.getIntOrFloatBitWidth() == 32)
          return appendValue.template operator()<std::int32_t>(
              list, [](py::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return v.cast<std::int32_t>();
              });
        return appendValue.template operator()<std::int64_t>(
            list, [](py::handle v, std::size_t i) {
              checkListElementType<py_ext::Int>(v, i);
              return v.cast<std::int64_t>();
            });
      })
      .Case([&](mlir::Float32Type ty) {
        return appendValue.template operator()<float>(
            list, [](py::handle v, std::size_t i) {
              checkListElementType<py_ext::Float>(v, i);
              return v.cast<float>();
            });
      })
      .Case([&](mlir::Float64Type ty) {
        return appendValue.template operator()<double>(
            list, [](py::handle v, std::size_t i) {
              checkListElementType<py_ext::Float>(v, i);
              return v.cast<double>();
            });
      })
      .Case([&](cudaq::cc::CharspanType type) {
        return appendValue.template operator()<std::string>(
            list, [](py::handle v, std::size_t i) {
              return v.cast<cudaq::pauli_word>().str();
            });
      })
      .Case([&](mlir::ComplexType type) {
        if (mlir::isa<mlir::Float64Type>(type.getElementType()))
          return appendValue.template operator()<std::complex<double>>(
              list, [](py::handle v, std::size_t i) {
                checkListElementType<py_ext::Complex>(v, i);
                return v.cast<std::complex<double>>();
              });
        return appendValue.template operator()<std::complex<float>>(
            list, [](py::handle v, std::size_t i) {
              checkListElementType<py_ext::Complex>(v, i);
              return v.cast<std::complex<float>>();
            });
      })
      .Case([&](cudaq::cc::StdvecType ty) {
        auto appendVectorValue = []<typename T>(mlir::Type eleTy,
                                                py::list list) -> void * {
          auto *values = new std::vector<std::vector<T>>();
          for (std::size_t i = 0; i < list.size(); i++) {
            auto ptr = handleVectorElements(eleTy, list[i]);
            auto *element = static_cast<std::vector<T> *>(ptr);
            values->emplace_back(std::move(*element));
          }
          return values;
        };

        auto eleTy = ty.getElementType();
        if (ty.getElementType().isInteger(1))
          // Special case for a `std::vector<bool>`.
          return appendVectorValue.template operator()<char>(eleTy, list);

        // All other `std::Vector<T>` types, including nested vectors.
        return appendVectorValue.template operator()<std::size_t>(eleTy, list);
      })
      .Default([&](mlir::Type ty) {
        throw std::runtime_error("invalid list element type (" +
                                 mlirTypeToString(ty) + ").");
        return nullptr;
      });
}

std::string cudaq::mlirTypeToString(mlir::Type ty) {
  std::string msg;
  {
    llvm::raw_string_ostream os(msg);
    ty.print(os);
  }
  return msg;
}

void cudaq::packArgs(OpaqueArguments &argData, py::list args,
                     mlir::ArrayRef<mlir::Type> mlirTys,
                     const std::function<bool(OpaqueArguments &, py::object &,
                                              unsigned)> &backupHandler,
                     mlir::func::FuncOp kernelFuncOp) {
  if (args.size() == 0)
    return;

  for (auto [i, zippy] : llvm::enumerate(llvm::zip(args, mlirTys))) {
    py::object arg = py::reinterpret_borrow<py::object>(std::get<0>(zippy));
    Type kernelArgTy = std::get<1>(zippy);
    llvm::TypeSwitch<Type, void>(kernelArgTy)
        .Case([&](ComplexType ty) {
          checkArgumentType<py_ext::Complex>(arg, i);
          if (isa<Float64Type>(ty.getElementType())) {
            addArgument(argData, arg.cast<std::complex<double>>());
          } else if (isa<Float32Type>(ty.getElementType())) {
            addArgument(argData, arg.cast<std::complex<float>>());
          } else {
            throw std::runtime_error("Invalid complex type argument: " +
                                     py::str(args).cast<std::string>() +
                                     " Type: " + mlirTypeToString(ty));
          }
        })
        .Case([&](Float64Type ty) {
          checkArgumentType<py_ext::Float>(arg, i);
          addArgument(argData, arg.cast<double>());
        })
        .Case([&](Float32Type ty) {
          checkArgumentType<py_ext::Float>(arg, i);
          addArgument(argData, arg.cast<float>());
        })
        .Case([&](IntegerType ty) {
          if (ty.getIntOrFloatBitWidth() == 1) {
            checkArgumentType<py::bool_>(arg, i);
            addArgument(argData, static_cast<char>(arg.cast<bool>()));
            return;
          }

          checkArgumentType<py_ext::Int>(arg, i);
          addArgument(argData, arg.cast<std::int64_t>());
        })
        .Case([&](cc::CharspanType ty) {
          addArgument(argData, arg.cast<pauli_word>().str());
        })
        .Case([&](cc::PointerType ty) {
          if (isa<quake::StateType>(ty.getElementType())) {
            argData.emplace_back(arg.cast<state *>(), [](void *ptr) {
              /* Do nothing, state is passed as reference */
            });
          } else {
            throw std::runtime_error("Invalid pointer type argument: " +
                                     py::str(arg).cast<std::string>() +
                                     " Type: " + mlirTypeToString(ty));
          }
        })
        .Case([&](cc::StructType ty) {
          auto mod = kernelFuncOp->getParentOfType<mlir::ModuleOp>();
          auto [size, offsets] = getTargetLayout(mod, ty);
          auto memberTys = ty.getMembers();
          auto allocatedArg = std::malloc(size);
          if (ty.getName() == "tuple") {
            auto elements = arg.cast<py::tuple>();
            for (std::size_t i = 0; i < offsets.size(); i++)
              handleStructMemberVariable(allocatedArg, offsets[i], memberTys[i],
                                         elements[i]);
          } else {
            py::dict attributes = arg.attr("__annotations__").cast<py::dict>();
            for (std::size_t i = 0;
                 const auto &[attr_name, unused] : attributes) {
              py::object attr_value =
                  arg.attr(attr_name.cast<std::string>().c_str());
              handleStructMemberVariable(allocatedArg, offsets[i], memberTys[i],
                                         attr_value);
              i++;
            }
          }
          argData.emplace_back(allocatedArg, [](void *ptr) { std::free(ptr); });
        })
        .Case([&](cc::StdvecType ty) {
          auto appendVectorValue = [&argData]<typename T>(Type eleTy,
                                                          py::list list) {
            auto allocatedArg = handleVectorElements(eleTy, list);
            argData.emplace_back(allocatedArg, [](void *ptr) {
              delete static_cast<std::vector<T> *>(ptr);
            });
          };

          checkArgumentType<py::list>(arg, i);
          auto list = py::cast<py::list>(arg);
          auto eleTy = ty.getElementType();
          if (eleTy.isInteger(1)) {
            // Special case for a `std::vector<bool>`.
            appendVectorValue.template operator()<char>(eleTy, list);
            return;
          }
          // All other `std::vector<T>` types, including nested vectors.
          appendVectorValue.template operator()<std::int64_t>(eleTy, list);
        })
        .Case([&](cc::CallableType ty) {
          // arg must be a DecoratorCapture object.
          checkArgumentType<py::object>(arg, i);
          py::object decorator = arg.attr("decorator");
          auto kernelName = decorator.attr("uniqName").cast<std::string>();
          auto kernelModule =
              unwrap(decorator.attr("qkeModule").cast<MlirModule>());
          auto calledFuncOp = kernelModule.lookupSymbol<func::FuncOp>(
              cudaq::runtime::cudaqGenPrefixName + kernelName);
          py::list arguments = arg.attr("resolved");
          auto startLiftedArgs = [&]() -> std::optional<unsigned> {
            if (!arguments.empty())
              return decorator.attr("firstLiftedPos").cast<unsigned>();
            return std::nullopt;
          }();
          // build the recursive closure in a C++ object
          auto *closure = [&]() {
            OpaqueArguments resolvedArgs;
            if (startLiftedArgs) {
              auto fnTy = calledFuncOp.getFunctionType();
              auto liftedTys = fnTy.getInputs().drop_front(*startLiftedArgs);
              packArgs(resolvedArgs, arguments, liftedTys, backupHandler,
                       calledFuncOp);
            }
            return new runtime::CallableClosureArgument(
                kernelName, kernelModule, std::move(startLiftedArgs),
                std::move(resolvedArgs));
          }();
          argData.emplace_back(closure, [](void *that) {
            delete static_cast<runtime::CallableClosureArgument *>(that);
          });
        })
        .Default([&](Type ty) {
          // See if we have a backup type handler.
          bool success = backupHandler(argData, arg, i);
          if (!success)
            throw std::runtime_error(
                "Could not pack argument: " + py::str(arg).cast<std::string>() +
                " Type: " + mlirTypeToString(ty));
        });
  }
}

void cudaq::packArgs(OpaqueArguments &argData, py::args args,
                     mlir::func::FuncOp kernelFuncOp,
                     const std::function<bool(OpaqueArguments &, py::object &,
                                              unsigned)> &backupHandler,
                     std::size_t startingArgIdx) {
  if (args.size() == 0) {
    // Nothing to pack. This may be a full QIR pre-compile, which is perfectly
    // legit. At any rate, there is nothing to pack so return.
    return;
  }

  if (kernelFuncOp.getNumArguments() != args.size())
    throw std::runtime_error("Invalid runtime arguments - kernel expected " +
                             std::to_string(kernelFuncOp.getNumArguments()) +
                             " but was provided " +
                             std::to_string(args.size()) + " arguments.");

  // Move the args to a list, lopping off startingArgIdx args from the front.
  py::list pyList;
  for (auto [i, h] : llvm::enumerate(args)) {
    if (i < startingArgIdx)
      continue;
    pyList.append(h);
  }
  return packArgs(
      argData, pyList,
      kernelFuncOp.getFunctionType().getInputs().drop_front(startingArgIdx),
      backupHandler, kernelFuncOp);
}

//===----------------------------------------------------------------------===//

/// Mechanical merge of a callable argument (captured in a python decorator)
/// when the call site is executed.
static bool linkResolvedCallable(ModuleOp currMod, func::FuncOp entryPoint,
                                 unsigned argPos, py::object arg) {
  if (!py::hasattr(arg, "qkeModule"))
    return false;
  auto uniqName = arg.attr("uniqName").cast<std::string>();
  auto otherModule = arg.attr("qkeModule").cast<MlirModule>();
  ModuleOp otherMod = unwrap(otherModule);
  std::string calleeName = cudaq::runtime::cudaqGenPrefixName + uniqName;
  auto callee = cudaq::getKernelFuncOp(otherModule, calleeName);
  // TODO: Consider just merging the declaration of the symbol instead of the
  // entire module here. Then leaning into the execution engine linking to the
  // correct LLVM code. Beware though! That only makes sense when the kernel is
  // already lowered to machine code and is available in-process (i.e., local
  // simulation).
  cudaq::opt::factory::mergeModules(currMod, otherMod);
  // Replace the `argPos`-th argument of `entryPoint`, which must be a
  // `cc.callable`, with the function constant with the symbol
  // `cudaqGenPrefixName` + `uniqName`.
  auto *ctx = currMod.getContext();
  OpBuilder builder(ctx);
  auto loc = entryPoint.getLoc();
  Block &entry = entryPoint.front();
  builder.setInsertionPoint(&entry.front());
  auto resolved = builder.create<func::ConstantOp>(
      loc, callee.getFunctionType(), calleeName);
  entry.getArgument(argPos).replaceAllUsesWith(resolved);
  return true;
}

/// @brief Create a new OpaqueArguments pointer and pack the python arguments
/// in it. Clients must delete the memory.
cudaq::OpaqueArguments *cudaq::toOpaqueArgs(py::args &args, MlirModule mod,
                                            const std::string &name) {
  auto kernelFunc = getKernelFuncOp(mod, name);
  auto *argData = new cudaq::OpaqueArguments();
  args = simplifiedValidateInputArguments(args);
  setDataLayout(mod);
  cudaq::packArgs(
      *argData, args, kernelFunc,
      [](OpaqueArguments &, py::object &, unsigned) { return false; });
  return argData;
}

/// Append result buffer to \p runtimeArgs.
/// The result buffer is a pointer to a preallocated heap location in which the
/// result value of the kernel is to be stored.
static void appendTheResultValue(ModuleOp module, const std::string &name,
                                 cudaq::OpaqueArguments &runtimeArgs,
                                 Type returnType) {
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
        runtimeArgs.emplace_back(
            ourAllocatedArg, [](void *ptr) { delete static_cast<vec *>(ptr); });
      })
      .Case([&](cudaq::cc::StructType ty) {
        auto [size, offsets] = cudaq::getTargetLayout(module, ty);
        auto ourAllocatedArg = std::malloc(size);
        runtimeArgs.emplace_back(ourAllocatedArg,
                                 [](void *ptr) { std::free(ptr); });
      })
      .Case([&](cudaq::cc::CallableType ty) {
        // Callables may not be returned from entry-point kernels. Append a
        // dummy value as a placeholder.
        runtimeArgs.emplace_back(nullptr, [](void *) {});
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
}

// Launching the module \p mod will modify its content, such as by argument
// synthesis into the entry-point kernel. Make a clone before we launch to
// preserve (cache) the IR, and erase the clone after the kernel is done.
static cudaq::KernelThunkResultType
pyLaunchModule(const std::string &name, ModuleOp mod,
               const std::vector<void *> &rawArgs, Type resultTy) {
  auto clone = mod.clone();
  auto res = cudaq::streamlinedLaunchModule(name, clone, rawArgs, resultTy);
  clone.erase();
  return res;
}

static bool isCurrentTargetFullQIR() {
  auto transport = getTransportLayer();
  // Biased. Most likely expected pattern first.
  return transport.starts_with("qir:") || transport == "qir" ||
         transport == "qir-full" || transport.starts_with("qir-full:");
}

static void pyAltLaunchAnalogKernel(const std::string &name,
                                    std::string &programArgs) {
  if (name.find(cudaq::runtime::cudaqAHKPrefixName) != 0)
    throw std::runtime_error("Unexpected type of kernel.");
  auto dynamicResult = cudaq::altLaunchKernel(
      name.c_str(), cudaq::KernelThunkType(nullptr),
      (void *)(const_cast<char *>(programArgs.c_str())), 0, 0);
  if (dynamicResult.data_buffer || dynamicResult.size)
    throw std::runtime_error("Not implemented: support dynamic results");
}

template <typename T>
py::object readPyObject(Type ty, char *arg) {
  std::size_t bytes = cudaq::byteSize(ty);
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

/// Convert bytes in buffer, \p data, which are the result of the kernel
/// launched to python object.
py::object cudaq::convertResult(ModuleOp module, Type ty, char *data) {
  auto isRunContext = module->hasAttr(runtime::enableCudaqRun);

  return TypeSwitch<Type, py::object>(ty)
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
      .Case([&](ComplexType ty) -> py::object {
        auto eleTy = ty.getElementType();
        return TypeSwitch<Type, py::object>(eleTy)
            .Case([&](Float64Type eTy) -> py::object {
              return readPyObject<std::complex<double>>(ty, data);
            })
            .Case([&](Float32Type eTy) -> py::object {
              return readPyObject<std::complex<float>>(ty, data);
            })
            .Default([](Type eTy) -> py::object {
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
          // Note: in the `cudaq::run` context the `std::vector<bool>` is
          // constructed in the host runtime by parsing the output log to
          // `std::vector<bool>`.
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
            list.append(convertResult(module, eleTy, i));
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
          list.append(convertResult(module, eleTy, v->data + i));
        return list;
      })
      .Case([&](cudaq::cc::StructType ty) -> py::object {
        auto name = ty.getName().str();
        // Handle tuples.
        if (name == "tuple") {
          auto [size, offsets] = getTargetLayout(module, ty);
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
            list.append(convertResult(module, eleTy, data + offsets[i]));
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
        auto [size, offsets] = getTargetLayout(module, ty);
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
                convertResult(module, eleTy, data + offsets[i]);
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

static const std::vector<void *> &
appendResultToArgsVector(cudaq::OpaqueArguments &runtimeArgs, Type returnType,
                         ModuleOp module, const std::string &name) {
  if (returnType && !isa<NoneType>(returnType))
    appendTheResultValue(module, name, runtimeArgs, returnType);
  return runtimeArgs.getArgs();
}

cudaq::KernelThunkResultType
cudaq::clean_launch_module(const std::string &name, ModuleOp mod, Type retTy,
                           cudaq::OpaqueArguments &args) {
  // Append space for a result, as needed, to the vector of arguments.
  auto rawArgs = appendResultToArgsVector(args, retTy, mod, name);
  Type resTy = isa<NoneType>(retTy) ? Type{} : retTy;
  return pyLaunchModule(name, mod, rawArgs, resTy);
}

cudaq::OpaqueArguments
cudaq::marshal_arguments_for_module_launch(ModuleOp mod, py::args runtimeArgs,
                                           func::FuncOp kernelFunc) {
  // Convert python arguments to opaque form.
  cudaq::OpaqueArguments args;
  cudaq::packArgs(
      args, runtimeArgs, kernelFunc,
      [&](cudaq::OpaqueArguments &args, py::object &pyArg, unsigned pos) {
        return linkResolvedCallable(mod, kernelFunc, pos, pyArg);
      });
  return args;
}

py::object cudaq::marshal_and_launch_module(const std::string &name,
                                            MlirModule module,
                                            MlirType returnType,
                                            py::args runtimeArgs) {
  ScopedTraceWithContext("marshal_and_launch_module", name);
  auto kernelFunc = getKernelFuncOp(module, name);
  auto mod = unwrap(module);
  Type retTy = unwrap(returnType);
  auto args = marshal_arguments_for_module_launch(mod, runtimeArgs, kernelFunc);
  [[maybe_unused]] auto resultPtr = clean_launch_module(name, mod, retTy, args);
  // FIXME: handle dynamic sized results!

  if (isa<NoneType>(retTy))
    return py::none();
  return cudaq::convertResult(mod, retTy,
                              reinterpret_cast<char *>(args.getArgs().back()));
}

// NB: `cachedEngine` is actually of type `mlir::ExecutionEngine**`.
static void *marshal_and_retain_module(const std::string &name,
                                       MlirModule module, MlirType returnType,
                                       void *cachedEngine,
                                       py::args runtimeArgs) {
  ScopedTraceWithContext("marshal_and_retain_module", name);
  if (!cachedEngine)
    throw std::runtime_error(
        "Must have a storage location to retain the ExecutionEngine provided");
  auto kernelFunc = cudaq::getKernelFuncOp(module, name);
  auto mod = unwrap(module);
  Type retTy = unwrap(returnType);
  auto args =
      cudaq::marshal_arguments_for_module_launch(mod, runtimeArgs, kernelFunc);
  // Append space for a result, as needed, to the vector of arguments.
  auto rawArgs = appendResultToArgsVector(args, retTy, mod, name);
  Type resTy = isa<NoneType>(retTy) ? Type{} : retTy;
  auto clone = mod.clone();
  // Returns the pointer to the JITted LLVM code for the entry point function.
  void *funcPtr = cudaq::streamlinedSpecializeModule(name, clone, rawArgs,
                                                     resTy, cachedEngine);
  clone.erase();
  return funcPtr;
}

static MlirModule synthesizeKernel(py::object kernel, py::args runtimeArgs) {
  auto module = kernel.attr("qkeModule").cast<MlirModule>();
  auto mod = unwrap(module);
  auto name = kernel.attr("uniqName").cast<std::string>();
  if (mod->hasAttr(cudaq::runtime::pythonUniqueAttrName)) {
    StringRef n =
        cast<StringAttr>(mod->getAttr(cudaq::runtime::pythonUniqueAttrName));
    name = n.str();
  }
  auto kernelFuncOp = cudaq::getKernelFuncOp(module, name);
  cudaq::OpaqueArguments args;
  cudaq::setDataLayout(module);
  cudaq::packArgs(
      args, runtimeArgs, kernelFuncOp,
      [](cudaq::OpaqueArguments &, py::object &, unsigned) { return false; });

  ScopedTraceWithContext(cudaq::TIMING_JIT, "synthesizeKernel", name);
  auto rawArgs = appendResultToArgsVector(args, {}, mod, name);
  auto cloned = mod.clone();
  auto context = cloned.getContext();
  registerLLVMDialectTranslation(*context);

  // Get additional debug values
  auto disableMLIRthreading =
      cudaq::getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", false);
  auto enablePrintMLIREachPass =
      cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);

  auto &platform = cudaq::get_platform();
  auto isRemoteSimulator = platform.get_remote_capabilities().isRemoteSimulator;
  auto isLocalSimulator = platform.is_simulator() && !platform.is_emulated();
  auto isSimulator = isLocalSimulator || isRemoteSimulator;

  cudaq::opt::ArgumentConverter argCon(name, mod);
  argCon.gen(args.getArgs());

  // Store kernel and substitution strings on the stack.
  // We pass string references to the `createArgumentSynthesisPass`.
  SmallVector<std::string> kernels;
  SmallVector<std::string> substs;
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
  SmallVector<StringRef> kernelRefs{kernels.begin(), kernels.end()};
  SmallVector<StringRef> substRefs{substs.begin(), substs.end()};

  PassManager pm(context);
  pm.addPass(cudaq::opt::createArgumentSynthesisPass(
      kernelRefs, substRefs, /*changeSemantics=*/false));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());

  // Run state preparation for quantum devices (or their emulation) only.
  // Simulators have direct implementation of state initialization
  // in their runtime.
  if (!isSimulator) {
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createConstantPropagation());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createLiftArrayAlloc());
    pm.addPass(cudaq::opt::createGlobalizeArrayValues());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopUnroll());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());

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
  if (disableMLIRthreading || enablePrintMLIREachPass)
    context->disableMultithreading();
  if (enablePrintMLIREachPass)
    pm.enableIRPrinting();
  if (failed(pm.run(cloned))) {
    engine.eraseHandler(handlerId);
    throw std::runtime_error(
        "failed to JIT compile the Quake representation\n" + error_msg);
  }
  timingScope.stop();
  engine.eraseHandler(handlerId);
  return wrap(cloned);
}

static void executeMLIRPassManager(ModuleOp mod, PassManager &pm) {
  auto enablePrintMLIREachPass =
      cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);
  auto context = mod.getContext();
  if (enablePrintMLIREachPass) {
    context->disableMultithreading();
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

  if (failed(pm.run(mod))) {
    engine.eraseHandler(handlerId);
    throw std::runtime_error(
        "failed to JIT compile the Quake representation\n" + error_msg);
  }
  timingScope.stop();
  engine.eraseHandler(handlerId);
}

static ModuleOp cleanLowerToCodegenKernel(ModuleOp mod,
                                          const cudaq::OpaqueArguments &args) {
  if (false && isCurrentTargetFullQIR() && args.empty()) {
    // Generate a portable full QIR module with no argument synthesis. All
    // arguments will be resolved and marshaled at the kernel call site.
    auto *ctx = mod.getContext();
    PassManager pm(ctx);
    std::string transport = getTransportLayer();
    cudaq::opt::addAOTPipelineConvertToQIR(pm, transport);
    executeMLIRPassManager(mod, pm);
    return mod;
  }
  // Optionally we will run the JIT as specified by the current platform. This
  // is the late form of JIT compilation, which will be done (or not) when we
  // dispatch (launch module) to the platform.
  return mod;
}

static MlirModule lower_to_codegen(const std::string &kernelName,
                                   MlirModule module, py::args runtimeArgs) {
  auto kernelFunc = cudaq::getKernelFuncOp(module, kernelName);
  cudaq::OpaqueArguments args;
  auto mod = unwrap(module);
  cudaq::packArgs(
      args, runtimeArgs, kernelFunc,
      [&](cudaq::OpaqueArguments &args, py::object &pyArg, unsigned pos) {
        return linkResolvedCallable(mod, kernelFunc, pos, pyArg);
      });
  return wrap(cleanLowerToCodegenKernel(mod, args));
}

static std::size_t get_launch_args_required(MlirModule module,
                                            const std::string &entryPointName) {
  auto entryPointKernel = cudaq::getKernelFuncOp(module, entryPointName);
  if (!entryPointKernel || entryPointKernel.empty())
    throw std::runtime_error(entryPointName + " must be present in module");
  Block &entry = entryPointKernel.front();
  std::size_t result = 0;
  // For each argument, count the ones that have uses.
  for (auto blkArg : entry.getArguments())
    if (!blkArg.getUses().empty())
      ++result;
  return result;
}

void cudaq::bindAltLaunchKernel(py::module &mod,
                                std::function<std::string()> &&getTL) {
  jitCache = std::make_unique<JITExecutionCache>();
  getTransportLayer = std::move(getTL);

  mod.def("lower_to_codegen", lower_to_codegen,
          "Lower a kernel module to CC dialect. Never launches the kernel.");

  mod.def("clean_launch_module", cudaq::clean_launch_module,
          "Launch a kernel. Does not perform other mischief.");
  mod.def("marshal_and_launch_module", cudaq::marshal_and_launch_module,
          "Launch a kernel. Marshaling of arguments and unmarshalling of "
          "results is performed.");
  mod.def("marshal_and_retain_module", marshal_and_retain_module,
          "Marshaling of arguments and unmarshalling of results is performed. "
          "The kernel undergoes argument synthesis and final code generation. "
          "The kernel is NOT executed, but rather cached to a location managed "
          "by the calling code. This allows the calling code to invoke the "
          "entry point with a regular C++ call.");

  mod.def("pyAltLaunchAnalogKernel", pyAltLaunchAnalogKernel,
          "Launch an analog Hamiltonian simulation kernel with given JSON "
          "payload.");

  mod.def("synthesize", synthesizeKernel, "FIXME: document!");

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
      [](MlirModule modA, MlirModule modB) {
        auto moduleA = unwrap(modA).clone();
        cudaq::opt::factory::mergeModules(moduleA, unwrap(modB));
        return wrap(moduleA);
      },
      "Merge the two Modules into a single Module.");

  mod.def(
      "mergeMLIRString",
      [](MlirModule modA, const std::string &text) {
        auto moduleA = unwrap(modA).clone();
        auto *ctx = moduleA.getContext();
        auto moduleB = mlir::parseSourceString<mlir::ModuleOp>(text, ctx);
        auto modB = moduleB.get();
        if (!modB)
          throw std::runtime_error("could not translate text");
        cudaq::opt::factory::mergeModules(moduleA, modB);
        return wrap(moduleA);
      },
      "Merge the first Module and the Quake text into a single new Module.");

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
      "get_launch_args_required",
      [](MlirModule mod, const std::string &shortName) {
        return get_launch_args_required(mod, shortName);
      },
      "Determine the number of formal arguments to the entry-point kernel that"
      "are used in the function.");

  mod.def(
      "is_current_target_full_qir",
      []() -> bool { return isCurrentTargetFullQIR(); },
      "Determine if the current selected target in the Python interpreter uses "
      "full QIR as the transport layer.");

  mod.def(
      "set_data_layout", [](MlirModule mod) { cudaq::setDataLayout(mod); },
      "Set the data layout on the module.");
}
