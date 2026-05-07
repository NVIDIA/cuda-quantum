/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_alt_launch_kernel.h"
#include "common/AnalogHamiltonian.h"
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
#include "cudaq/platform/nvqpp_interface.h"
#include "cudaq/platform/qpu.h"
#include "cudaq_internal/compiler/ArgumentConversion.h"
#include "cudaq_internal/compiler/LayoutInfo.h"
#include "cudaq_internal/compiler/TracePassInstrumentation.h"
#include "runtime/cudaq/algorithms/py_utils.h"
#include "runtime/cudaq/platform/PythonSignalCheck.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"
#include "utils/PyTypes.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

using namespace mlir;

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
  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target)
    throw std::runtime_error("Cannot create target");

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  auto hostFeatures = llvm::sys::getHostCPUFeatures();
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

nanobind::args cudaq::simplifiedValidateInputArguments(nanobind::args &args) {
  nanobind::args processed =
      nanobind::steal<nanobind::args>(PyTuple_New((Py_ssize_t)args.size()));
  for (std::size_t i = 0; i < args.size(); ++i) {
    nanobind::object arg = nanobind::borrow(args[i]);
    // Check if it has tolist, so it might be a 1d buffer (array / numpy
    // ndarray)
    if (nanobind::hasattr(args[i], "tolist")) {
      // This is a valid ndarray if it has tolist and shape
      if (!nanobind::hasattr(args[i], "shape"))
        throw std::runtime_error(
            "Invalid input argument type, could not get shape of array.");

      // This is an ndarray with tolist() and shape attributes
      // get the shape and check its size
      auto shape = nanobind::cast<nanobind::tuple>(args[i].attr("shape"));
      if (shape.size() != 1)
        throw std::runtime_error("Cannot pass ndarray with shape != (N,).");

      arg = args[i].attr("tolist")();
    } else if (nanobind::isinstance<nanobind::str>(arg)) {
      arg = nanobind::cast(nanobind::cast<std::string>(arg));
    } else if (nanobind::isinstance<nanobind::list>(arg)) {
      nanobind::list arg_list = nanobind::cast<nanobind::list>(arg);
      const bool all_strings = [&]() {
        for (auto item : arg_list)
          if (!nanobind::isinstance<nanobind::str>(item))
            return false;
        return true;
      }();
      if (all_strings) {
        std::vector<cudaq::pauli_word> pw_list;
        pw_list.reserve(arg_list.size());
        for (auto item : arg_list)
          pw_list.emplace_back(nanobind::cast<std::string>(item));
        arg = nanobind::cast(std::move(pw_list));
      }
    }

    PyTuple_SET_ITEM(processed.ptr(), (Py_ssize_t)i, arg.inc_ref().ptr());
  }

  return processed;
}

template <cudaq::PackingStyle style>
void cudaq::handleStructMemberVariable(void *data, std::size_t offset,
                                       mlir::Type memberType,
                                       nanobind::object value) {
  auto appendValue = [](void *data, auto &&value, std::size_t offset) {
    std::memcpy(((char *)data) + offset, &value,
                sizeof(std::remove_cvref_t<decltype(value)>));
  };
  llvm::TypeSwitch<mlir::Type, void>(memberType)
      .Case([&](mlir::IntegerType ty) {
        if (ty.isInteger(1)) {
          appendValue(data, nanobind::cast<bool>(value), offset);
          return;
        }
        appendValue(data, nanobind::cast<std::int64_t>(value), offset);
      })
      .Case([&](mlir::Float64Type ty) {
        appendValue(data, nanobind::cast<double>(value), offset);
      })
      .Case([&](cudaq::cc::StdvecType ty) {
        auto appendVectorValue = []<typename T>(nanobind::object value,
                                                void *data, std::size_t offset,
                                                T) {
          auto asList = nanobind::cast<nanobind::list>(value);
          // Use the correct element type T (not always double).
          auto *values = new std::vector<T>(asList.size());
          for (std::size_t i = 0; auto v : asList)
            (*values)[i++] = nanobind::cast<T>(v);

          // synthesis path: span {ptr, size_t}
          // argsCreator path: std::vector<T> {ptr, ptr, ptr}
          constexpr std::size_t copySize =
              sizeof(std::conditional_t<style == cudaq::PackingStyle::synthesis,
                                        std::pair<char *, std::size_t>,
                                        std::vector<T>>);
          std::memcpy(((char *)data) + offset, values, copySize);
        };

        mlir::TypeSwitch<mlir::Type, void>(ty.getElementType())
            .Case([&](mlir::IntegerType type) {
              if (type.isInteger(1)) {
                appendVectorValue(value, data, offset, BoolVecElem<style>{});
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
            })
            .Case([&](cudaq::cc::StdvecType innerVecType) {
              if constexpr (style == cudaq::PackingStyle::synthesis) {
                throw std::runtime_error(
                    "Type not supported for custom struct in kernel.");
              } else {
                // Nested vector (e.g., list[list[int]]): delegate to
                // handleVectorElements which handles the recursive case.
                auto asList = nanobind::cast<nanobind::list>(value);
                auto *values =
                    handleVectorElements<cudaq::PackingStyle::argsCreator>(
                        innerVecType, asList);
                std::memcpy(((char *)data) + offset, values,
                            sizeof(std::vector<std::vector<std::size_t>>));
              }
            });
      })
      .Default([&](mlir::Type ty) {
        ty.dump();
        throw std::runtime_error(
            "Type not supported for custom struct in kernel.");
      });
}

template <cudaq::PackingStyle style>
void *cudaq::handleVectorElements(mlir::Type eleTy, nanobind::list list) {
  auto appendValue = []<typename T>(nanobind::list list,
                                    auto &&converter) -> void * {
    std::vector<T> *values = new std::vector<T>(list.size());
    for (std::size_t i = 0; auto v : list) {
      auto converted = converter(v, i);
      (*values)[i++] = converted;
    }
    return values;
  };

  return llvm::TypeSwitch<mlir::Type, void *>(eleTy)
      .Case([&](mlir::IntegerType ty) {
        if (ty.getIntOrFloatBitWidth() == 1) {
          return appendValue.template operator()<BoolVecElem<style>>(
              list, [](nanobind::handle v, std::size_t i) {
                checkListElementType<nanobind::bool_>(v, i);
                return static_cast<BoolVecElem<style>>(nanobind::cast<bool>(v));
              });
        }
        if (ty.getIntOrFloatBitWidth() == 8)
          return appendValue.template operator()<std::int8_t>(
              list, [](nanobind::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return nanobind::cast<std::int8_t>(v);
              });
        if (ty.getIntOrFloatBitWidth() == 16)
          return appendValue.template operator()<std::int16_t>(
              list, [](nanobind::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return nanobind::cast<std::int16_t>(v);
              });
        if (ty.getIntOrFloatBitWidth() == 32)
          return appendValue.template operator()<std::int32_t>(
              list, [](nanobind::handle v, std::size_t i) {
                checkListElementType<py_ext::Int>(v, i);
                return nanobind::cast<std::int32_t>(v);
              });
        return appendValue.template operator()<std::int64_t>(
            list, [](nanobind::handle v, std::size_t i) {
              checkListElementType<py_ext::Int>(v, i);
              return nanobind::cast<std::int64_t>(v);
            });
      })
      .Case([&](mlir::Float32Type ty) {
        return appendValue.template operator()<float>(
            list, [](nanobind::handle v, std::size_t i) {
              checkListElementType<py_ext::Float>(v, i);
              return nanobind::cast<float>(v);
            });
      })
      .Case([&](mlir::Float64Type ty) {
        return appendValue.template operator()<double>(
            list, [](nanobind::handle v, std::size_t i) {
              checkListElementType<py_ext::Float>(v, i);
              return nanobind::cast<double>(v);
            });
      })
      .Case([&](cudaq::cc::CharspanType type) {
        return appendValue.template operator()<std::string>(
            list, [](nanobind::handle v, std::size_t i) {
              return nanobind::cast<cudaq::pauli_word>(v).str();
            });
      })
      .Case([&](mlir::ComplexType type) {
        if (mlir::isa<mlir::Float64Type>(type.getElementType()))
          return appendValue.template operator()<std::complex<double>>(
              list, [](nanobind::handle v, std::size_t i) {
                checkListElementType<py_ext::Complex>(v, i);
                return nanobind::cast<std::complex<double>>(v);
              });
        return appendValue.template operator()<std::complex<float>>(
            list, [](nanobind::handle v, std::size_t i) {
              checkListElementType<py_ext::Complex>(v, i);
              return nanobind::cast<std::complex<float>>(v);
            });
      })
      .Case([&](cudaq::cc::StdvecType ty) {
        auto appendVectorValue = []<typename T>(mlir::Type eleTy,
                                                nanobind::list list) -> void * {
          auto *values = new std::vector<std::vector<T>>();
          for (std::size_t i = 0; i < list.size(); i++) {
            auto ptr = handleVectorElements<style>(eleTy, list[i]);
            auto *element = static_cast<std::vector<T> *>(ptr);
            values->emplace_back(std::move(*element));
          }
          return values;
        };

        auto eleTy = ty.getElementType();
        if (ty.getElementType().isInteger(1)) {
          // Special case for a `std::vector<bool>`.
          return appendVectorValue.template operator()<BoolVecElem<style>>(
              eleTy, list);
        }

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

template <cudaq::PackingStyle style>
void cudaq::packArgs(
    OpaqueArguments &argData, nanobind::list args,
    mlir::ArrayRef<mlir::Type> mlirTys,
    const std::function<bool(OpaqueArguments &, nanobind::object &, unsigned)>
        &backupHandler,
    mlir::func::FuncOp kernelFuncOp) {
  if (args.size() == 0)
    return;

  for (auto [i, zippy] : llvm::enumerate(llvm::zip(args, mlirTys))) {
    nanobind::object arg =
        nanobind::borrow<nanobind::object>(std::get<0>(zippy));
    Type kernelArgTy = std::get<1>(zippy);
    if (arg.is_none()) {
      argData.emplace_back(nullptr, [](void *ptr) {});
      continue;
    }
    llvm::TypeSwitch<Type, void>(kernelArgTy)
        .Case([&](ComplexType ty) {
          checkArgumentType<py_ext::Complex>(arg, i);
          if (isa<Float64Type>(ty.getElementType())) {
            addArgument(argData, nanobind::cast<std::complex<double>>(arg));
          } else if (isa<Float32Type>(ty.getElementType())) {
            addArgument(argData, nanobind::cast<std::complex<float>>(arg));
          } else {
            throw std::runtime_error(
                "Invalid complex type argument: " +
                nanobind::cast<std::string>(
                    nanobind::steal(PyObject_Str(args.ptr()))) +
                " Type: " + mlirTypeToString(ty));
          }
        })
        .Case([&](Float64Type ty) {
          checkArgumentType<py_ext::Float>(arg, i);
          addArgument(argData, nanobind::cast<double>(arg));
        })
        .Case([&](Float32Type ty) {
          checkArgumentType<py_ext::Float>(arg, i);
          addArgument(argData, nanobind::cast<float>(arg));
        })
        .Case([&](IntegerType ty) {
          if (ty.getIntOrFloatBitWidth() == 1) {
            checkArgumentType<nanobind::bool_>(arg, i);
            addArgument(argData, static_cast<BoolVecElem<style>>(
                                     nanobind::cast<bool>(arg)));
            return;
          }

          checkArgumentType<py_ext::Int>(arg, i);
          addArgument(argData, nanobind::cast<std::int64_t>(arg));
        })
        .Case([&](cc::CharspanType ty) {
          addArgument(argData, nanobind::cast<pauli_word>(arg).str());
        })
        .Case([&](cc::PointerType ty) {
          if (isa<quake::StateType>(ty.getElementType())) {
            auto *stateArg = nanobind::cast<state *>(arg);

            if (stateArg == nullptr)
              throw std::runtime_error("Null cudaq::state* argument passed.");
            auto simState = cudaq::state_helper::getSimulationState(
                const_cast<cudaq::state *>(stateArg));
            if (!simState)
              throw std::runtime_error("Error: Unable to retrieve simulation "
                                       "state from cudaq::state. The state "
                                       "contains no simulation state.");
            if (simState->getKernelInfo().has_value()) {
              // For state arguments represented by a kernel, we need to make a
              // copy of the state since this state is lazily evaluated. Note:
              // the state that holds the kernel info also holds ownership of
              // the packed arguments, hence the unravelling the correct
              // arguments when evaluated.
              state *copyState = new state(*stateArg);
              argData.emplace_back(copyState, [](void *ptr) {
                delete static_cast<state *>(ptr);
              });
            } else {
              argData.emplace_back(
                  stateArg,
                  [](void *ptr) { /* do nothing, we don't own the state */ });
            }
          } else {
            throw std::runtime_error(
                "Invalid pointer type argument: " +
                nanobind::cast<std::string>(
                    nanobind::steal(PyObject_Str(arg.ptr()))) +
                " Type: " + mlirTypeToString(ty));
          }
        })
        .Case([&](cc::StructType ty) {
          auto mod = kernelFuncOp->getParentOfType<mlir::ModuleOp>();
          cc::StructType layoutTy = ty;
          if constexpr (style == cudaq::PackingStyle::argsCreator)
            layoutTy = cast<cc::StructType>(
                cudaq::opt::factory::convertToHostSideType(ty, mod));
          auto [size, offsets] =
              cudaq_internal::compiler::getTargetLayout(mod, layoutTy);
          auto memberTys = ty.getMembers();
          auto allocatedArg = std::malloc(size);
          if (ty.getName() == "tuple") {
            auto elements = nanobind::cast<nanobind::tuple>(arg);
            for (std::size_t i = 0; i < offsets.size(); i++)
              handleStructMemberVariable<style>(allocatedArg, offsets[i],
                                                memberTys[i], elements[i]);
          } else {
            nanobind::dict attributes =
                nanobind::cast<nanobind::dict>(arg.attr("__annotations__"));
            for (std::size_t i = 0;
                 const auto &[attr_name, unused] : attributes) {
              nanobind::object attr_value =
                  arg.attr(nanobind::cast<std::string>(attr_name).c_str());
              handleStructMemberVariable<style>(allocatedArg, offsets[i],
                                                memberTys[i], attr_value);
              i++;
            }
          }
          argData.emplace_back(allocatedArg, [](void *ptr) { std::free(ptr); });
        })
        .Case([&](cc::StdvecType ty) {
          auto appendVectorValue = [&argData]<typename T>(Type eleTy,
                                                          nanobind::list list) {
            auto allocatedArg = handleVectorElements<style>(eleTy, list);
            argData.emplace_back(allocatedArg, [](void *ptr) {
              delete static_cast<std::vector<T> *>(ptr);
            });
          };

          checkArgumentType<nanobind::list>(arg, i);
          auto list = nanobind::cast<nanobind::list>(arg);
          auto eleTy = ty.getElementType();
          if (eleTy.isInteger(1)) {
            // Special case for a `std::vector<bool>`.
            appendVectorValue.template operator()<BoolVecElem<style>>(eleTy,
                                                                      list);
            return;
          }
          // All other `std::vector<T>` types, including nested vectors.
          appendVectorValue.template operator()<std::int64_t>(eleTy, list);
        })
        .Case([&](cc::CallableType ty) {
          // arg must be a DecoratorCapture object.
          checkArgumentType<nanobind::object>(arg, i);
          if (nanobind::hasattr(arg, "linkedKernel")) {
            auto kernelName =
                nanobind::cast<std::string>(arg.attr("linkedKernel"));
            // TODO: This is kinda yucky to have to remove because it's already
            // present
            kernelName.erase(0, strlen(cudaq::runtime::cudaqGenPrefixName));
            auto kernelModule =
                unwrap(nanobind::cast<MlirModule>(arg.attr("qkeModule")));
            OpaqueArguments resolvedArgs;
            argData.emplace_back(
                new runtime::CallableClosureArgument(kernelName, kernelModule,
                                                     std::nullopt,
                                                     std::move(resolvedArgs)),
                [](void *that) {
                  delete static_cast<runtime::CallableClosureArgument *>(that);
                });
          } else {
            nanobind::object decorator = arg.attr("decorator");
            auto kernelName =
                nanobind::cast<std::string>(decorator.attr("uniqName"));
            auto kernelModule =
                unwrap(nanobind::cast<MlirModule>(decorator.attr("qkeModule")));
            auto calledFuncOp = kernelModule.lookupSymbol<func::FuncOp>(
                cudaq::runtime::cudaqGenPrefixName + kernelName);
            nanobind::list arguments = arg.attr("resolved");
            auto startLiftedArgs = [&]() -> std::optional<unsigned> {
              if (!arguments.empty())
                return nanobind::cast<unsigned>(
                    decorator.attr("formal_arity")());
              return std::nullopt;
            }();
            // build the recursive closure in a C++ object
            auto *closure = [&]() {
              OpaqueArguments resolvedArgs;
              if (startLiftedArgs) {
                auto fnTy = calledFuncOp.getFunctionType();
                auto liftedTys = fnTy.getInputs().drop_front(*startLiftedArgs);
                packArgs<style>(resolvedArgs, arguments, liftedTys,
                                backupHandler, calledFuncOp);
              }
              return new runtime::CallableClosureArgument(
                  kernelName, kernelModule, std::move(startLiftedArgs),
                  std::move(resolvedArgs));
            }();
            argData.emplace_back(closure, [](void *that) {
              delete static_cast<runtime::CallableClosureArgument *>(that);
            });
          }
        })
        .Default([&](Type ty) {
          // See if we have a backup type handler.
          bool success = backupHandler(argData, arg, i);
          if (!success)
            throw std::runtime_error(
                "Could not pack argument: " +
                nanobind::cast<std::string>(
                    nanobind::steal(PyObject_Str(arg.ptr()))) +
                " Type: " + mlirTypeToString(ty));
        });
  }
}

template <cudaq::PackingStyle style>
void cudaq::packArgs(
    OpaqueArguments &argData, nanobind::args args,
    mlir::func::FuncOp kernelFuncOp,
    const std::function<bool(OpaqueArguments &, nanobind::object &, unsigned)>
        &backupHandler,
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
  nanobind::list pyList;
  for (auto [i, h] : llvm::enumerate(args)) {
    if (i < startingArgIdx)
      continue;
    pyList.append(h);
  }
  return packArgs<style>(
      argData, pyList,
      kernelFuncOp.getFunctionType().getInputs().drop_front(startingArgIdx),
      backupHandler, kernelFuncOp);
}

//===----------------------------------------------------------------------===//

/// Mechanical merge of a callable argument (captured in a python decorator)
/// when the call site is executed.
static bool linkResolvedCallable(ModuleOp currMod, func::FuncOp entryPoint,
                                 unsigned argPos, nanobind::object arg) {
  if (!nanobind::hasattr(arg, "qkeModule"))
    return false;
  auto uniqName = nanobind::cast<std::string>(arg.attr("uniqName"));
  auto otherModule = nanobind::cast<MlirModule>(arg.attr("qkeModule"));
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
  auto resolved = func::ConstantOp::create(
      builder, loc, callee.getFunctionType(), calleeName);
  entry.getArgument(argPos).replaceAllUsesWith(resolved);
  return true;
}

/// @brief Create a new OpaqueArguments pointer and pack the python arguments
/// in it. Clients must delete the memory.
cudaq::OpaqueArguments *cudaq::toOpaqueArgs(nanobind::args &args,
                                            MlirModule mod,
                                            const std::string &name) {
  auto kernelFunc = getKernelFuncOp(mod, name);
  auto *argData = new cudaq::OpaqueArguments();
  args = simplifiedValidateInputArguments(args);
  setDataLayout(mod);
  cudaq::packArgs(
      *argData, args, kernelFunc,
      [](OpaqueArguments &, nanobind::object &, unsigned) { return false; });
  return argData;
}

/// Append result buffer to \p runtimeArgs.
/// The result buffer is a pointer to a preallocated heap location in which the
/// result value of the kernel is to be stored.
static void appendTheResultValue(ModuleOp module, const std::string &name,
                                 cudaq::OpaqueArguments &runtimeArgs,
                                 Type returnType) {
  auto [bufferSize, offsets] =
      cudaq_internal::compiler::getResultBufferLayout(module, returnType);
  if (bufferSize == 0)
    return;
  auto *buf = std::calloc(1, bufferSize);
  runtimeArgs.emplace_back(buf, [](void *ptr) { std::free(ptr); });
}

// Launching the module \p mod will modify its content, such as by argument
// synthesis into the entry-point kernel. Make a clone before we launch to
// preserve (cache) the IR, and erase the clone after the kernel is done.
static cudaq::KernelThunkResultType
pyLaunchModule(const std::string &name, ModuleOp mod,
               const std::vector<void *> &rawArgs) {
  auto clone = mod.clone();
  auto compiled = cudaq::streamlinedCompileModule(name, clone, rawArgs, true);
  auto res = cudaq::streamlinedLaunchModule(compiled, rawArgs);
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
nanobind::object readPyObject(Type ty, char *arg) {
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
nanobind::object cudaq::convertResult(ModuleOp module, Type ty, char *data) {
  auto isRunContext = module->hasAttr(runtime::enableCudaqRun);

  return TypeSwitch<Type, nanobind::object>(ty)
      .Case([&](IntegerType ty) -> nanobind::object {
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
      .Case([&](ComplexType ty) -> nanobind::object {
        auto eleTy = ty.getElementType();
        return TypeSwitch<Type, nanobind::object>(eleTy)
            .Case([&](Float64Type eTy) -> nanobind::object {
              return readPyObject<std::complex<double>>(ty, data);
            })
            .Case([&](Float32Type eTy) -> nanobind::object {
              return readPyObject<std::complex<float>>(ty, data);
            })
            .Default([](Type eTy) -> nanobind::object {
              eTy.dump();
              throw std::runtime_error(
                  "Unsupported float element type for complex type return.");
            });
      })
      .Case([&](Float64Type ty) -> nanobind::object {
        return readPyObject<double>(ty, data);
      })
      .Case([&](Float32Type ty) -> nanobind::object {
        return readPyObject<float>(ty, data);
      })
      .Case([&](cudaq::cc::StdvecType ty) -> nanobind::object {
        auto eleTy = ty.getElementType();
        // Nested StdvecType elements have a different in-memory size than
        // scalar types: span ({ptr,size_t} = 16 bytes) in direct-call context,
        // std::vector ({ptr,ptr,ptr} = 24 bytes) in run context.
        auto getEleByteSize = [&](Type eTy) -> std::size_t {
          if (isa<cudaq::cc::StdvecType>(eTy))
            return isRunContext ? 3 * sizeof(void *)
                                : sizeof(char *) + sizeof(std::size_t);
          return byteSize(eTy);
        };

        if (isRunContext) {
          // cudaq.run return.
          auto eleByteSize = getEleByteSize(eleTy);

          // Vector of booleans has a special layout.
          // Read the vector and create a list of booleans.
          // Note: in the `cudaq::run` context the `std::vector<bool>` is
          // constructed in the host runtime by parsing the output log to
          // `std::vector<bool>`.
          if (eleTy.isInteger(1)) {
            auto v = reinterpret_cast<std::vector<bool> *>(data);
            nanobind::list list;
            for (auto const bit : *v)
              list.append(nanobind::bool_(bit));
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
          nanobind::list list;
          for (char *i = v->begin; i < v->end; i += eleByteSize)
            list.append(convertResult(module, eleTy, i));
          return list;
        }

        // Direct call return.
        auto eleByteSize = getEleByteSize(eleTy);

        // Vector is a span: `{ data, length }`.
        // Read `data` and `length` from the buffer.
        struct vec {
          char *data;
          std::size_t length;
        };
        auto v = reinterpret_cast<vec *>(data);

        // Read vector elements.
        nanobind::list list;
        std::size_t byteLength = v->length * eleByteSize;
        for (std::size_t i = 0; i < byteLength; i += eleByteSize)
          list.append(convertResult(module, eleTy, v->data + i));
        return list;
      })
      .Case([&](cudaq::cc::StructType ty) -> nanobind::object {
        auto name = ty.getName().str();
        // Handle tuples.
        if (name == "tuple") {
          auto [size, offsets] =
              cudaq_internal::compiler::getTargetLayout(module, ty);
          auto memberTys = ty.getMembers();
          nanobind::list list;
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
          return nanobind::tuple(list);
        }

        // Handle data class objects.
        if (!DataClassRegistry::isRegisteredClass(name))
          throw std::runtime_error("Dataclass is not registered: " + name);

        // Find class information.
        auto [cls, attributes] = DataClassRegistry::getClassAttributes(name);

        // Collect field names.
        std::vector<nanobind::str> fieldNames;
        for (const auto &[attr_name, unused] : attributes)
          fieldNames.emplace_back(nanobind::str(attr_name));

        // Read field values and create the constructor `kwargs`
        auto [size, offsets] =
            cudaq_internal::compiler::getTargetLayout(module, ty);
        auto memberTys = ty.getMembers();
        nanobind::dict kwargs;
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
      .Default([](Type ty) -> nanobind::object {
        ty.dump();
        throw std::runtime_error("Unsupported return type.");
      });
}

static const std::vector<void *> &
appendResultToArgsVector(cudaq::OpaqueArguments &runtimeArgs, Type returnType,
                         ModuleOp module, const std::string &name) {
  if (returnType)
    appendTheResultValue(module, name, runtimeArgs, returnType);
  return runtimeArgs.getArgs();
}

cudaq::KernelThunkResultType
cudaq::clean_launch_module(const std::string &name, ModuleOp mod,
                           cudaq::OpaqueArguments &args) {
  // Release the GIL for MLIR compilation and JIT. PyEval_SaveThread requires
  // the GIL to be held, so guard with PyGILState_Check. Async paths invoke
  // this from worker threads that never held the GIL.
  std::optional<nanobind::gil_scoped_release> release;
  if (PyGILState_Check())
    release.emplace();
  auto kernelFunc = getKernelFuncOp(mod, name);
  Type retTy = cudaq::runtime::getReturnType(kernelFunc);
  // Append space for a result, as needed, to the vector of arguments.
  auto rawArgs = appendResultToArgsVector(args, retTy, mod, name);
  return pyLaunchModule(name, mod, rawArgs);
}

cudaq::OpaqueArguments cudaq::marshal_arguments_for_module_launch(
    ModuleOp mod, nanobind::args runtimeArgs, func::FuncOp kernelFunc) {
  // Convert python arguments to opaque form.
  cudaq::OpaqueArguments args;
  bool isLocalSimulator =
      !(cudaq::is_remote_platform() || cudaq::is_emulated_platform());
  auto handler = [&](cudaq::OpaqueArguments &args, nanobind::object &pyArg,
                     unsigned pos) {
    return linkResolvedCallable(mod, kernelFunc, pos, pyArg);
  };
  if (isLocalSimulator)
    cudaq::packArgs<cudaq::PackingStyle::argsCreator>(args, runtimeArgs,
                                                      kernelFunc, handler);
  else
    cudaq::packArgs<cudaq::PackingStyle::synthesis>(args, runtimeArgs,
                                                    kernelFunc, handler);
  return args;
}

nanobind::object cudaq::marshal_and_launch_module(const std::string &name,
                                                  MlirModule module,
                                                  nanobind::args runtimeArgs) {
  // Marker span identifying every nested pass / scoped trace as part of the
  // JIT-time pipeline. Paired with the cudaq.pipeline.aot span emitted around
  // aot-prep-pipeline in compile_to_mlir; tooling reads the trace ancestry to
  // attribute pass events to AOT vs JIT.
  //
  // This site is the funnel for kernel-call / sample / observe /
  // estimate_resources execution paths: each ultimately calls
  // marshal_and_launch_module, so a single span here attributes their JIT
  // pass events to the JIT pipeline. The cudaq.translate path has its own
  // marker in py_translate.cpp::translate_impl since it does not pass
  // through this function.
  cudaq::ScopedTrace pipelineJitMarker(cudaq::TraceContext(__builtin_FUNCTION(),
                                                           __builtin_FILE(),
                                                           __builtin_LINE()),
                                       "cudaq.pipeline.jit");
  ScopedTraceWithContext("marshal_and_launch_module", name);
  auto kernelFunc = getKernelFuncOp(module, name);
  auto mod = unwrap(module);
  Type retTy = cudaq::runtime::getReturnType(kernelFunc);
  auto args = marshal_arguments_for_module_launch(mod, runtimeArgs, kernelFunc);

  [[maybe_unused]] auto resultPtr = clean_launch_module(name, mod, args);

  if (!retTy)
    return nanobind::none();
  return cudaq::convertResult(mod, retTy,
                              reinterpret_cast<char *>(args.getArgs().back()));
}

// Compile (specialize + JIT) the kernel module and return a CompiledModule.
static cudaq::CompiledModule
marshal_and_retain_module(const std::string &name, MlirModule module,
                          bool isEntryPoint, nanobind::args runtimeArgs) {
  ScopedTraceWithContext("marshal_and_retain_module", name);

  auto kernelFunc = cudaq::getKernelFuncOp(module, name);
  auto mod = unwrap(module);
  Type retTy = cudaq::runtime::getReturnType(kernelFunc);
  auto args =
      cudaq::marshal_arguments_for_module_launch(mod, runtimeArgs, kernelFunc);
  // Append space for a result, as needed, to the vector of arguments.
  auto rawArgs = appendResultToArgsVector(args, retTy, mod, name);
  auto clone = mod.clone();
  auto compiled =
      cudaq::streamlinedCompileModule(name, clone, rawArgs, isEntryPoint);
  clone.erase();
  return compiled;
}

static MlirModule synthesizeKernel(nanobind::object kernel,
                                   nanobind::args runtimeArgs) {
  auto module = nanobind::cast<MlirModule>(kernel.attr("qkeModule"));
  auto mod = unwrap(module);
  auto name = nanobind::cast<std::string>(kernel.attr("uniqName"));
  if (mod->hasAttr(cudaq::runtime::pythonUniqueAttrName)) {
    StringRef n =
        cast<StringAttr>(mod->getAttr(cudaq::runtime::pythonUniqueAttrName));
    name = n.str();
  }
  auto kernelFuncOp = cudaq::getKernelFuncOp(module, name);
  cudaq::OpaqueArguments args;
  cudaq::setDataLayout(module);
  cudaq::packArgs(args, runtimeArgs, kernelFuncOp,
                  [](cudaq::OpaqueArguments &, nanobind::object &, unsigned) {
                    return false;
                  });

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

  cudaq_internal::compiler::ArgumentConverter argCon(name, mod);
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
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
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
  bool pmFailed = failed(cudaq::runPassManagerReleasingGIL(pm, cloned));
  timingScope.stop();
  engine.eraseHandler(handlerId);
  if (pmFailed)
    throw std::runtime_error(
        "failed to JIT compile the Quake representation\n" + error_msg);
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

  bool pmFailed = failed(cudaq::runPassManagerReleasingGIL(pm, mod));
  timingScope.stop();
  engine.eraseHandler(handlerId);
  if (pmFailed)
    throw std::runtime_error(
        "failed to JIT compile the Quake representation\n" + error_msg);
  engine.eraseHandler(handlerId);
}

static ModuleOp cleanLowerToCodegenKernel(ModuleOp mod,
                                          const cudaq::OpaqueArguments &args) {
  if (false && isCurrentTargetFullQIR() && args.empty()) {
    // Generate a portable full QIR module with no argument synthesis. All
    // arguments will be resolved and marshaled at the kernel call site.
    auto *ctx = mod.getContext();
    PassManager pm(ctx);
    cudaq::addPythonSignalInstrumentation(pm);
    pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
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
                                   MlirModule module,
                                   nanobind::args runtimeArgs) {
  auto kernelFunc = cudaq::getKernelFuncOp(module, kernelName);
  cudaq::OpaqueArguments args;
  auto mod = unwrap(module);
  cudaq::packArgs(
      args, runtimeArgs, kernelFunc,
      [&](cudaq::OpaqueArguments &args, nanobind::object &pyArg, unsigned pos) {
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

void cudaq::bindAltLaunchKernel(nanobind::module_ &mod,
                                std::function<std::string()> &&getTL) {
  getTransportLayer = std::move(getTL);

  nanobind::class_<cudaq::CompiledModule>(mod, "CompiledModule")
      .def_prop_ro(
          "entry_point",
          [](const cudaq::CompiledModule &ck) {
            return reinterpret_cast<std::uintptr_t>(ck.getJit()->getFn());
          },
          "The address of the JIT-compiled entry point.")
      .def_prop_ro("is_fully_specialized",
                   &cudaq::CompiledModule::isFullySpecialized,
                   "Whether all arguments have been specialized.");

  mod.def("lower_to_codegen", lower_to_codegen,
          "Lower a kernel module to CC dialect. Never launches the kernel.");

  mod.def("clean_launch_module", cudaq::clean_launch_module,
          "Launch a kernel. Does not perform other mischief.");
  mod.def("marshal_and_launch_module", cudaq::marshal_and_launch_module,
          "Launch a kernel. Marshaling of arguments and unmarshalling of "
          "results is performed.");
  mod.def("marshal_and_retain_module", marshal_and_retain_module,
          "Compile (specialize + JIT) a kernel module. Returns a "
          "CompiledModule object that owns the JIT engine.");
  mod.def("pyAltLaunchAnalogKernel", pyAltLaunchAnalogKernel,
          "Launch an analog Hamiltonian simulation kernel with given JSON "
          "payload.");

  mod.def("synthesize", synthesizeKernel, "FIXME: document!");

  mod.def(
      "storePointerToStateData",
      [](const std::string &name, const std::string &hash,
         nanobind::ndarray<> data, simulation_precision precision) {
        auto ptr = data.data();
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
      [](const std::string &name, const std::string &hash,
         nanobind::object data) {
        auto state = nanobind::cast<cudaq::state>(data);
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
        cudaq::addPythonSignalInstrumentation(pm);
        pm.addInstrumentation(
            std::make_unique<cudaq::TracePassInstrumentation>());
        pm.addNestedPass<func::FuncOp>(
            cudaq::opt::createPySynthCallableBlockArgs(
                SmallVector<StringRef>(funcNames.begin(), funcNames.end()),
                true));
        if (failed(cudaq::runPassManagerReleasingGIL(pm, m)))
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
