/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PyTypes.h"
#include "common/FmtCore.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/builder/kernel_builder.h"
#include "cudaq/qis/pauli_word.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include <chrono>
#include <complex>
#include <functional>
#include <future>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
using namespace std::chrono_literals;
using namespace mlir;

namespace cudaq {

/// @brief The OpaqueArguments type wraps a vector
/// of function arguments represented as opaque
/// pointers. For each element in the vector of opaque
/// pointers, we also track the arguments corresponding
/// deletion function - a function invoked upon destruction
/// of this OpaqueArguments to clean up the memory.
class OpaqueArguments {
public:
  using OpaqueArgDeleter = std::function<void(void *)>;

private:
  /// @brief The opaque argument pointers
  std::vector<void *> args;

  /// @brief Deletion functions for the arguments.
  std::vector<OpaqueArgDeleter> deleters;

public:
  /// @brief Add an opaque argument and its `deleter` to this OpaqueArguments
  template <typename ArgPointer, typename Deleter>
  void emplace_back(ArgPointer &&pointer, Deleter &&deleter) {
    args.emplace_back(pointer);
    deleters.emplace_back(deleter);
  }

  /// @brief Return the `args` as a pointer to void*.
  void **data() { return args.data(); }

  /// @brief Return the number of arguments
  std::size_t size() { return args.size(); }

  /// Destructor, clean up the memory
  ~OpaqueArguments() {
    for (std::size_t counter = 0; auto &ptr : args)
      deleters[counter++](ptr);

    args.clear();
    deleters.clear();
  }
};

/// @brief This function modifies input arguments to convert them into valid
/// CUDA-Q argument types. Future work should make this function perform more
/// checks, we probably want to take the Kernel MLIR argument Types as input and
/// use that to validate that the passed arguments are good to go.
inline py::args simplifiedValidateInputArguments(py::args &args) {
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
      arg = cudaq::pauli_word(py::cast<std::string>(arg));
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

/// @brief Search the given Module for the function with provided name.
inline mlir::func::FuncOp getKernelFuncOp(MlirModule module,
                                          const std::string &kernelName) {
  using namespace mlir;
  ModuleOp mod = unwrap(module);
  func::FuncOp kernelFunc;
  mod.walk([&](func::FuncOp function) {
    if (function.getName().equals("__nvqpp__mlirgen__" + kernelName)) {
      kernelFunc = function;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!kernelFunc)
    throw std::runtime_error("Could not find " + kernelName +
                             " function in current module.");

  return kernelFunc;
}

template <typename T>
void checkArgumentType(py::handle arg, int index) {
  if (!py_ext::isConvertible<T>(arg)) {
    throw std::runtime_error(
        "kernel argument type is '" + std::string(py_ext::typeName<T>()) + "'" +
        " but argument provided is not (argument " + std::to_string(index) +
        ", value=" + py::str(arg).cast<std::string>() +
        ", type=" + py::str(py::type::of(arg)).cast<std::string>() + ").");
  }
}

template <typename T>
void checkListElementType(py::handle arg, int index, int elementIndex) {
  if (!py_ext::isConvertible<T>(arg)) {
    throw std::runtime_error(
        "kernel argument's element type is '" +
        std::string(py_ext::typeName<T>()) + "'" +
        " but argument provided is not (argument " + std::to_string(index) +
        ", element " + std::to_string(elementIndex) +
        ", value=" + py::str(arg).cast<std::string>() +
        ", type=" + py::str(py::type::of(arg)).cast<std::string>() + ").");
  }
}

template <typename T>
inline void addArgument(OpaqueArguments &argData, T &&arg) {
  T *allocatedArg = new T(std::move(arg));
  argData.emplace_back(allocatedArg,
                       [](void *ptr) { delete static_cast<T *>(ptr); });
}

template <typename T>
inline void valueArgument(OpaqueArguments &argData, T *arg) {
  argData.emplace_back(static_cast<void *>(arg), [](void *) {});
}

inline std::string mlirTypeToString(mlir::Type ty) {
  std::string msg;
  {
    llvm::raw_string_ostream os(msg);
    ty.print(os);
  }
  return msg;
}

/// @brief Return the size and member variable offsets for the input struct.
inline std::pair<std::size_t, std::vector<std::size_t>>
getTargetLayout(func::FuncOp func, cudaq::cc::StructType structTy) {
  auto mod = func->getParentOfType<ModuleOp>();
  StringRef dataLayoutSpec = "";
  if (auto attr = mod->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
    dataLayoutSpec = cast<StringAttr>(attr);
  auto dataLayout = llvm::DataLayout(dataLayoutSpec);
  // Convert bufferTy to llvm.
  llvm::LLVMContext context;
  LLVMTypeConverter converter(func.getContext());
  cudaq::opt::initializeTypeConversions(converter);
  auto llvmDialectTy = converter.convertType(structTy);
  LLVM::TypeToLLVMIRTranslator translator(context);
  auto *llvmStructTy =
      cast<llvm::StructType>(translator.translateType(llvmDialectTy));
  auto *layout = dataLayout.getStructLayout(llvmStructTy);
  auto strSize = layout->getSizeInBytes();
  std::vector<std::size_t> fieldOffsets;
  for (std::size_t i = 0, I = structTy.getMembers().size(); i != I; ++i)
    fieldOffsets.emplace_back(layout->getElementOffset(i));
  return {strSize, fieldOffsets};
}

/// @brief For the current struct member variable type, insert the
/// value into the dynamically-constructed struct.
inline void handleStructMemberVariable(void *data, std::size_t offset,
                                       Type memberType, py::object value) {
  auto appendValue = [](void *data, auto &&value, std::size_t offset) {
    std::memcpy(((char *)data) + offset, &value,
                sizeof(std::remove_cvref_t<decltype(value)>));
  };
  llvm::TypeSwitch<Type, void>(memberType)
      .Case([&](IntegerType ty) {
        if (ty.isInteger(1)) {
          appendValue(data, value.cast<py::bool_>(), offset);
          return;
        }
        appendValue(data, (std::size_t)value.cast<py::int_>(), offset);
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

        TypeSwitch<Type, void>(ty.getElementType())
            .Case([&](IntegerType type) {
              if (type.isInteger(1)) {
                appendVectorValue(value, data, offset, bool());
                return;
              }

              appendVectorValue(value, data, offset, std::size_t());
              return;
            })
            .Case([&](FloatType type) {
              if (type.isF32()) {
                appendVectorValue(value, data, offset, float());
                return;
              }

              appendVectorValue(value, data, offset, double());
              return;
            });
      })
      .Default([&](Type ty) {
        ty.dump();
        throw std::runtime_error(
            "Type not supported for custom struct in kernel.");
      });
}

inline void packArgs(OpaqueArguments &argData, py::args args,
                     mlir::func::FuncOp kernelFuncOp,
                     const std::function<bool(OpaqueArguments &argData,
                                              py::object &arg)> &backupHandler,
                     std::size_t startingArgIdx = 0) {
  if (kernelFuncOp.getNumArguments() != args.size())
    throw std::runtime_error("Invalid runtime arguments - kernel expected " +
                             std::to_string(kernelFuncOp.getNumArguments()) +
                             " but was provided " +
                             std::to_string(args.size()) + " arguments.");

  for (std::size_t i = startingArgIdx; i < args.size(); i++) {
    py::object arg = args[i];
    auto kernelArgTy = kernelFuncOp.getArgument(i).getType();
    llvm::TypeSwitch<mlir::Type, void>(kernelArgTy)
        .Case([&](mlir::ComplexType ty) {
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
        .Case([&](mlir::Float64Type ty) {
          checkArgumentType<py_ext::Float>(arg, i);
          addArgument(argData, arg.cast<double>());
        })
        .Case([&](mlir::Float32Type ty) {
          checkArgumentType<py_ext::Float>(arg, i);
          addArgument(argData, arg.cast<float>());
        })
        .Case([&](mlir::Float32Type ty) {
          if (!py::isinstance<py::float_>(arg))
            throw std::runtime_error("kernel argument type is `float` but "
                                     "argument provided is not (argument " +
                                     std::to_string(i) + ", value=" +
                                     py::str(arg).cast<std::string>() + ").");
          float *ourAllocatedArg = new float();
          *ourAllocatedArg = arg.cast<float>();
          argData.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<float *>(ptr);
          });
        })
        .Case([&](mlir::IntegerType ty) {
          if (ty.getIntOrFloatBitWidth() == 1) {
            checkArgumentType<py::bool_>(arg, i);
            addArgument(argData, arg.cast<bool>());
            return;
          }

          checkArgumentType<py::int_>(arg, i);
          addArgument(argData, arg.cast<long>());
        })
        .Case([&](cudaq::cc::CharspanType ty) {
          addArgument(argData,
                      cudaq::pauli_word(arg.cast<cudaq::pauli_word>().str()));
        })
        .Case([&](cudaq::cc::PointerType ty) {
          if (isa<cudaq::cc::StateType>(ty.getElementType())) {
            valueArgument(argData, arg.cast<cudaq::state *>());
          } else {
            throw std::runtime_error("Invalid pointer type argument: " +
                                     py::str(arg).cast<std::string>() +
                                     " Type: " + mlirTypeToString(ty));
          }
        })
        .Case([&](cudaq::cc::StructType ty) {
          auto [size, offsets] = getTargetLayout(kernelFuncOp, ty);
          auto memberTys = ty.getMembers();
          auto allocatedArg = std::malloc(size);
          py::dict attributes = arg.attr("__annotations__").cast<py::dict>();
          for (std::size_t i = 0;
               const auto &[attr_name, unused] : attributes) {
            py::object attr_value =
                arg.attr(attr_name.cast<std::string>().c_str());
            handleStructMemberVariable(allocatedArg, offsets[i], memberTys[i],
                                       attr_value);
            i++;
          }

          argData.emplace_back(allocatedArg, [](void *ptr) { std::free(ptr); });
        })
        .Case([&](cudaq::cc::StdvecType ty) {
          checkArgumentType<py::list>(arg, i);
          auto casted = py::cast<py::list>(arg);
          auto eleTy = ty.getElementType();
          if (casted.empty()) {
            // Handle boolean different since C++ library implementation
            // for vectors of bool is different than other types.
            if (eleTy.isInteger(1)) {
              addArgument(argData, std::vector<bool>());
              return;
            }

            // If its empty, just put any vector on the `argData`,
            // it won't matter since it is empty and all
            // vectors have the same memory footprint (span-like).
            addArgument(argData, std::vector<std::size_t>());
            return;
          }

          // Define a generic vector allocator as a
          // templated lambda so we can capture argData and casted.
          auto genericVecAllocator = [&]<typename VecTy>(auto &&converter) {
            auto values = std::vector<VecTy>(casted.size());
            for (std::size_t counter = 0; auto el : casted) {
              auto converted = converter(el, i, counter);
              values[counter++] = converted;
            }
            addArgument(argData, std::move(values));
          };

          // Switch on the vector element type.
          TypeSwitch<Type, void>(eleTy)
              .Case([&](IntegerType type) {
                // Handle vec<bool> and vec<int>
                if (type.getIntOrFloatBitWidth() == 1) {
                  genericVecAllocator.template operator()<bool>(
                      [](py::handle element, int index, int elementIndex) {
                        checkListElementType<py::bool_>(element, index,
                                                        elementIndex);
                        return element.cast<bool>();
                      });
                  return;
                }

                genericVecAllocator.template operator()<long>(
                    [](py::handle element, int index,
                       int elementIndex) -> long {
                      checkListElementType<py::int_>(element, index,
                                                     elementIndex);
                      return element.cast<long>();
                    });
                return;
              })
              .Case([&](Float64Type type) {
                genericVecAllocator.template operator()<double>(
                    [](py::handle element, int index, int elementIndex) {
                      checkListElementType<py_ext::Float>(element, index,
                                                          elementIndex);
                      return element.cast<double>();
                    });
                return;
              })
              .Case([&](Float32Type type) {
                genericVecAllocator.template operator()<float>(
                    [](py::handle element, int index, int elementIndex) {
                      checkListElementType<py_ext::Float>(element, index,
                                                          elementIndex);
                      return element.cast<float>();
                    });
                return;
              })
              .Case([&](cudaq::cc::CharspanType type) {
                genericVecAllocator.template operator()<cudaq::pauli_word>(
                    [](py::handle element, int index, int elementIndex) {
                      auto pw = element.cast<cudaq::pauli_word>();
                      return cudaq::pauli_word(pw.str());
                    });
                return;
              })
              .Case([&](ComplexType type) {
                if (isa<Float64Type>(type.getElementType())) {
                  genericVecAllocator.template operator()<std::complex<double>>(
                      [](py::handle element, int index,
                         int elementIndex) -> std::complex<double> {
                        checkListElementType<py_ext::Complex>(element, index,
                                                              elementIndex);
                        return element.cast<std::complex<double>>();
                      });
                } else {
                  genericVecAllocator.template operator()<std::complex<float>>(
                      [](py::handle element, int index,
                         int elementIndex) -> std::complex<float> {
                        checkListElementType<py_ext::Complex>(element, index,
                                                              elementIndex);
                        return element.cast<std::complex<float>>();
                      });
                }
                return;
              })
              .Default([](Type ty) {
                throw std::runtime_error("invalid list element type (" +
                                         mlirTypeToString(ty) + ").");
              });
        })
        .Default([&](Type ty) {
          // See if we have a backup type handler.
          auto worked = backupHandler(argData, arg);
          if (!worked)
            throw std::runtime_error(
                "Could not pack argument: " + py::str(arg).cast<std::string>() +
                " Type: " + mlirTypeToString(ty));
        });
  }
}

/// @brief Return true if the given `py::args` represents a request for
/// broadcasting sample or observe over all argument sets. `args` types can be
/// `int`, `float`, `list`, so  we should check if `args[i]` is a `list` or
/// `ndarray`.
inline bool isBroadcastRequest(kernel_builder<> &builder, py::args &args) {
  if (args.empty())
    return false;

  auto arg = args[0];
  // Just need to check the leading argument
  if (py::isinstance<py::list>(arg) && !builder.isArgStdVec(0))
    return true;

  if (py::hasattr(arg, "tolist")) {
    if (!py::hasattr(arg, "shape"))
      return false;

    auto shape = arg.attr("shape").cast<py::tuple>();
    if (shape.size() == 1 && !builder.isArgStdVec(0))
      return true;

    // // If shape is 2, then we know its a list of list
    if (shape.size() == 2)
      return true;
  }

  return false;
}

/// @brief Create a new OpaqueArguments pointer and pack the
/// python arguments in it. Clients must delete the memory.
inline OpaqueArguments *toOpaqueArgs(py::args &args, MlirModule mod,
                                     const std::string &name) {
  auto kernelFunc = getKernelFuncOp(mod, name);
  auto *argData = new cudaq::OpaqueArguments();
  args = simplifiedValidateInputArguments(args);
  cudaq::packArgs(*argData, args, kernelFunc,
                  [](OpaqueArguments &, py::object &) { return false; });
  return argData;
}

} // namespace cudaq
