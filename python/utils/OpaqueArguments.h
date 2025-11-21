/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
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

  const std::vector<void *> &getArgs() const { return args; }

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

private:
  /// @brief The opaque argument pointers
  std::vector<void *> args;

  /// @brief Deletion functions for the arguments.
  std::vector<OpaqueArgDeleter> deleters;
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

/// @brief Search the given Module for the function with provided name.
inline mlir::func::FuncOp getKernelFuncOp(MlirModule module,
                                          const std::string &kernelName) {
  mlir::ModuleOp mod = unwrap(module);
  mlir::func::FuncOp kernelFunc;
  for (auto &artifact : mod) {
    if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(artifact);
        function &&
        function.getName() == cudaq::runtime::cudaqGenPrefixName + kernelName) {
      kernelFunc = function;
      break;
    }
  }

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
void checkListElementType(py::handle arg, int index) {
  if (!py_ext::isConvertible<T>(arg)) {
    throw std::runtime_error(
        "kernel argument's element type is '" +
        std::string(py_ext::typeName<T>()) + "'" +
        " but argument provided is not (argument " + std::to_string(index) +
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
getTargetLayout(mlir::func::FuncOp func, cudaq::cc::StructType structTy) {
  auto mod = func->getParentOfType<mlir::ModuleOp>();
  mlir::StringRef dataLayoutSpec = "";
  if (auto attr = mod->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
    dataLayoutSpec = mlir::cast<mlir::StringAttr>(attr);
  else
    throw std::runtime_error("No data layout attribute is set on the module.");

  auto dataLayout = llvm::DataLayout(dataLayoutSpec);
  // Convert bufferTy to llvm.
  llvm::LLVMContext context;
  mlir::LLVMTypeConverter converter(func.getContext());
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

/// @brief For the current struct member variable type, insert the
/// value into the dynamically-constructed struct.
inline void handleStructMemberVariable(void *data, std::size_t offset,
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
        throw std::runtime_error("dynamically sized element types for function "
                                 "arguments are not yet supported");
      })
      .Default([&](mlir::Type ty) {
        ty.dump();
        throw std::runtime_error(
            "Type not supported for custom struct in kernel.");
      });
}

/// @brief For the current vector element type, insert the
/// value into the dynamically-constructed vector.
inline void *handleVectorElements(mlir::Type eleTy, py::list list) {
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
          return appendValue.template operator()<bool>(
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
          return appendVectorValue.template operator()<bool>(eleTy, list);

        // All other `std::Vector<T>` types, including nested vectors.
        return appendVectorValue.template operator()<std::size_t>(eleTy, list);
      })
      .Default([&](mlir::Type ty) {
        throw std::runtime_error("invalid list element type (" +
                                 mlirTypeToString(ty) + ").");
        return nullptr;
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
          if (mlir::isa<mlir::Float64Type>(ty.getElementType())) {
            addArgument(argData, arg.cast<std::complex<double>>());
          } else if (mlir::isa<mlir::Float32Type>(ty.getElementType())) {
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

          checkArgumentType<py_ext::Int>(arg, i);
          addArgument(argData, arg.cast<std::int64_t>());
        })
        .Case([&](cudaq::cc::CharspanType ty) {
          addArgument(argData, arg.cast<cudaq::pauli_word>().str());
        })
        .Case([&](cudaq::cc::PointerType ty) {
          if (isa<quake::StateType>(ty.getElementType())) {
            addArgument(argData, cudaq::state(*arg.cast<cudaq::state *>()));
          } else {
            throw std::runtime_error("Invalid pointer type argument: " +
                                     py::str(arg).cast<std::string>() +
                                     " Type: " + mlirTypeToString(ty));
          }
        })
        .Case([&](cudaq::cc::StructType ty) {
          if (ty.getName() == "tuple") {
            auto [size, offsets] = getTargetLayout(kernelFuncOp, ty);
            auto memberTys = ty.getMembers();
            auto allocatedArg = std::malloc(size);
            auto elements = arg.cast<py::tuple>();
            for (std::size_t i = 0; i < offsets.size(); i++)
              handleStructMemberVariable(allocatedArg, offsets[i], memberTys[i],
                                         elements[i]);

            argData.emplace_back(allocatedArg,
                                 [](void *ptr) { std::free(ptr); });
          } else {
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

            argData.emplace_back(allocatedArg,
                                 [](void *ptr) { std::free(ptr); });
          }
        })
        .Case([&](cudaq::cc::StdvecType ty) {
          auto appendVectorValue = [&argData]<typename T>(mlir::Type eleTy,
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
            appendVectorValue.template operator()<bool>(eleTy, list);
            return;
          }
          // All other `std::vector<T>` types, including nested vectors.
          appendVectorValue.template operator()<std::int64_t>(eleTy, list);
        })
        .Default([&](mlir::Type ty) {
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

} // namespace cudaq
