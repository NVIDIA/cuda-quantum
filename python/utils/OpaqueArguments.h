/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/FmtCore.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/builder/kernel_builder.h"
#include "cudaq/qis/pauli_word.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <chrono>
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

/// @brief Validate that the number of arguments provided is
/// correct for the given kernel_builder. Throw an exception if not.
inline py::args validateInputArguments(kernel_builder<> &kernel,
                                       py::args &args) {
  // Ensure the user input is correct.
  auto nInputArgs = args.size();
  auto nRequiredParams = kernel.getNumParams();
  if (nRequiredParams != nInputArgs)
    throw std::runtime_error(
        fmt::format("Kernel requires {} input parameter{} but {} provided.",
                    nRequiredParams, nRequiredParams > 1 ? "s" : "",
                    nInputArgs == 0 ? "none" : std::to_string(nInputArgs)));

  // Look at the input arguments, validate they are ok
  // Specifically here we'll check if we've been given
  // other list-like types as input for a stdvec (like numpy array)
  py::args processed = py::tuple(args.size());
  for (std::size_t i = 0; i < args.size(); ++i) {
    auto arg = args[i];
    if (kernel.isArgStdVec(i)) {

      auto nRequiredElements = kernel.getArguments()[i].getRequiredElements();

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
      }

      // has to be a list if its not a nd array
      if (!py::isinstance<py::list>(arg))
        throw std::runtime_error(
            "Invalid list-like argument to Kernel.__call__()");

      auto nElements = arg.cast<py::list>().size();
      if (nRequiredElements != nElements)
        throw std::runtime_error("Kernel list argument requires " +
                                 std::to_string(nRequiredElements) + " but " +
                                 std::to_string(nElements) + " were provided.");
    }

    processed[i] = arg;
  }

  // TODO: Handle more type checking

  return processed;
}

/// @brief FIXME. This is a simple version of the above function,
// it will only process numpy arrays of 1D shape and make them compatible
// with our MLIR code. Future work should make this function perform more
// checks, we probably want to take the Kernel MLIR argument Types as input and
// use that to validate that the passed arguments are good to go.
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

inline void
packArgs(OpaqueArguments &argData, py::args args,
         mlir::func::FuncOp kernelFuncOp,
         const std::function<bool(OpaqueArguments &argData, py::object &arg)>
             &backupHandler) {
  if (kernelFuncOp.getNumArguments() != args.size())
    throw std::runtime_error("Invalid runtime arguments - kernel expected " +
                             std::to_string(kernelFuncOp.getNumArguments()) +
                             " but was provided " +
                             std::to_string(args.size()) + " arguments.");

  for (std::size_t i = 0; i < args.size(); i++) {
    py::object arg = args[i];
    auto kernelArgTy = kernelFuncOp.getArgument(i).getType();
    llvm::TypeSwitch<mlir::Type, void>(kernelArgTy)
        .Case([&](mlir::Float64Type ty) {
          if (!py::isinstance<py::float_>(arg))
            throw std::runtime_error("kernel argument type is `float` but "
                                     "argument provided is not (argument " +
                                     std::to_string(i) + ", value=" +
                                     py::str(arg).cast<std::string>() + ").");
          double *ourAllocatedArg = new double();
          *ourAllocatedArg = PyFloat_AsDouble(arg.ptr());
          argData.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<double *>(ptr);
          });
        })
        .Case([&](mlir::IntegerType ty) {
          if (ty.getIntOrFloatBitWidth() == 1) {
            if (!py::isinstance<py::bool_>(arg))
              throw std::runtime_error("kernel argument type is `bool` but "
                                       "argument provided is not (argument " +
                                       std::to_string(i) + ", value=" +
                                       py::str(arg).cast<std::string>() + ").");
            bool *ourAllocatedArg = new bool();
            *ourAllocatedArg = arg.ptr() == (PyObject *)&_Py_TrueStruct;
            argData.emplace_back(ourAllocatedArg, [](void *ptr) {
              delete static_cast<bool *>(ptr);
            });
            return;
          }

          if (!py::isinstance<py::int_>(arg))
            throw std::runtime_error("kernel argument type is `int` but "
                                     "argument provided is not (argument " +
                                     std::to_string(i) + ", value=" +
                                     py::str(arg).cast<std::string>() + ").");

          long *ourAllocatedArg = new long();
          *ourAllocatedArg = PyLong_AsLong(arg.ptr());
          argData.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<long *>(ptr);
          });
        })
        .Case([&](cudaq::cc::CharspanType ty) {
          // pauli word
          cudaq::pauli_word *ourAllocatedArg =
              new cudaq::pauli_word(arg.cast<cudaq::pauli_word>().str());
          argData.emplace_back(ourAllocatedArg, [](void *ptr) {
            delete static_cast<cudaq::pauli_word *>(ptr);
          });
        })
        .Case([&](cudaq::cc::StdvecType ty) {
          if (!py::isinstance<py::list>(arg))
            throw std::runtime_error("kernel argument type is `list` but "
                                     "argument provided is not (argument " +
                                     std::to_string(i) + ", value=" +
                                     py::str(arg).cast<std::string>() + ").");
          auto casted = py::cast<py::list>(arg);
          auto eleTy = ty.getElementType();
          if (casted.empty()) {
            // Handle boolean different since C++ library implementation
            // for vectors of bool is different than other types.
            if (eleTy.isInteger(1)) {
              std::vector<bool> *ourAllocatedArg = new std::vector<bool>();
              argData.emplace_back(ourAllocatedArg, [](void *ptr) {
                delete static_cast<std::vector<bool> *>(ptr);
              });
              return;
            }

            // If its empty, just put any vector on the `argData`,
            // it won't matter since it is empty and all
            // vectors have the same memory footprint (span-like).
            std::vector<std::size_t> *ourAllocatedArg =
                new std::vector<std::size_t>();
            argData.emplace_back(ourAllocatedArg, [](void *ptr) {
              delete static_cast<std::vector<std::size_t> *>(ptr);
            });
            return;
          }

          // Define a generic vector allocator as a
          // templated lambda so we can capture argData and casted.
          auto genericVecAllocator = [&]<typename VecTy>(auto &&converter) {
            std::vector<VecTy> *ourAllocatedArg =
                new std::vector<VecTy>(casted.size());
            for (std::size_t counter = 0; auto el : casted) {
              (*ourAllocatedArg)[counter++] = converter(el);
            }
            argData.emplace_back(ourAllocatedArg, [](void *ptr) {
              delete static_cast<std::vector<VecTy> *>(ptr);
            });
          };

          // Switch on the vector element type
          TypeSwitch<Type, void>(eleTy)
              .Case([&](IntegerType type) {
                // Handle vec<bool> and vec<int>
                if (type.getIntOrFloatBitWidth() == 1) {
                  genericVecAllocator.template operator()<bool>(
                      [](py::handle element) {
                        return element.ptr() == (PyObject *)&_Py_TrueStruct;
                      });
                  return;
                }

                genericVecAllocator.template operator()<std::size_t>(
                    [](py::handle element) -> std::size_t {
                      return PyLong_AsLong(element.ptr());
                    });
                return;
              })
              .Case([&](Float64Type type) {
                genericVecAllocator.template operator()<double>(
                    [](py::handle element) {
                      return PyFloat_AsDouble(element.ptr());
                    });
                return;
              })
              .Case([&](cudaq::cc::CharspanType type) {
                genericVecAllocator.template operator()<cudaq::pauli_word>(
                    [](py::handle element) {
                      auto pw = element.cast<cudaq::pauli_word>();
                      return cudaq::pauli_word(pw.str());
                    });
                return;
              })
              .Case([&](ComplexType type) {
                if (isa<Float64Type>(type.getElementType())) {
                  genericVecAllocator.template operator()<std::complex<double>>(
                      [](py::handle element) -> std::complex<double> {
                        if (!py::hasattr(element, "real"))
                          throw std::runtime_error(
                              "invalid complex element type");
                        if (!py::hasattr(element, "imag"))
                          throw std::runtime_error(
                              "invalid complex element type");
                        return {PyFloat_AsDouble(element.attr("real").ptr()),
                                PyFloat_AsDouble(element.attr("imag").ptr())};
                      });
                } else {
                  genericVecAllocator.template operator()<std::complex<float>>(
                      [](py::handle element) -> std::complex<float> {
                        if (!py::hasattr(element, "real"))
                          throw std::runtime_error(
                              "invalid complex element type");
                        if (!py::hasattr(element, "imag"))
                          throw std::runtime_error(
                              "invalid complex element type");
                        return {element.attr("real").cast<float>(),
                                element.attr("imag").cast<float>()};
                      });
                }
                return;
              })
              .Default([](Type ty) {
                std::string msg;
                {
                  llvm::raw_string_ostream os(msg);
                  ty.print(os);
                }
                throw std::runtime_error("invalid list element type (" + msg +
                                         ").");
              });
        })
        .Default([&](Type) {
          // See if we have a backup type handler.
          auto worked = backupHandler(argData, arg);
          if (!worked)
            throw std::runtime_error("Could not pack argument: " +
                                     py::str(args).cast<std::string>());
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
