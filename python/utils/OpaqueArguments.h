/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PyTypes.h"
#include "common/ArgumentWrapper.h"
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

namespace py = pybind11;

namespace cudaq {

// NB: The OpaqueArguments class was moved out of Python and into the general
// runtime, so it can be used to launch kernels. See ArgumentWrapper.h.
class OpaqueArguments;

/// This function modifies input arguments to convert them into valid CUDA-Q
/// argument types. Future work should make this function perform more checks,
/// we probably want to take the kernel MLIR argument types as input and use
/// that to validate that the passed arguments are good to go.
py::args simplifiedValidateInputArguments(py::args &args);

/// @brief Search the given Module for the function with provided name.
template <bool noThrow = false>
mlir::func::FuncOp getKernelFuncOp(mlir::ModuleOp mod,
                                   const std::string &name) {
  std::string lookupName = name;
  if (mod->hasAttr(cudaq::runtime::pythonUniqueAttrName)) {
    mlir::StringRef uniqName = cast<mlir::StringAttr>(
        mod->getAttr(cudaq::runtime::pythonUniqueAttrName));
    if (uniqName.substr(0, name.size()) == name) {
      // FIXME: The path to this call used the wrong name.
      lookupName = uniqName;
    }
  }
  lookupName = cudaq::runtime::cudaqGenPrefixName + lookupName;
  if (auto result = mod.lookupSymbol<mlir::func::FuncOp>(lookupName))
    return result;
  if constexpr (noThrow) {
    return {};
  } else /*constexpr*/ {
    mod.dump();
    throw std::runtime_error("Could not find " + lookupName +
                             " function in current module.");
  }
}

template <bool noThrow = false>
mlir::func::FuncOp getKernelFuncOp(MlirModule module,
                                   const std::string &kernelName) {
  return getKernelFuncOp<noThrow>(unwrap(module), kernelName);
}

template <typename T>
void checkArgumentType(py::handle arg, int index, const std::string &word) {
  if (!py_ext::isConvertible<T>(arg)) {
    throw std::runtime_error(
        "kernel argument" + word + " type is '" +
        std::string(py_ext::typeName<T>()) + "'" +
        " but argument provided is not (argument " + std::to_string(index) +
        ", value=" + py::str(arg).cast<std::string>() +
        ", type=" + py::str(py::type::of(arg)).cast<std::string>() + ").");
  }
}

template <typename T>
void checkArgumentType(py::handle arg, int index) {
  checkArgumentType<T>(arg, index, "");
}

template <typename T>
void checkListElementType(py::handle arg, int index) {
  checkArgumentType<T>(arg, index, "'s element");
}

template <typename T>
void addArgument(OpaqueArguments &argData, T &&arg) {
  T *allocatedArg = new T(std::move(arg));
  argData.emplace_back(allocatedArg,
                       [](void *ptr) { delete static_cast<T *>(ptr); });
}

template <typename T>
void valueArgument(OpaqueArguments &argData, T *arg) {
  argData.emplace_back(static_cast<void *>(arg), [](void *) {});
}

std::string mlirTypeToString(mlir::Type ty);

/// @brief Return the size and member variable offsets for the input struct.
std::pair<std::size_t, std::vector<std::size_t>>
getTargetLayout(mlir::ModuleOp mod, cudaq::cc::StructType structTy);

/// For the current struct member variable type, insert the value into the
/// dynamically constructed struct.
void handleStructMemberVariable(void *data, std::size_t offset,
                                mlir::Type memberType, py::object value);

/// For the current vector element type, insert the value into the dynamically
/// constructed vector.
void *handleVectorElements(mlir::Type eleTy, py::list list);

/// Take a list of python objects (the arguments) and convert them to C++
/// objects on the heap. The results are returned in \p argData and include
/// special `deletors` so that the argument data is cleaned up correctly.
void packArgs(OpaqueArguments &argData, py::list args,
              mlir::ArrayRef<mlir::Type> mlirTys,
              const std::function<bool(OpaqueArguments &, py::object &,
                                       unsigned)> &backupHandler,
              mlir::func::FuncOp kernelFuncOp);

/// This overload handles dropping the front \p startingArgIdx arguments on the
/// floor. They are not packed in \p argData and are simply ignored.
void packArgs(OpaqueArguments &argData, py::args args,
              mlir::func::FuncOp kernelFuncOp,
              const std::function<bool(OpaqueArguments &, py::object &,
                                       unsigned)> &backupHandler,
              std::size_t startingArgIdx = 0);

/// Return `true` if the given \p args represents a request for broadcasting
/// sample or observe over all argument sets. \p args types can be `int`,
/// `float`, `list`, so must check if `args[i]` is a `list` or `ndarray`.
inline bool isBroadcastRequest(kernel_builder<> &builder, py::args &args) {
  // FIXME: The use of isArgStdVec in this function inhibits moving this code
  // out of the header file.
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
