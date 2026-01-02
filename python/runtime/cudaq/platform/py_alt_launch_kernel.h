/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/algorithms/run.h"
#include "utils/OpaqueArguments.h"
#include "utils/PyTypes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace cudaq {

/// @brief Set current architecture's data layout attribute on a module.
void setDataLayout(MlirModule module);

/// @brief Get the default callable argument handler for packing arguments.
std::function<bool(OpaqueArguments &argData, py::object &arg)>
getCallableArgHandler();

/// @brief Get the names of callable arguments from the given kernel and
/// arguments.
// As we process the arguments, we also perform any extra processing required
// for callable arguments.
std::vector<std::string> getCallableNames(py::object &kernel, py::args &args);

/// @brief Create a new OpaqueArguments pointer and pack the
/// python arguments in it. Clients must delete the memory.
OpaqueArguments *
toOpaqueArgs(py::args &args, MlirModule mod, const std::string &name,
             const std::optional<
                 std::function<bool(OpaqueArguments &argData, py::object &arg)>>
                 &optionalBackupHandler = std::nullopt);

inline std::size_t byteSize(mlir::Type ty) {
  if (isa<mlir::ComplexType>(ty)) {
    auto eleTy = cast<mlir::ComplexType>(ty).getElementType();
    return 2 * cudaq::opt::convertBitsToBytes(eleTy.getIntOrFloatBitWidth());
  }
  if (ty.isIntOrFloat())
    return cudaq::opt::convertBitsToBytes(ty.getIntOrFloatBitWidth());
  ty.dump();
  throw std::runtime_error("Expected a complex, floating, or integral type");
}

/// @brief Convert raw return of kernel to python object.
py::object convertResult(mlir::ModuleOp module, mlir::func::FuncOp kernelFuncOp,
                         mlir::Type ty, char *data);

/// @brief Launch python kernel with arguments.
void pyAltLaunchKernel(const std::string &name, MlirModule module,
                       cudaq::OpaqueArguments &runtimeArgs,
                       const std::vector<std::string> &names);

/// @brief Launch python kernel with arguments.
std::tuple<void *, std::size_t, std::int32_t, KernelThunkType>
pyAltLaunchKernelBase(const std::string &name, MlirModule module,
                      mlir::Type returnType,
                      cudaq::OpaqueArguments &runtimeArgs,
                      const std::vector<std::string> &names,
                      std::size_t startingArgIdx = 0, bool launch = true);

/// @brief Launch python kernel with arguments.
void pyLaunchKernel(const std::string &name, KernelThunkType thunk,
                    mlir::ModuleOp mod, cudaq::OpaqueArguments &runtimeArgs,
                    void *rawArgs, std::size_t size, std::uint32_t returnOffset,
                    const std::vector<std::string> &names);

void bindAltLaunchKernel(py::module &mod, std::function<std::string()> &&);

std::string getQIR(const std::string &name, MlirModule module,
                   cudaq::OpaqueArguments &runtimeArgs,
                   const std::string &profile);

std::string getASM(const std::string &name, MlirModule module,
                   cudaq::OpaqueArguments &runtimeArgs);
} // namespace cudaq
