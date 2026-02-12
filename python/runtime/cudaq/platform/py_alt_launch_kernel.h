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
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <nanobind/stl/complex.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/map.h>
#include <string>
#include <vector>

namespace py = nanobind;

namespace cudaq {

/// @brief Set current architecture's data layout attribute on a module.
void setDataLayout(MlirModule module);

/// @brief Create a new OpaqueArguments pointer and pack the
/// python arguments in it. Clients must delete the memory.
OpaqueArguments *toOpaqueArgs(py::args &args, MlirModule mod,
                              const std::string &name);

// FIXME: Document!
std::size_t byteSize(mlir::Type ty);

/// @brief Convert raw return of kernel to python object.
py::object convertResult(mlir::ModuleOp module, mlir::Type ty, char *data);

/// Create python bindings for C++ code in this compilation unit.
void bindAltLaunchKernel(py::module_ &mod, std::function<std::string()> &&);

/// Launch the kernel \p kernelName from module \p module. \p runtimeArgs are
/// the python arguments to the kernel. Pre-condition: all arguments must be
/// resolved at this `callsite` \e prior to launching this module. In particular
/// this means \p module is ready for beta reduction of callables. If the kernel
/// has a result, it has type \p returnType. \p module must be modifiable.
py::object marshal_and_launch_module(const std::string &kernelName,
                                     MlirModule module, MlirType returnType,
                                     py::args runtimeArgs);

/// Pure C++ code that launches a kernel. Argument marshaling and result
/// unmarshalling is \e not performed.
KernelThunkResultType clean_launch_module(const std::string &kernelName,
                                          mlir::ModuleOp mod, mlir::Type retTy,
                                          OpaqueArguments &args);

OpaqueArguments
marshal_arguments_for_module_launch(mlir::ModuleOp mod, py::args runtimeArgs,
                                    mlir::func::FuncOp kernelFunc);

} // namespace cudaq
