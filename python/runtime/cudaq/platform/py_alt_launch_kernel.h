/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompiledModule.h"
#include "utils/OpaqueArguments.h"
#include "utils/PyTypes.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/algorithms/run.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {

/// A Python-owned mutable handle to a CompiledModule. Python passes one of
/// these through the launch entry points so that C++ can install (cache) a
/// JIT-compiled module and reuse it on subsequent calls.
using CompiledModulePtr = std::shared_ptr<CompiledModule>;

/// @brief Set current architecture's data layout attribute on a module.
void setDataLayout(MlirModule module);

/// @brief Create a new OpaqueArguments pointer and pack the
/// python arguments in it. Clients must delete the memory.
OpaqueArguments *toOpaqueArgs(nanobind::args &args, MlirModule mod,
                              const std::string &name);

// FIXME: Document!
std::size_t byteSize(mlir::Type ty);

/// @brief Convert raw return of kernel to python object.
nanobind::object convertResult(mlir::ModuleOp module, mlir::Type ty,
                               char *data);

/// Create python bindings for C++ code in this compilation unit.
void bindAltLaunchKernel(nanobind::module_ &mod,
                         std::function<std::string()> &&);

/// Launch the kernel \p kernelName from module \p module. \p runtimeArgs are
/// the python arguments to the kernel. Pre-condition: all arguments must be
/// resolved at this `callsite` \e prior to launching this module. In particular
/// this means \p module is ready for beta reduction of callables. The return
/// type is obtained from the kernel's FuncOp. \p module must be modifiable.
nanobind::object marshal_and_launch_module(const std::string &kernelName,
                                           MlirModule module,
                                           CompiledModulePtr *compiled,
                                           const std::string &launchInfo,
                                           nanobind::args runtimeArgs);

/// Pure C++ code that launches a kernel. Argument marshaling and result
/// unmarshalling is \e not performed.
KernelThunkResultType clean_launch_module(const std::string &kernelName,
                                          mlir::ModuleOp mod,
                                          CompiledModulePtr *compiled,
                                          const std::string &launchInfo,
                                          OpaqueArguments &args);

/// Marshal python arguments into an OpaqueArguments for kernel launch.
/// Encodes arguments in the runtime ABI layout for direct local simulation,
/// and the synthesis-pass layout for all other targets.
OpaqueArguments
marshal_arguments_for_module_launch(mlir::ModuleOp mod,
                                    nanobind::args runtimeArgs,
                                    mlir::func::FuncOp kernelFunc);

} // namespace cudaq
