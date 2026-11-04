/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CompiledModuleCache.h"
#include "common/CompiledModule.h"
#include "utils/OpaqueArguments.h"
#include "utils/PyTypes.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/algorithms/run.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {

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
/// \p level, if given, overrides the platform's default quantum optimization
/// level for this launch.
nanobind::object marshal_and_launch_module(
    const std::string &kernelName, MlirModule module,
    nanobind::args runtimeArgs,
    std::shared_ptr<detail::CompiledModuleCache> cache = nullptr,
    std::optional<OptimizationLevel> level = std::nullopt);

/// Pure C++ code that launches a kernel. Argument marshaling and result
/// unmarshalling is \e not performed. \p level, if given, overrides the
/// platform's default quantum optimization level for this launch.
KernelThunkResultType clean_launch_module(
    const std::string &kernelName, mlir::ModuleOp mod, OpaqueArguments &args,
    std::shared_ptr<detail::CompiledModuleCache> cache = nullptr,
    std::optional<OptimizationLevel> level = std::nullopt);

/// Convert the `optimization_level` passed from Python to an optimization
/// level. Throws if it is not between 0 and 3.
std::optional<OptimizationLevel>
toOptimizationLevel(std::optional<int> optimizationLevel);

/// Whether a launch on QPU \p qpuId compiles a kernel with its argument values
/// substituted. See `cudaq_internal::compiler::specializesArguments`.
bool launchSpecializesArguments(std::size_t qpuId);

/// Pack the Python arguments so the kernel can be launched with them.
///
/// Set \p specializeArguments to true when the compiler will replace the
/// arguments with their values. The compiler reads the arguments differently
/// in that case, so this must match what the compiler actually does. If it
/// doesn't, some types, such as `list[bool]`, are read as the wrong values,
/// and nothing reports an error.
OpaqueArguments marshal_arguments_for_module_launch(
    mlir::ModuleOp mod, nanobind::args runtimeArgs,
    mlir::func::FuncOp kernelFunc, bool specializeArguments);

} // namespace cudaq
