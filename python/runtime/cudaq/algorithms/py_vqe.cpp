/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "common/ArgumentWrapper.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/algorithms/gradient.h"
#include "cudaq/algorithms/optimizer.h"
#include "py_vqe.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq {

void *pyGetKernelArgs(const std::string &name, MlirModule module,
                      cudaq::OpaqueArguments &runtimeArgs,
                      const std::vector<std::string> &names,
                      std::size_t startingArgIdx);

/// @brief Return the quantum kernel `FuncOp` from the given `ModuleOp`.
mlir::func::FuncOp getKernelFuncOp(mlir::ModuleOp &mod,
                                   const std::string &kernelName) {
  mlir::func::FuncOp kernel;
  mod.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == cudaq::runtime::cudaqGenPrefixName + kernelName)
      kernel = func;
    return mlir::WalkResult::advance();
  });

  if (!kernel)
    throw std::runtime_error(
        "Could not find kernel to get number of function arguments.");
  return kernel;
}

/// @brief Return the number of function arguments for the kernel.
std::size_t getNumArguments(MlirModule &module, const std::string &kernelName) {
  auto mod = unwrap(module);
  auto kernel = getKernelFuncOp(mod, kernelName);
  return kernel.getNumArguments();
}

/// @brief Return true if the `argIdx` argument is of `StdvecType`.
bool isArgumentStdVec(MlirModule &module, const std::string &kernelName,
                      std::size_t argIdx) {
  auto mod = unwrap(module);
  auto kernel = getKernelFuncOp(mod, kernelName);
  return isa<cudaq::cc::StdvecType>(kernel.getArgument(argIdx).getType());
}

/// @brief Return the kernel name and MLIR module for a kernel.
static inline std::pair<std::string, MlirModule>
getKernelNameAndModule(py::object &kernel) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();
  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  return std::make_pair(kernelName, kernelMod);
}

/// @brief Run `cudaq::observe` on the provided kernel and spin operator.
observe_result pyObserve(py::object &kernel, spin_op &spin_operator,
                         py::args args, const int shots,
                         bool argMapperProvided = false) {
  auto [kernelName, kernelMod] = getKernelNameAndModule(kernel);
  auto &platform = cudaq::get_platform();
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);

  auto numKernelArgs = getNumArguments(kernelMod, kernelName);
  if (numKernelArgs == 0)
    throw std::runtime_error(
        "[cudaq.vqe] cannot run vqe on a kernel with no input arguments.");

  auto isFirstArgStdVec = isArgumentStdVec(kernelMod, kernelName, 0);
  // If user did not provide an argument mapper, then
  // we need to make sure the kernel has 1 argument
  // and it is a stdvec argument.
  if (!argMapperProvided && (numKernelArgs != 1 || !isFirstArgStdVec))
    throw std::runtime_error(
        "[cudaq.vqe] kernels with signature other than "
        "`void(List[float])` must provide an `argument_mapper`.");

  if (argMapperProvided && numKernelArgs == 1 && isFirstArgStdVec)
    throw std::runtime_error(
        "Argument mapper provided but not needed since this kernel takes a "
        "single `list[float]` argument.");

  // Launch the observation task
  return details::runObservation(
             [&]() mutable {
               pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
               delete argData;
             },
             spin_operator, platform, shots, kernelName)
      .value();
}

/// @brief Return whether or not \p kernel is compatible with the remote VQE
/// implementation that requires the variation parameters to be the first
/// argument in the kernel.
static bool firstArgIsCompatibleWithRemoteVQE(py::object &kernel) {
  auto [kernelName, kernelMod] = getKernelNameAndModule(kernel);
  auto kernelFunc = getKernelFuncOp(kernelMod, kernelName);
  if (kernelFunc.getNumArguments() < 1)
    return false;
  auto firstKernelArgTy = kernelFunc.getArgument(0).getType();
  if (auto stdVec = dyn_cast<cudaq::cc::StdvecType>(firstKernelArgTy)) {
    auto eleTy = stdVec.getElementType();
    return isa<mlir::Float32Type, mlir::Float64Type>(eleTy);
  } else {
    return false;
  }
}

/// @brief Perform VQE on a remote platform using the C++ capabilities. This
/// function is used for many of the pyVQE variants below, so some of the
/// parameters may be nullptr.
static optimization_result
pyVQE_remote_cpp(cudaq::quantum_platform &platform, py::object &kernel,
                 spin_op &hamiltonian, cudaq::optimizer &optimizer,
                 cudaq::gradient *gradient, py::function *argumentMapper,
                 const int n_params, const int shots) {
  auto [kernelName, kernelMod] = getKernelNameAndModule(kernel);
  auto ctx = std::make_unique<ExecutionContext>("observe", /*shots=*/0);
  ctx->kernelName = kernelName;
  ctx->spin = cudaq::spin_op::canonicalize(hamiltonian);
  platform.set_exec_ctx(ctx.get());

  constexpr std::size_t startingArgIdx = 1;
  cudaq::OpaqueArguments args;
  void *kernelArgs = nullptr;
  if (argumentMapper) {
    std::vector<double> myArg(n_params);
    py::list py_list = py::cast(myArg);
    py::tuple result = (*argumentMapper)(py_list);
    py::args runtimeArgs = result;

    // Serialize arguments (all concrete parameters except for the first one)
    // into kernelArgs buffer space.
    auto kernelFunc = getKernelFuncOp(kernelMod, kernelName);
    setDataLayout(kernelMod);
    cudaq::packArgs(
        args, runtimeArgs, kernelFunc,
        [](OpaqueArguments &, py::object &) { return false; }, startingArgIdx);
  }
  kernelArgs = pyGetKernelArgs(kernelName, kernelMod, args, /*names=*/{},
                               startingArgIdx);

  // Need to form cudaq::ArgWrapper and pass that into launchVQE.
  std::vector<std::string> names;
  auto *wrapper = new cudaq::ArgWrapper{unwrap(kernelMod), names, kernelArgs};

  platform.launchVQE(kernelName, wrapper, gradient, ctx->spin.value(),
                     optimizer, n_params, shots);
  platform.reset_exec_ctx();
  delete wrapper;
  if (kernelArgs)
    std::free(kernelArgs);
  return ctx->optResult.value_or(optimization_result{});
}

/// @brief Throw an exception instructing the user how to achieve optimal
/// performance
static void throwPerformanceError() {
  throw std::runtime_error(
      "ERROR: Achieving optimal VQE kernel on this platform requires the first "
      "parameter in the kernel to be the variational parameter (list of "
      "floats). Please update your VQE kernel to have list[float] as a its "
      "first parameter\n");
}

/// @brief Run `cudaq.vqe()` without a gradient strategy.
optimization_result pyVQE(py::object &kernel, spin_op &hamiltonian,
                          cudaq::optimizer &optimizer, const int n_params,
                          const int shots = -1) {
  auto &platform = cudaq::get_platform();
  if (platform.get_remote_capabilities().vqe) {
    if (firstArgIsCompatibleWithRemoteVQE(kernel))
      return pyVQE_remote_cpp(platform, kernel, hamiltonian, optimizer,
                              /*gradient=*/nullptr, /*argumentMapper=*/nullptr,
                              n_params, shots);
    throwPerformanceError();
  }
  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    py::args params = py::make_tuple(x);
    observe_result result = pyObserve(kernel, hamiltonian, params, shots);
    double energy = result.expectation();
    return energy;
  });
}

/// @brief Run `cudaq.vqe()` without a gradient strategy, using the
/// user provided `argument_mapper`.
optimization_result pyVQE(py::object &kernel, spin_op &hamiltonian,
                          cudaq::optimizer &optimizer, const int n_params,
                          py::function &argumentMapper, const int shots = -1) {
  auto &platform = cudaq::get_platform();
  if (platform.get_remote_capabilities().vqe) {
    if (firstArgIsCompatibleWithRemoteVQE(kernel))
      return pyVQE_remote_cpp(platform, kernel, hamiltonian, optimizer,
                              /*gradient=*/nullptr, &argumentMapper, n_params,
                              shots);
    throwPerformanceError();
  }
  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    py::args params;
    auto hasToBeTuple = argumentMapper(x);
    if (py::isinstance<py::tuple>(hasToBeTuple))
      params = hasToBeTuple;
    else
      params = py::make_tuple(hasToBeTuple);
    observe_result result = pyObserve(kernel, hamiltonian, params, shots, true);
    double energy = result.expectation();
    return energy;
  });
}

/// @brief Run `cudaq.vqe()` with the provided gradient strategy.
optimization_result pyVQE(py::object &kernel, cudaq::gradient &gradient,
                          spin_op &hamiltonian, cudaq::optimizer &optimizer,
                          const int n_params, const int shots = -1) {
  // Get the expected value of the system, <H> at the provided
  // vector of parameters. This is passed to `cudaq::gradient::compute`
  // to allow for the calculation of the gradient vector with the
  // provided gradient strategy.
  auto &platform = cudaq::get_platform();
  if (platform.get_remote_capabilities().vqe) {
    if (firstArgIsCompatibleWithRemoteVQE(kernel))
      return pyVQE_remote_cpp(platform, kernel, hamiltonian, optimizer,
                              &gradient,
                              /*argumentMapper=*/nullptr, n_params, shots);
    throwPerformanceError();
  }
  std::function<double(std::vector<double>)> get_expected_value =
      [&](std::vector<double> x) {
        py::args params = py::make_tuple(x);
        observe_result result = pyObserve(kernel, hamiltonian, params, shots);
        double energy = result.expectation();
        return energy;
      };
  auto requires_grad = optimizer.requiresGradients();
  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    double energy = get_expected_value(x);
    if (requires_grad) {
      grad_vec = gradient.compute(x, get_expected_value, energy);
    }
    return energy;
  });
}

/// @brief Run `cudaq.vqe()` with the provided gradient strategy,
/// using the provided `argument_mapper`.
optimization_result pyVQE(py::object &kernel, cudaq::gradient &gradient,
                          spin_op &hamiltonian, cudaq::optimizer &optimizer,
                          const int n_params, py::function &argumentMapper,
                          const int shots = -1) {
  // Get the expected value of the system, <H> at the provided
  // vector of parameters. This is passed to `cudaq::gradient::compute`
  // to allow for the calculation of the gradient vector with the
  // provided gradient strategy.
  auto &platform = cudaq::get_platform();
  if (platform.get_remote_capabilities().vqe) {
    if (firstArgIsCompatibleWithRemoteVQE(kernel))
      return pyVQE_remote_cpp(platform, kernel, hamiltonian, optimizer,
                              &gradient, &argumentMapper, n_params, shots);
    throwPerformanceError();
  }
  std::function<double(std::vector<double>)> get_expected_value =
      [&](std::vector<double> x) {
        py::args params;
        auto hasToBeTuple = argumentMapper(x);
        if (py::isinstance<py::tuple>(hasToBeTuple))
          params = hasToBeTuple;
        else
          params = py::make_tuple(hasToBeTuple);
        observe_result result =
            pyObserve(kernel, hamiltonian, params, shots, true);
        double energy = result.expectation();
        return energy;
      };
  auto requires_grad = optimizer.requiresGradients();
  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    double energy = get_expected_value(x);
    if (requires_grad) {
      grad_vec = gradient.compute(x, get_expected_value, energy);
    }
    return energy;
  });
}

void bindVQE(py::module &mod) {
  // FIXME(OperatorCpp): Remove this when the operator class is implemented in
  // C++
  const auto convertToSpinOp = [](py::object &obj) -> cudaq::spin_op {
    if (py::hasattr(obj, "_to_spinop"))
      return obj.attr("_to_spinop")().cast<cudaq::spin_op>();
    return obj.cast<cudaq::spin_op>();
  };

  /// @brief Gradient-Free `cudaq.optimizer` overloads:
  mod.def(
      "vqe",
      [&](py::object &kernel, py::object &spin_operator,
          cudaq::optimizer &optimizer, const int parameter_count,
          const int shots) {
        auto requires_grad = optimizer.requiresGradients();
        if (requires_grad) {
          std::runtime_error("Provided optimizer requires a gradient strategy "
                             "but none was given.\n");
        }
        auto asSpinOp = convertToSpinOp(spin_operator);
        return pyVQE(kernel, asSpinOp, optimizer, parameter_count, shots);
      },
      py::arg("kernel"), py::arg("spin_operator"), py::arg("optimizer"),
      py::arg("parameter_count"), py::arg("shots") = -1, "");

  // With a provided `argument_mapper`.
  mod.def(
      "vqe",
      [&](py::object &kernel, py::object &spin_operator,
          cudaq::optimizer &optimizer, const int parameter_count,
          py::function &argumentMapper, const int shots) {
        auto requires_grad = optimizer.requiresGradients();
        if (requires_grad) {
          std::runtime_error("Provided optimizer requires a gradient strategy "
                             "but none was given.\n");
        }
        auto asSpinOp = convertToSpinOp(spin_operator);
        return pyVQE(kernel, asSpinOp, optimizer, parameter_count,
                     argumentMapper, shots);
      },
      py::arg("kernel"), py::arg("spin_operator"), py::arg("optimizer"),
      py::arg("parameter_count"), py::arg("argument_mapper"),
      py::arg("shots") = -1, "");

  /// @brief Gradient based `cudaq.optimizers` overloads:
  mod.def(
      "vqe",
      [&](py::object &kernel, cudaq::gradient &gradient,
          py::object &spin_operator, cudaq::optimizer &optimizer,
          const int parameter_count, const int shots) {
        auto asSpinOp = convertToSpinOp(spin_operator);
        return pyVQE(kernel, gradient, asSpinOp, optimizer, parameter_count,
                     shots);
      },
      py::arg("kernel"), py::arg("gradient_strategy"), py::arg("spin_operator"),
      py::arg("optimizer"), py::arg("parameter_count"), py::arg("shots") = -1,
      "");

  // With a provided `argument_mapper`.
  mod.def(
      "vqe",
      [&](py::object &kernel, cudaq::gradient &gradient,
          py::object &spin_operator, cudaq::optimizer &optimizer,
          const int parameter_count, py::function &argumentMapper,
          const int shots) {
        auto asSpinOp = convertToSpinOp(spin_operator);
        return pyVQE(kernel, gradient, asSpinOp, optimizer, parameter_count,
                     argumentMapper, shots);
      },
      py::arg("kernel"), py::arg("gradient_strategy"), py::arg("spin_operator"),
      py::arg("optimizer"), py::arg("parameter_count"),
      py::arg("argument_mapper"), py::arg("shots") = -1, "");
}

} // namespace cudaq
