/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "py_utils.h"
#include "py_vqe.h"

#include "common/JsonConvert.h"
#include "common/RemoteKernelExecutor.h"
#include "common/SerializedCodeExecutionContext.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/algorithms/gradient.h"
#include "cudaq/algorithms/optimizer.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

/// @brief Return the quantum kernel `FuncOp` from the given `ModuleOp`.
mlir::func::FuncOp getKernelFuncOp(mlir::ModuleOp &mod,
                                   const std::string &kernelName) {
  mlir::func::FuncOp kernel;
  mod.walk([&](mlir::func::FuncOp func) {
    if (func.getName().equals("__nvqpp__mlirgen__" + kernelName))
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

/// @brief Run `cudaq::observe` on the provided kernel and spin operator.
observe_result pyObserve(py::object &kernel, spin_op &spin_operator,
                         py::args args, const int shots,
                         bool argMapperProvided = false) {
  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();
  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
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

static optimization_result
pyVQE_remote(cudaq::quantum_platform &platform, py::object &kernel,
             spin_op &hamiltonian, cudaq::optimizer &optimizer,
             cudaq::gradient *gradient, py::function *argumentMapper,
             const int n_params, const int shots) {
  py::object pickle = py::module_::import("pickle");
  py::object base64 = py::module_::import("base64");
  py::object inspect = py::module_::import("inspect");

  // Form scoped_vars_str
  py::dict scoped_vars;
  // We only need to capture user variables when an argumentMapper is provided.
  if (argumentMapper)
    scoped_vars = get_serializable_var_dict();
  scoped_vars["__hamiltonian"] = pickle.attr("dumps")(hamiltonian);
  scoped_vars["__optimizer"] = pickle.attr("dumps")(optimizer);
  scoped_vars["__vqe_kernel"] = pickle.attr("dumps")(kernel);
  if (gradient)
    scoped_vars["__gradient"] = pickle.attr("dumps")(gradient);
  auto scoped_vars_str = b64encode_dict(scoped_vars);

  // Form SerializedCodeExecutionContext.source_code
  std::ostringstream os;
  if (argumentMapper) {
    std::string source_code = cudaq::get_source_code(*argumentMapper);
    os << "__arg_mapper = " << source_code << '\n';
  }
  os << "energy, params_at_energy = cudaq.vqe(";
  os << "kernel=__vqe_kernel, ";
  if (gradient)
    os << "gradient_strategy=__gradient, ";
  os << "spin_operator=__hamiltonian, ";
  os << "optimizer=__optimizer, ";
  os << "parameter_count=" << n_params << ", ";
  if (argumentMapper)
    os << "argument_mapper=__arg_mapper, ";
  os << "shots=" << shots << ")\n";
  os << "_json_request_result['executionContext']['optResult'] = [energy, "
        "params_at_energy]\n";
  auto function_call = os.str();

  SerializedCodeExecutionContext scCtx;
  scCtx.imports = "import cudaq\n";
  scCtx.scoped_var_dict = std::move(scoped_vars_str);
  scCtx.source_code = std::move(function_call);

  auto ctx = std::make_unique<cudaq::ExecutionContext>("sample", 0);
  platform.set_exec_ctx(ctx.get());
  platform.launchSerializedCodeExecution(
      kernel.attr("name").cast<std::string>(), scCtx);
  platform.reset_exec_ctx();
  auto result = cudaq::optimization_result{};
  if (ctx->optResult)
    result = std::move(*ctx->optResult);
  return result;
}

/// @brief Run `cudaq.vqe()` without a gradient strategy.
optimization_result pyVQE(py::object &kernel, spin_op &hamiltonian,
                          cudaq::optimizer &optimizer, const int n_params,
                          const int shots = -1) {
  auto &platform = cudaq::get_platform();
  if (platform.supports_remote_serialized_code())
    return pyVQE_remote(platform, kernel, hamiltonian, optimizer,
                        /*gradient=*/nullptr, /*argumentMapper=*/nullptr,
                        n_params, shots);
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
  if (platform.supports_remote_serialized_code())
    return pyVQE_remote(platform, kernel, hamiltonian, optimizer,
                        /*gradient=*/nullptr, &argumentMapper, n_params, shots);
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
  if (platform.supports_remote_serialized_code())
    return pyVQE_remote(platform, kernel, hamiltonian, optimizer, &gradient,
                        /*argumentMapper=*/nullptr, n_params, shots);
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
  if (platform.supports_remote_serialized_code())
    return pyVQE_remote(platform, kernel, hamiltonian, optimizer, &gradient,
                        &argumentMapper, n_params, shots);
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
  /// @brief Gradient-Free `cudaq.optimizer` overloads:
  mod.def(
      "vqe",
      [](py::object &kernel, cudaq::spin_op &spin_operator,
         cudaq::optimizer &optimizer, const int parameter_count,
         const int shots) {
        auto requires_grad = optimizer.requiresGradients();
        if (requires_grad) {
          std::runtime_error("Provided optimizer requires a gradient strategy "
                             "but none was given.\n");
        }
        return pyVQE(kernel, spin_operator, optimizer, parameter_count, shots);
      },
      py::arg("kernel"), py::arg("spin_operator"), py::arg("optimizer"),
      py::arg("parameter_count"), py::arg("shots") = -1, "");

  // With a provided `argument_mapper`.
  mod.def(
      "vqe",
      [](py::object &kernel, cudaq::spin_op &spin_operator,
         cudaq::optimizer &optimizer, const int parameter_count,
         py::function &argumentMapper, const int shots) {
        auto requires_grad = optimizer.requiresGradients();
        if (requires_grad) {
          std::runtime_error("Provided optimizer requires a gradient strategy "
                             "but none was given.\n");
        }
        return pyVQE(kernel, spin_operator, optimizer, parameter_count,
                     argumentMapper, shots);
      },
      py::arg("kernel"), py::arg("spin_operator"), py::arg("optimizer"),
      py::arg("parameter_count"), py::arg("argument_mapper"),
      py::arg("shots") = -1, "");

  /// @brief Gradient based `cudaq.optimizers` overloads:
  mod.def(
      "vqe",
      [](py::object &kernel, cudaq::gradient &gradient,
         cudaq::spin_op &spin_operator, cudaq::optimizer &optimizer,
         const int parameter_count, const int shots) {
        return pyVQE(kernel, gradient, spin_operator, optimizer,
                     parameter_count, shots);
      },
      py::arg("kernel"), py::arg("gradient_strategy"), py::arg("spin_operator"),
      py::arg("optimizer"), py::arg("parameter_count"), py::arg("shots") = -1,
      "");

  // With a provided `argument_mapper`.
  mod.def(
      "vqe",
      [](py::object &kernel, cudaq::gradient &gradient,
         cudaq::spin_op &spin_operator, cudaq::optimizer &optimizer,
         const int parameter_count, py::function &argumentMapper,
         const int shots) {
        return pyVQE(kernel, gradient, spin_operator, optimizer,
                     parameter_count, argumentMapper, shots);
      },
      py::arg("kernel"), py::arg("gradient_strategy"), py::arg("spin_operator"),
      py::arg("optimizer"), py::arg("parameter_count"),
      py::arg("argument_mapper"), py::arg("shots") = -1, "");
}

} // namespace cudaq
