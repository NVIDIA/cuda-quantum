/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "common/ArgumentWrapper.h"
#include "common/JsonConvert.h"
#include "common/SerializedCodeExecutionContext.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/algorithms/gradient.h"
#include "cudaq/algorithms/optimizer.h"
#include "py_utils.h"
#include "py_vqe.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

void *pyGetKernelArgs(const std::string &name, MlirModule module,
                      cudaq::OpaqueArguments &runtimeArgs,
                      const std::vector<std::string> &names,
                      std::size_t startingArgIdx);

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
  ctx->spin = &hamiltonian;
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
    cudaq::packArgs(
        args, runtimeArgs, kernelFunc,
        [](OpaqueArguments &, py::object &) { return false; }, startingArgIdx);
  }
  kernelArgs = pyGetKernelArgs(kernelName, kernelMod, args, /*names=*/{},
                               startingArgIdx);

  // Need to form cudaq::ArgWrapper and pass that into launchVQE.
  std::vector<std::string> names;
  auto *wrapper = new cudaq::ArgWrapper{unwrap(kernelMod), names, kernelArgs};

  platform.launchVQE(kernelName, wrapper, gradient, hamiltonian, optimizer,
                     n_params, shots);
  platform.reset_exec_ctx();
  delete wrapper;
  if (kernelArgs)
    std::free(kernelArgs);
  return ctx->optResult.value_or(optimization_result{});
}

/// @brief Perform VQE on a remote platform. This function is used for many of
/// the pyVQE variants below, so some of the parameters may be nullptr.
static optimization_result
pyVQE_remote(cudaq::quantum_platform &platform, py::object &kernel,
             spin_op &hamiltonian, cudaq::optimizer &optimizer,
             cudaq::gradient *gradient, py::function *argumentMapper,
             const int n_params, const int shots) {
  py::object json = py::module_::import("json");
  py::object inspect = py::module_::import("inspect");

  // Form scoped_vars_str. This is needed for a) capturing user variables when
  // an argumentMapper is provided, and b) automatically capturing all nested
  // cudaq.kernels.
  py::dict scoped_vars = get_serializable_var_dict();

// This macro loads a JSON-like object into scoped_vars[] as
// scoped_vars["__varname"] = varname. This roughly corresponds to the
// following Python code:
//  scoped_vars["__varname/module.name"] = json.loads(varname.to_json())
#define LOAD_VAR(VAR_NAME)                                                     \
  do {                                                                         \
    py::object val = py::cast(VAR_NAME);                                       \
    scoped_vars[py::str(                                                       \
        std::string("__" #VAR_NAME "/") +                                      \
        val.get_type().attr("__module__").cast<std::string>() + "." +          \
        val.get_type().attr("__name__").cast<std::string>())] =                \
        json.attr("loads")(val.attr("to_json")());                             \
  } while (0)
#define LOAD_VAR_NO_CAST(VAR_NAME)                                             \
  do {                                                                         \
    scoped_vars[py::str(                                                       \
        std::string("__" #VAR_NAME "/") +                                      \
        VAR_NAME.get_type().attr("__module__").cast<std::string>() + "." +     \
        VAR_NAME.get_type().attr("__name__").cast<std::string>())] =           \
        json.attr("loads")(VAR_NAME.attr("to_json")());                        \
  } while (0)
  LOAD_VAR(hamiltonian);
  LOAD_VAR(optimizer);
  LOAD_VAR_NO_CAST(kernel);
  if (gradient)
    LOAD_VAR(gradient);

  // Get a string representation of the scoped_vars dictionary. This is
  // guaranteed to be a JSON-friendly dictionary, so the conversion should occur
  // cleanly.
  auto scoped_vars_str = json.attr("dumps")(scoped_vars).cast<std::string>();

  // Form SerializedCodeExecutionContext.source_code
  std::ostringstream os;
  if (argumentMapper) {
    std::string source_code = cudaq::get_source_code(*argumentMapper);
    // If it is a lambda function and it is used inline with a function call, it
    // can sometimes include the trailing comma. Remove that here.
    auto end = source_code.find_last_not_of(", \t\r\n");
    if (end != std::string::npos)
      source_code.erase(end + 1);
    os << "__arg_mapper = " << source_code << '\n';
  }
  os << "energy, params_at_energy = cudaq.vqe(";
  os << "kernel=__kernel, ";
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
  if (platform.get_remote_capabilities().serializedCodeExec)
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
  if (platform.get_remote_capabilities().vqe) {
    if (firstArgIsCompatibleWithRemoteVQE(kernel))
      return pyVQE_remote_cpp(platform, kernel, hamiltonian, optimizer,
                              /*gradient=*/nullptr, &argumentMapper, n_params,
                              shots);
    throwPerformanceError();
  }
  if (platform.get_remote_capabilities().serializedCodeExec)
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
  if (platform.get_remote_capabilities().vqe) {
    if (firstArgIsCompatibleWithRemoteVQE(kernel))
      return pyVQE_remote_cpp(platform, kernel, hamiltonian, optimizer,
                              &gradient,
                              /*argumentMapper=*/nullptr, n_params, shots);
    throwPerformanceError();
  }
  if (platform.get_remote_capabilities().serializedCodeExec)
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
  if (platform.get_remote_capabilities().vqe) {
    if (firstArgIsCompatibleWithRemoteVQE(kernel))
      return pyVQE_remote_cpp(platform, kernel, hamiltonian, optimizer,
                              &gradient, &argumentMapper, n_params, shots);
    throwPerformanceError();
  }
  if (platform.get_remote_capabilities().serializedCodeExec)
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
