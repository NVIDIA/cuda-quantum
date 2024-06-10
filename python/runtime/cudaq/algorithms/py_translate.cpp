/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/algorithms/draw.h" // TODO  translate.h
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <iostream>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

std::string getQIRLL(const std::string &name, MlirModule module,
                     cudaq::OpaqueArguments &runtimeArgs,
                     std::string &profile);

std::string getASM(const std::string &name, MlirModule module,
                     cudaq::OpaqueArguments &runtimeArgs,
                     std::string &profile);

/// @brief Run `cudaq::translate` on the provided kernel.
std::string pyTranslate(py::object &kernel, py::str format, py::args args) {

  auto formatString = py::cast<std::string>(format);
  std::cout << " === Translating kernel to " << formatString << std::endl;
  
  if (formatString == "mlir") {
      return py::str(kernel).cast<std::string>();
  }

  if (formatString == "qir" || formatString == "qir-base" || formatString == "qir-adaptive") {
    if (py::hasattr(kernel, "compile"))
        kernel.attr("compile")();

    auto profile = formatString == "qir" ? "": formatString;
    auto name = kernel.attr("name").cast<std::string>();
    auto module = kernel.attr("module").cast<MlirModule>();
    args = simplifiedValidateInputArguments(args);
    auto *argData = toOpaqueArgs(args, module, name);
    auto result = getQIRLL(name, module, *argData, profile);
    delete argData;
    return result;
  }

  if (formatString == "openqasm") {
    if (py::len(kernel.attr("arguments")) != args.size())
      throw std::runtime_error("Invalid number of arguments passed to translate.");

    if (py::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    auto name = kernel.attr("name").cast<std::string>();
    auto module = kernel.attr("module").cast<MlirModule>();
    args = simplifiedValidateInputArguments(args);
    auto *argData = toOpaqueArgs(args, module, name);

    auto result = getASM(name, module, *argData, formatString);
    delete argData;
    return result;
  }

  if (formatString == "ascii") {
    // draw
    if (py::len(kernel.attr("arguments")) != args.size())
      throw std::runtime_error("Invalid number of arguments passed to translate.");

    if (py::hasattr(kernel, "compile"))
      kernel.attr("compile")();

    auto kernelName = kernel.attr("name").cast<std::string>();
    auto kernelMod = kernel.attr("module").cast<MlirModule>();
    args = simplifiedValidateInputArguments(args);
    auto *argData = toOpaqueArgs(args, kernelMod, kernelName);

    return details::extractTrace([&]() mutable {
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
      delete argData;
    });
  }

  throw std::runtime_error("Invalid format to translate to: " + formatString);
}

/// @brief Bind the draw cudaq function
void bindPyTranslate(py::module &mod) {
  mod.def(
      "translate", &pyTranslate,
      py::arg("kernel"), py::arg("format") = "mlir", py::kw_only(),
      R"#(Return a UTF-8 encoded string representing drawing of the execution 
path, i.e., the trace, of the provided `kernel`.
      
Args:
  format (str): format to translate to. Available formats: `mlir`, `qir`,
    `qir-base`, `qir-adaptive`, `openqasm`, `ascii`.
  kernel (:class:`Kernel`): The :class:`Kernel` to translate.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.

Returns:
  The UTF-8 encoded string of the circuit, without measurement operations.

.. code-block:: python

  # Example
  import cudaq
  @cudaq.kernel
  def bell_pair():
      q = cudaq.qvector(2)
      h(q[0])
      cx(q[0], q[1])
      mz(q)
  print(cudaq.translate(bell_pair, format:"qir"))

  # Output
  ; ModuleID = 'LLVMDialectModule'
  source_filename = 'LLVMDialectModule'

  %Array = type opaque
  %Result = type opaque
  %Qubit = type opaque

  ...
  ...

  define void @__nvqpp__mlirgen__function_variable_qreg._Z13variable_qregv() local_unnamed_addr {
    %1 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 2)
    %2 = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %1, i64 0)
    %3 = bitcast i8* %2 to %Qubit**
    %4 = load %Qubit*, %Qubit** %3, align 8
    tail call void @__quantum__qis__h(%Qubit* %4)
    %5 = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %1, i64 1)
    %6 = bitcast i8* %5 to %Qubit**
    %7 = load %Qubit*, %Qubit** %6, align 8
    tail call void (i64, void (%Array*, %Qubit*)*, ...) @invokeWithControlQubits(i64 1, void (%Array*, %Qubit*)* nonnull @__quantum__qis__x__ctl, %Qubit* %4, %Qubit* %7)
    %8 = tail call %Result* @__quantum__qis__mz(%Qubit* %4)
    %9 = tail call %Result* @__quantum__qis__mz(%Qubit* %7)
    tail call void @__quantum__rt__qubit_release_array(%Array* %1)
    ret void
  })#");
}

} // namespace cudaq
