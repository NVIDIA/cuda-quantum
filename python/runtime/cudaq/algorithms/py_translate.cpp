/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/algorithms/draw.h" // TODO  translate.h
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <iostream>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace cudaq {
std::string getQIR(const std::string &name, MlirModule module,
                   cudaq::OpaqueArguments &runtimeArgs,
                   const std::string &profile);

std::string getASM(const std::string &name, MlirModule module,
                   cudaq::OpaqueArguments &runtimeArgs);

/// @brief Run `cudaq::translate` on the provided kernel.
std::string pyTranslate(py::object &kernel, py::args args,
                        const std::string &format) {

  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto name = kernel.attr("name").cast<std::string>();
  auto module = kernel.attr("module").cast<MlirModule>();

  auto result =
      llvm::StringSwitch<std::function<std::string()>>(format)
          .Case("qir",
                [&]() {
                  cudaq::OpaqueArguments args;
                  std::string profile = "";
                  return getQIR(name, module, args, profile);
                })
          .Cases("qir-adaptive", "qir-base",
                 [&]() {
                   cudaq::OpaqueArguments args;
                   return getQIR(name, module, args, format);
                 })
          .Case("openqasm2",
                [&]() {
                  if (py::hasattr(kernel, "arguments") &&
                      py::len(kernel.attr("arguments")) > 0) {
                    throw std::runtime_error("Cannot translate function with "
                                             "arguments to OpenQASM 2.0.");
                  }
                  cudaq::OpaqueArguments args;
                  return getASM(name, module, args);
                })
          .Default([&]() {
            throw std::runtime_error("Invalid format to translate to: " +
                                     format);
            return "Failed to translate to " + format;
          })();

  return result;
}

/// @brief Bind the translate cudaq function
void bindPyTranslate(py::module &mod) {
  mod.def(
      "translate", &pyTranslate, py::arg("kernel"), py::kw_only(),
      py::arg("format") = "qir",
      R"#(Return a UTF-8 encoded string representing drawing of the execution
path, i.e., the trace, of the provided `kernel`.

Args:
  format (str): format to translate to. Available formats: `qir`, `qir-base`,
    `qir-adaptive`, `openqasm2`.
  kernel (:class:`Kernel`): The :class:`Kernel` to translate.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel
    function at. Leave empty if the kernel doesn't accept any arguments.
  Note: Translating functions with arguments to OpenQASM 2.0 is not supported.

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
  print(cudaq.translate(bell_pair, format="qir"))

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
    ...
    %8 = tail call %Result* @__quantum__qis__mz(%Qubit* %4)
    %9 = tail call %Result* @__quantum__qis__mz(%Qubit* %7)
    tail call void @__quantum__rt__qubit_release_array(%Array* %1)
    ret void
  })#");
}

} // namespace cudaq
