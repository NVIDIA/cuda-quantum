/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/algorithms/draw.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <string>
#include <tuple>
#include <vector>

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

namespace {
std::tuple<std::string, MlirModule, OpaqueArguments *>
getKernelLaunchParameters(py::object &kernel, py::args args) {
  if (py::len(kernel.attr("arguments")) != args.size())
    throw std::runtime_error("Invalid number of arguments passed to draw:" +
                             std::to_string(args.size()) + " expected " +
                             std::to_string(py::len(kernel.attr("arguments"))));

  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args, kernelMod, kernelName);

  return {kernelName, kernelMod, argData};
}
} // namespace

/// @brief Run `cudaq::draw` on the provided kernel.
std::string pyDraw(py::object &kernel, py::args args) {
  auto [kernelName, kernelMod, argData] =
      getKernelLaunchParameters(kernel, args);

  return details::extractTrace([&]() mutable {
    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    delete argData;
  });
}

/// @brief Run `cudaq::draw`'s string overload on the provided kernel.
std::string pyDraw(std::string format, py::object &kernel, py::args args) {
  if (format == "ascii") {
    return pyDraw(kernel, args);
  } else if (format == "latex") {
    auto [kernelName, kernelMod, argData] =
        getKernelLaunchParameters(kernel, args);

    return details::extractTraceLatex([&]() mutable {
      pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
      delete argData;
    });
  } else {
    throw std::runtime_error("Invalid format passed to draw.");
  }
}

/// @brief Bind the draw cudaq function
void bindPyDraw(py::module &mod) {
  mod.def("draw",
          py::overload_cast<std::string, py::object &, py::args>(&pyDraw),
          R"#(Return a string representing the drawing of the execution path, 
in the format specified as the first argument. If the format is 
'ascii', the output will be a UTF-8 encoded string. If the format 
is 'latex', the output will be a LaTeX string.

Args:
  format (str): The format of the output. Can be 'ascii' or 'latex'.
  kernel (:class:`Kernel`): The :class:`Kernel` to draw.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.)#")
      .def(
          "draw", py::overload_cast<py::object &, py::args>(&pyDraw),
          R"#(Return a UTF-8 encoded string representing drawing of the execution 
path, i.e., the trace, of the provided `kernel`.
      
Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to draw.
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
  print(cudaq.draw(bell_pair))
  # Output
  #      ╭───╮     
  # q0 : ┤ h ├──●──
  #      ╰───╯╭─┴─╮
  # q1 : ─────┤ x ├
  #           ╰───╯
  
  # Example with arguments
  import cudaq
  @cudaq.kernel
  def kernel(angle:float):
      q = cudaq.qubit()
      h(q)
      ry(angle, q)
  print(cudaq.draw(kernel, 0.59))
  # Output
  #      ╭───╮╭──────────╮
  # q0 : ┤ h ├┤ ry(0.59) ├
  #      ╰───╯╰──────────╯)#");
}

} // namespace cudaq
