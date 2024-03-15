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

#include <iostream>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace cudaq {

void pyAltLaunchKernel(const std::string &, MlirModule, OpaqueArguments &,
                       const std::vector<std::string> &);

/// @brief Run `cudaq::draw` on the provided kernel.
std::string pyDraw(py::object &kernel, py::args args) {

  if (py::len(kernel.attr("arguments")) != args.size())
    throw std::runtime_error("Invalid number of arguments passed to draw.");

  if (py::hasattr(kernel, "compile"))
    kernel.attr("compile")();

  auto kernelName = kernel.attr("name").cast<std::string>();
  auto kernelMod = kernel.attr("module").cast<MlirModule>();
  args = simplifiedValidateInputArguments(args);
  auto *argData = toOpaqueArgs(args);

  return details::extractTrace([&]() mutable {
    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    delete argData;
  });
}

/// @brief Bind the draw cudaq function
void bindPyDraw(py::module &mod) {
  mod.def(
      "draw", &pyDraw,
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
