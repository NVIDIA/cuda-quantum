/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_draw.h"
#include "cudaq/algorithms/draw.h"
#include "cudaq/platform/nvqpp_interface.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"

namespace py = pybind11;

/// @brief Run `cudaq::contrib::draw`'s string overload on the provided kernel.
/// \p kernel is a kernel decorator object and \p args are the arguments to
/// launch \p kernel.
static std::string pyDraw(const std::string &format,
                          const std::string &shortName, MlirModule mod,
                          MlirType retTy, py::args runtimeArgs) {
  if (format != "ascii" && format != "latex")
    throw std::runtime_error("format argument must be \"ascii\" or \"latex\".");

  auto f = [=]() {
    return cudaq::marshal_and_launch_module(shortName, mod, retTy, runtimeArgs);
  };
  if (format == "ascii")
    return cudaq::contrib::extractTrace(std::move(f));
  return cudaq::contrib::extractTraceLatex(std::move(f));
}

/// @brief Bind the draw cudaq function
void cudaq::bindPyDraw(py::module &mod) {
  mod.def(
      "draw_impl",
      [](const std::string &format, const std::string &shortName,
         MlirModule mod, MlirType retTy, py::args runtimeArgs) {
        return pyDraw(format, shortName, mod, retTy, runtimeArgs);
      },
      R"#(
Return a string representing the drawing of the execution path, in the format
specified as the first argument. If the format is 'ascii', the output will be a
UTF-8 encoded string. If the format is 'latex', the output will be a LaTeX
string.

Args:
  format (str): The format of the output. Can be 'ascii' or 'latex'.
  kernel (:class:`Kernel`): The :class:`Kernel` to draw.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
      function at. Leave empty if the kernel doesn't accept any arguments.

Returns:
  The "ascii" format returns a UTF-8 encoded string of the circuit, without
  measurement operations.

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
  #      ╰───╯╰──────────╯

Note: This function is only available when using simulator backends.)#");
}
