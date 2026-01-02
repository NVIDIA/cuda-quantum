/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/draw.h"
#include "cudaq/algorithms/unitary.h"
#include "runtime/cudaq/algorithms/py_draw.h"
#include "runtime/cudaq/operators/py_helpers.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace cudaq {

py::array pyGetUnitary(py::object &kernel, py::args args) {
  // Prepare kernel launch parameters (see py_draw.cpp for pattern)
  auto [kernelName, kernelMod, argData] =
      details::getKernelLaunchParameters(kernel, args);

  // Compute the unitary
  auto cmat = contrib::get_unitary_cmat([&]() mutable {
    pyAltLaunchKernel(kernelName, kernelMod, *argData, {});
    delete argData;
  });
  // Return as numpy array (dim, dim), complex128
  return details::cmat_to_numpy(cmat);
}

/// @brief Bind the get_unitary cudaq function
void bindPyUnitary(py::module &mod) {
  mod.def(
      "get_unitary", &pyGetUnitary,
      R"#(Return the unitary matrix of the execution path of the provided kernel.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to analyze.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel at.

Returns:
  numpy.ndarray: The unitary matrix as a complex-valued NumPy array.

.. code-block:: python

  import cudaq
  @cudaq.kernel
  def bell():
      q = cudaq.qvector(2)
      h(q[0])
      cx(q[0], q[1])
  U = cudaq.get_unitary(bell)
  print(U)
)#");
}

} // namespace cudaq
