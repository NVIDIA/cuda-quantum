/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <optional>
#include <pybind11/numpy.h>
#include "py_EvolveResult.h"
#include "common/EvolveResult.h"
#include "cudaq/algorithms/evolve.h"

namespace py = pybind11;

namespace cudaq {
/// @brief Bind the `cudaq::evolve_result` and `cudaq::async_evolve_result`
/// data classes to python as `cudaq.EvolveResult` and `cudaq.AsyncEvolveResult`.
void bindEvolveResult(py::module &mod) {
  py::class_<evolve_result>(
      mod, "EvolveResult",
      "Stores the execution data from an invocation of :func:`evolve`.\n")
      .def(py::init<state>())
      .def(
          "get_final_state", [](evolve_result &self) { return self.get_final_state(); },
          "Stores the final state produced by a call to :func:`evolve`. "
          "Represent the state of a quantum system after time evolution under "
          "a set of operators, see the :func:`evolve` documentation for more "
          "detail.\n")
      .def(
          "intermediate_states", [](evolve_result &self) { return self.get_intermediate_states(); },
          "Stores all intermediate states, meaning the state after each step "
          "in a defined schedule, produced by a call to :func:`evolve`, "
          "including the final state. This property is only populated if "
          "saving intermediate results was requested in the call to "
          ":func:`evolve`.\n");

  py::class_<async_evolve_result>(
      mod, "AsyncEvolveResult",
      "Stores the execution data from an invocation of :func:`evolve_async`.\n")
      .def(
          "get", [](async_evolve_result &self) { return self.get(); },
          py::call_guard<py::gil_scoped_release>(),
          "Return the evolution result from the asynchronous evolve execution\n.");
}

} // namespace cudaq
