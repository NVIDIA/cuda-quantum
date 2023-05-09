/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
#include "common/Logger.h"
#include "src/runtime/common/py_NoiseModel.h"
#include "src/runtime/common/py_ObserveResult.h"
#include "src/runtime/common/py_SampleResult.h"
#include "src/runtime/cudaq/algorithms/py_observe.h"
#include "src/runtime/cudaq/algorithms/py_optimizer.h"
#include "src/runtime/cudaq/algorithms/py_sample.h"
#include "src/runtime/cudaq/algorithms/py_state.h"
#include "src/runtime/cudaq/algorithms/py_vqe.h"
#include "src/runtime/cudaq/builder/py_kernel_builder.h"
#include "src/runtime/cudaq/kernels/py_chemistry.h"
#include "src/runtime/cudaq/spin/py_matrix.h"
#include "src/runtime/cudaq/spin/py_spin_op.h"
#include "utils/LinkedLibraryHolder.h"

PYBIND11_MODULE(_pycudaq, mod) {
  static cudaq::LinkedLibraryHolder holder;

  mod.doc() = "Python bindings for CUDA Quantum.";

  mod.def(
      "initialize_cudaq",
      [&](py::kwargs kwargs) {
        cudaq::info("Calling initialize_cudaq.");
        if (!kwargs)
          return;

        for (auto &[keyPy, valuePy] : kwargs) {
          std::string key = py::str(keyPy);
          std::string value = py::str(valuePy);
          cudaq::info("Processing Python Arg: {} - {}", key, value);
          if (key == "target")
            holder.setTarget(value);
        }
      },
      "");

  cudaq::bindRuntimeTarget(mod, holder);
  cudaq::bindBuilder(mod);
  cudaq::bindQuakeValue(mod);
  cudaq::bindObserve(mod);
  cudaq::bindObserveResult(mod);
  cudaq::bindNoiseModel(mod);
  cudaq::bindSample(mod);
  cudaq::bindMeasureCounts(mod);
  cudaq::bindComplexMatrix(mod);
  cudaq::bindSpinWrapper(mod);
  cudaq::bindOptimizerWrapper(mod);
  cudaq::bindVQE(mod);
  cudaq::bindPyState(mod);
  auto kernelSubmodule = mod.def_submodule("kernels");
  cudaq::bindChemistry(kernelSubmodule);
}
