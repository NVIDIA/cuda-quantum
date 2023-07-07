/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Logger.h"
#include "runtime/common/py_NoiseModel.h"
#include "runtime/common/py_ObserveResult.h"
#include "runtime/common/py_SampleResult.h"
#include "runtime/cudaq/algorithms/py_observe.h"
#include "runtime/cudaq/algorithms/py_optimizer.h"
#include "runtime/cudaq/algorithms/py_sample.h"
#include "runtime/cudaq/algorithms/py_state.h"
#include "runtime/cudaq/algorithms/py_vqe.h"
#include "runtime/cudaq/builder/py_kernel_builder.h"
#include "runtime/cudaq/kernels/py_chemistry.h"
#include "runtime/cudaq/spin/py_matrix.h"
#include "runtime/cudaq/spin/py_spin_op.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/TestingUtils.h"

#include "cudaq.h"

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

  auto mpiSubmodule = mod.def_submodule("mpi");
  mpiSubmodule.def(
      "initialize", []() { cudaq::mpi::initialize(); },
      "Initialize MPI if available.");
  mpiSubmodule.def(
      "is_initialized", []() { return cudaq::mpi::is_initialized(); },
      "Return true if MPI has already been initialized.");
  mpiSubmodule.def(
      "finalize", []() { cudaq::mpi::finalize(); }, "Finalize MPI.");

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
  cudaq::bindTestUtils(mod, holder);
}
