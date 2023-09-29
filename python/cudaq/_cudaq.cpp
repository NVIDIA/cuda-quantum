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
#include "runtime/cudaq/target/py_runtime_target.h"
#include "runtime/cudaq/target/py_testing_utils.h"
#include "utils/LinkedLibraryHolder.h"

#include "cudaq.h"

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

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
      "Initialize the CUDA Quantum environment.");

  mod.def("set_random_seed", &cudaq::set_random_seed,
          "Provide the seed for backend quantum kernel simulation.");

  mod.def("num_available_gpus", &cudaq::num_available_gpus,
          "The number of available GPUs detected on the system.");

  auto mpiSubmodule = mod.def_submodule("mpi");
  mpiSubmodule.def(
      "initialize", []() { cudaq::mpi::initialize(); },
      "Initialize MPI if available.");
  mpiSubmodule.def(
      "rank", []() { return cudaq::mpi::rank(); },
      "Return the rank of this process.");
  mpiSubmodule.def(
      "num_ranks", []() { return cudaq::mpi::num_ranks(); },
      "Return the total number of ranks.");
  mpiSubmodule.def(
      "all_gather",
      [](std::size_t globalVectorSize, std::vector<double> &local) {
        std::vector<double> global(globalVectorSize);
        cudaq::mpi::all_gather(global, local);
        return global;
      },
      "Gather and scatter the `local` list, returning a concatenation of all "
      "lists across all ranks. The total global list size must be provided.");
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
  cudaq::bindNoise(mod);
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
