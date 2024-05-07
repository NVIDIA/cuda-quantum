/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq.h"

#include <pybind11/complex.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "runtime/common/py_ExecutionContext.h"
#include "runtime/common/py_NoiseModel.h"
#include "runtime/common/py_ObserveResult.h"
#include "runtime/common/py_SampleResult.h"
#include "runtime/cudaq/algorithms/py_draw.h"
#include "runtime/cudaq/algorithms/py_observe_async.h"
#include "runtime/cudaq/algorithms/py_optimizer.h"
#include "runtime/cudaq/algorithms/py_sample_async.h"
#include "runtime/cudaq/algorithms/py_state.h"
#include "runtime/cudaq/algorithms/py_vqe.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "runtime/cudaq/qis/py_execution_manager.h"
#include "runtime/cudaq/qis/py_qubit_qis.h"
#include "runtime/cudaq/spin/py_matrix.h"
#include "runtime/cudaq/spin/py_spin_op.h"
#include "runtime/cudaq/target/py_runtime_target.h"
#include "runtime/cudaq/target/py_testing_utils.h"
#include "runtime/mlir/py_register_dialects.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"

#include "../runtime/cudaq/platform/orca/orca_qpu.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

namespace py = pybind11;

static std::unique_ptr<cudaq::LinkedLibraryHolder> holder;

namespace cudaq {
const char *getVersion();
const char *getFullRepositoryVersion();
} // namespace cudaq

PYBIND11_MODULE(_quakeDialects, m) {
  holder = std::make_unique<cudaq::LinkedLibraryHolder>();

  cudaq::bindRegisterDialects(m);

  auto cudaqRuntime = m.def_submodule("cudaq_runtime");
  cudaqRuntime.def(
      "registerLLVMDialectTranslation",
      [](MlirContext ctx) {
        mlir::registerLLVMDialectTranslation(*unwrap(ctx));
      },
      "Utility function for Python clients to register all LLVM Dialect "
      "Translation passes with the provided MLIR Context. Primarily used by "
      "kernel_builder and ast_bridge when created new MLIR Contexts.");
  cudaqRuntime.def(
      "initialize_cudaq",
      [&](py::kwargs kwargs) {
        cudaq::info("Calling initialize_cudaq.");
        if (!kwargs)
          return;

        std::map<std::string, std::string> extraConfig;
        for (auto &[keyPy, valuePy] : kwargs) {
          std::string key = py::str(keyPy);
          if (key == "emulate") {
            extraConfig.insert({"emulate", "true"});
            break;
          }
        }

        for (auto &[keyPy, valuePy] : kwargs) {
          std::string key = py::str(keyPy);
          std::string value = py::str(valuePy);
          cudaq::info("Processing Python Arg: {} - {}", key, value);
          if (key == "target")
            holder->setTarget(value, extraConfig);
        }
      },
      "Initialize the CUDA-Q environment.");

  cudaq::bindRuntimeTarget(cudaqRuntime, *holder.get());
  cudaq::bindMeasureCounts(cudaqRuntime);
  cudaq::bindObserveResult(cudaqRuntime);
  cudaq::bindComplexMatrix(cudaqRuntime);
  cudaq::bindSpinWrapper(cudaqRuntime);
  cudaq::bindQIS(cudaqRuntime);
  cudaq::bindOptimizerWrapper(cudaqRuntime);
  cudaq::bindNoise(cudaqRuntime);
  cudaq::bindExecutionContext(cudaqRuntime);
  cudaq::bindExecutionManager(cudaqRuntime);
  cudaq::bindPyState(cudaqRuntime);
  cudaq::bindPyDraw(cudaqRuntime);
  cudaq::bindSampleAsync(cudaqRuntime);
  cudaq::bindObserveAsync(cudaqRuntime);
  cudaq::bindVQE(cudaqRuntime);
  cudaq::bindAltLaunchKernel(cudaqRuntime);
  cudaq::bindTestUtils(cudaqRuntime, *holder.get());

  cudaqRuntime.def("set_random_seed", &cudaq::set_random_seed,
                   "Provide the seed for backend quantum kernel simulation.");
  cudaqRuntime.def("num_available_gpus", &cudaq::num_available_gpus,
                   "The number of available GPUs detected on the system.");

  std::stringstream ss;
  ss << "CUDA-Q Version " << cudaq::getVersion() << " ("
     << cudaq::getFullRepositoryVersion() << ")";
  cudaqRuntime.attr("__version__") = ss.str();

  auto mpiSubmodule = cudaqRuntime.def_submodule("mpi");
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
      "Gather and scatter the `local` list of floating-point numbers, "
      "returning a concatenation of all "
      "lists across all ranks. The total global list size must be provided.");
  mpiSubmodule.def(
      "all_gather",
      [](std::size_t globalVectorSize, std::vector<int> &local) {
        std::vector<int> global(globalVectorSize);
        cudaq::mpi::all_gather(global, local);
        return global;
      },
      "Gather and scatter the `local` list of integers, returning a "
      "concatenation of all "
      "lists across all ranks. The total global list size must be provided.");
  mpiSubmodule.def(
      "broadcast",
      [](std::vector<double> &data, std::size_t bcastSize, int rootRank) {
        if (data.size() < bcastSize)
          data.resize(bcastSize);
        cudaq::mpi::broadcast(data, rootRank);
        return data;
      },
      "Broadcast an array from a process (rootRank) to all other processes. "
      "The size of broadcast array must be provided.");
  mpiSubmodule.def(
      "is_initialized", []() { return cudaq::mpi::is_initialized(); },
      "Returns true if MPI has already been initialized.");
  mpiSubmodule.def(
      "finalize", []() { cudaq::mpi::finalize(); }, "Finalize MPI.");

  auto orcaSubmodule = cudaqRuntime.def_submodule("orca");
  orcaSubmodule.def("sample", &cudaq::orca::sample, "[Documentation TODO]");

  cudaqRuntime.def("cloneModule",
                   [](MlirModule mod) { return wrap(unwrap(mod).clone()); });
  cudaqRuntime.def("isTerminator", [](MlirOperation op) {
    return unwrap(op)->hasTrait<mlir::OpTrait::IsTerminator>();
  });
}
