/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq.h"
#include "cudaq/Support/Version.h"
#include "cudaq/platform/orca/orca_qpu.h"
#include "runtime/common/py_AnalogHamiltonian.h"
#include "runtime/common/py_CustomOpRegistry.h"
#include "runtime/common/py_EvolveResult.h"
#include "runtime/common/py_ExecutionContext.h"
#include "runtime/common/py_NoiseModel.h"
#include "runtime/common/py_ObserveResult.h"
#include "runtime/common/py_Resources.h"
#include "runtime/common/py_SampleResult.h"
#include "runtime/cudaq/algorithms/py_draw.h"
#include "runtime/cudaq/algorithms/py_evolve.h"
#include "runtime/cudaq/algorithms/py_observe_async.h"
#include "runtime/cudaq/algorithms/py_optimizer.h"
#include "runtime/cudaq/algorithms/py_resource_count.h"
#include "runtime/cudaq/algorithms/py_run.h"
#include "runtime/cudaq/algorithms/py_sample_async.h"
#include "runtime/cudaq/algorithms/py_state.h"
#include "runtime/cudaq/algorithms/py_translate.h"
#include "runtime/cudaq/algorithms/py_unitary.h"
#include "runtime/cudaq/algorithms/py_utils.h"
#include "runtime/cudaq/operators/py_boson_op.h"
#include "runtime/cudaq/operators/py_fermion_op.h"
#include "runtime/cudaq/operators/py_handlers.h"
#include "runtime/cudaq/operators/py_matrix.h"
#include "runtime/cudaq/operators/py_matrix_op.h"
#include "runtime/cudaq/operators/py_scalar_op.h"
#include "runtime/cudaq/operators/py_spin_op.h"
#include "runtime/cudaq/operators/py_super_op.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "runtime/cudaq/qis/py_execution_manager.h"
#include "runtime/cudaq/qis/py_qubit_qis.h"
#include "runtime/cudaq/target/py_runtime_target.h"
#include "runtime/cudaq/target/py_testing_utils.h"
#include "runtime/interop/PythonCppInterop.h"
#include "runtime/mlir/py_register_dialects.h"
#include "utils/LinkedLibraryHolder.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include <pybind11/complex.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace cudaq;

static std::unique_ptr<LinkedLibraryHolder> holder;

PYBIND11_MODULE(_quakeDialects, m) {
  holder = std::make_unique<LinkedLibraryHolder>();

  bindRegisterDialects(m);

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
        CUDAQ_INFO("Calling initialize_cudaq.");
        if (!kwargs)
          return;

        std::map<std::string, std::string> extraConfig;
        for (auto &[keyPy, valuePy] : kwargs) {
          std::string key = py::str(keyPy);
          if (key == "emulate") {
            extraConfig.insert({"emulate", "true"});
          }
          if (key == "option") {
            extraConfig.insert({"option", py::str(valuePy)});
          }
        }

        for (auto &[keyPy, valuePy] : kwargs) {
          std::string key = py::str(keyPy);
          std::string value = py::str(valuePy);
          CUDAQ_INFO("Processing Python Arg: {} - {}", key, value);
          if (key == "target")
            holder->setTarget(value, extraConfig);
        }
      },
      "Initialize the CUDA-Q environment.");

  bindRuntimeTarget(cudaqRuntime, *holder.get());
  bindMeasureCounts(cudaqRuntime);
  bindResources(cudaqRuntime);
  bindObserveResult(cudaqRuntime);
  bindComplexMatrix(cudaqRuntime);
  bindScalarWrapper(cudaqRuntime);
  bindSpinWrapper(cudaqRuntime);
  bindFermionWrapper(cudaqRuntime);
  bindBosonWrapper(cudaqRuntime);
  bindOperatorsWrapper(cudaqRuntime);
  bindHandlersWrapper(cudaqRuntime);
  bindSuperOperatorWrapper(cudaqRuntime);
  bindQIS(cudaqRuntime);
  bindOptimizerWrapper(cudaqRuntime);
  bindNoise(cudaqRuntime);
  bindExecutionContext(cudaqRuntime);
  bindExecutionManager(cudaqRuntime);
  bindPyState(cudaqRuntime, *holder.get());
  bindPyDataClassRegistry(cudaqRuntime);
  bindPyEvolve(cudaqRuntime);
  bindEvolveResult(cudaqRuntime);
  bindPyDraw(cudaqRuntime);
  bindPyUnitary(cudaqRuntime);
  bindPyRun(cudaqRuntime);
  bindPyRunAsync(cudaqRuntime);
  bindPyTranslate(cudaqRuntime);
  bindCountResources(cudaqRuntime);
  bindSampleAsync(cudaqRuntime);
  bindObserveAsync(cudaqRuntime);
  bindAltLaunchKernel(cudaqRuntime, [holderPtr = holder.get()]() {
    return python::getTransportLayer(holderPtr);
  });
  bindTestUtils(cudaqRuntime, *holder.get());
  bindCustomOpRegistry(cudaqRuntime);

  cudaqRuntime.def("set_random_seed", &set_random_seed,
                   "Provide the seed for backend quantum kernel simulation.");
  cudaqRuntime.def("num_available_gpus", &num_available_gpus,
                   "The number of available GPUs detected on the system.");

  std::stringstream ss;
  ss << "CUDA-Q Version " << getVersion() << " (" << getFullRepositoryVersion()
     << ")";
  cudaqRuntime.attr("__version__") = ss.str();

  auto mpiSubmodule = cudaqRuntime.def_submodule("mpi");
  mpiSubmodule.def(
      "initialize", []() { mpi::initialize(); },
      "Initialize MPI if available.");
  mpiSubmodule.def(
      "rank", []() { return mpi::rank(); }, "Return the rank of this process.");
  mpiSubmodule.def(
      "num_ranks", []() { return mpi::num_ranks(); },
      "Return the total number of ranks.");
  mpiSubmodule.def(
      "all_gather",
      [](std::size_t globalVectorSize, std::vector<double> &local) {
        std::vector<double> global(globalVectorSize);
        mpi::all_gather(global, local);
        return global;
      },
      "Gather and scatter the `local` list of floating-point numbers, "
      "returning a concatenation of all "
      "lists across all ranks. The total global list size must be provided.");
  mpiSubmodule.def(
      "all_gather",
      [](std::size_t globalVectorSize, std::vector<int> &local) {
        std::vector<int> global(globalVectorSize);
        mpi::all_gather(global, local);
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
        mpi::broadcast(data, rootRank);
        return data;
      },
      "Broadcast an array from a process (rootRank) to all other processes. "
      "The size of broadcast array must be provided.");
  mpiSubmodule.def(
      "is_initialized", []() { return mpi::is_initialized(); },
      "Returns true if MPI has already been initialized.");
  mpiSubmodule.def(
      "finalize", []() { mpi::finalize(); }, "Finalize MPI.");
  mpiSubmodule.def(
      "comm_dup",
      []() {
        const auto [commPtr, commSize] = mpi::comm_dup();
        return std::make_pair(reinterpret_cast<intptr_t>(commPtr), commSize);
      },
      "Duplicates the communicator. Return the new communicator address (as an "
      "integer) and its size in bytes");

  auto orcaSubmodule = cudaqRuntime.def_submodule("orca");
  orcaSubmodule.def(
      "sample",
      py::overload_cast<std::vector<std::size_t> &, std::vector<std::size_t> &,
                        std::vector<double> &, std::vector<double> &, int,
                        std::size_t>(&orca::sample),
      "Performs Time Bin Interferometer (TBI) boson sampling experiments on "
      "ORCA's backends",
      py::arg("input_state"), py::arg("loop_lengths"), py::arg("bs_angles"),
      py::arg("ps_angles"), py::arg("n_samples") = 10000,
      py::arg("qpu_id") = 0);
  orcaSubmodule.def(
      "sample",
      py::overload_cast<std::vector<std::size_t> &, std::vector<std::size_t> &,
                        std::vector<double> &, int, std::size_t>(&orca::sample),
      "Performs Time Bin Interferometer (TBI) boson sampling experiments on "
      "ORCA's backends",
      py::arg("input_state"), py::arg("loop_lengths"), py::arg("bs_angles"),
      py::arg("n_samples") = 10000, py::arg("qpu_id") = 0);
  orcaSubmodule.def(
      "sample_async",
      py::overload_cast<std::vector<std::size_t> &, std::vector<std::size_t> &,
                        std::vector<double> &, std::vector<double> &, int,
                        std::size_t>(&orca::sample_async),
      "Performs Time Bin Interferometer (TBI) boson sampling experiments on "
      "ORCA's backends",
      py::arg("input_state"), py::arg("loop_lengths"), py::arg("bs_angles"),
      py::arg("ps_angles"), py::arg("n_samples") = 10000,
      py::arg("qpu_id") = 0);
  orcaSubmodule.def(
      "sample_async",
      py::overload_cast<std::vector<std::size_t> &, std::vector<std::size_t> &,
                        std::vector<double> &, int, std::size_t>(
          &orca::sample_async),
      "Performs Time Bin Interferometer (TBI) boson sampling experiments on "
      "ORCA's backends",
      py::arg("input_state"), py::arg("loop_lengths"), py::arg("bs_angles"),
      py::arg("n_samples") = 10000, py::arg("qpu_id") = 0);

  auto photonicsSubmodule = cudaqRuntime.def_submodule("photonics");
  photonicsSubmodule.def(
      "allocate_qudit",
      [](std::size_t &level) {
        return getExecutionManager()->allocateQudit(level);
      },
      "Allocate a qudit of given level.", py::arg("level"));
  photonicsSubmodule.def(
      "apply_operation",
      [](const std::string &name, std::vector<double> &params,
         std::vector<std::vector<std::size_t>> &targets) {
        std::vector<QuditInfo> targetInfo;
        for (auto &t : targets) {
          if (t.size() != 2)
            throw std::runtime_error("Invalid qudit target");
          targetInfo.emplace_back(t[0], t[1]);
        }
        getExecutionManager()->apply(name, params, {}, targetInfo, false,
                                     spin_op::identity());
      },
      "Apply the input photonics operation on the target qudits.",
      py::arg("name"), py::arg("params"), py::arg("targets"));
  photonicsSubmodule.def(
      "measure",
      [](std::size_t level, std::size_t id, const std::string &regName) {
        return getExecutionManager()->measure(QuditInfo(level, id), regName);
      },
      "Measure the input qudit(s).", py::arg("level"), py::arg("qudit"),
      py::arg("register_name") = "");
  photonicsSubmodule.def(
      "release_qudit",
      [](std::size_t level, std::size_t id) {
        getExecutionManager()->returnQudit(QuditInfo(level, id));
      },
      "Release a qudit of given id.", py::arg("level"), py::arg("id"));
  cudaqRuntime.def("cloneModule",
                   [](MlirModule mod) { return wrap(unwrap(mod).clone()); });
  cudaqRuntime.def("isTerminator", [](MlirOperation op) {
    return unwrap(op)->hasTrait<mlir::OpTrait::IsTerminator>();
  });

  auto ahsSubmodule = cudaqRuntime.def_submodule("ahs");
  bindAnalogHamiltonian(ahsSubmodule);

  cudaqRuntime.def(
      "isRegisteredDeviceModule",
      [](const std::string &name) {
        return python::isRegisteredDeviceModule(name);
      },
      "Return true if the input name (mod1.mod2...) is a registered C++ device "
      "module.");

  cudaqRuntime.def(
      "checkRegisteredCppDeviceKernel",
      [](MlirModule mod,
         const std::string &moduleName) -> std::optional<std::string> {
        std::tuple<std::string, std::string> ret;
        try {
          ret = python::getDeviceKernel(moduleName);
        } catch (...) {
          return std::nullopt;
        }

        // Take the code for the kernel we found
        // and add it to the input module, return
        // the func op.
        auto [kName, code] = ret;
        auto ctx = unwrap(mod).getContext();
        auto moduleB = mlir::parseSourceString<mlir::ModuleOp>(code, ctx);
        auto moduleA = unwrap(mod);

        // Merge symbols from moduleB into moduleA.
        opt::factory::mergeModules(moduleA, *moduleB);
        return kName;
      },
      "Given a python module name like `mod1.mod2.func`, see if there is a "
      "registered C++ quantum kernel. If so, add the kernel to the Module and "
      "return its name.");

  cudaqRuntime.def(
      "appendKernelArgument",
      [](MlirOperation op, MlirType type) -> MlirValue {
        auto func = cast<mlir::func::FuncOp>(unwrap(op));
        auto ty = unwrap(type);
        auto funcTy = func.getFunctionType();
        mlir::SmallVector<mlir::Type> inpTy{funcTy.getInputs().begin(),
                                            funcTy.getInputs().end()};
        auto resTy = funcTy.getResults();
        inpTy.push_back(ty);
        auto *ctx = ty.getContext();
        func.setFunctionType(mlir::FunctionType::get(ctx, inpTy, resTy));
        auto &blk = func.getBody().front();
        auto pos = blk.getNumArguments();
        auto result = blk.addArgument(ty, mlir::UnknownLoc::get(ctx));
        // Add an attribute so we know PyBridge added them. This is convoluted
        // because of a bug in MLIR that doesn't let one add an attribute to a
        // new argument if one of the arguments already has an attribute.
        mlir::SmallVector<mlir::DictionaryAttr> allArgAttrs;
        for (unsigned i = 0; i < pos; ++i)
          allArgAttrs.push_back(func.getArgAttrDict(i));
        mlir::SmallVector<mlir::NamedAttribute> aa;
        aa.emplace_back(mlir::StringAttr::get(ctx, "quake.pylifted"),
                        mlir::UnitAttr::get(ctx));
        allArgAttrs.push_back(mlir::DictionaryAttr::get(ctx, aa));
        mlir::function_interface_impl::setAllArgAttrDicts(func, allArgAttrs);
        return wrap(result);
      },
      "Adds missing standard functionality to add arguments to a FuncOp.");

  cudaqRuntime.def(
      "isdeclaration",
      [](MlirOperation func) -> bool {
        mlir::Operation *op = unwrap(func);
        if (auto fn = mlir::dyn_cast_or_null<mlir::func::FuncOp>(op))
          return fn.empty();
        return false;
      },
      "Is the FuncOp `func` a declaration?");

  cudaqRuntime.def(
      "updateModule",
      [](const std::string &name, MlirModule to, MlirModule from) {
        auto toMod = unwrap(to);
        // Erase the specified symbol op, if it exists.
        if (!name.empty())
          if (auto *sop = toMod.lookupSymbol(name))
            sop->erase();
        opt::factory::mergeModules(toMod, unwrap(from));
      },
      "Merge the `from` module into the `to` module, overwriting `name`.");
}
