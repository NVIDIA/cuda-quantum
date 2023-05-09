/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
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

#include <pybind11/stl.h>

PYBIND11_MODULE(_pycudaq, mod) {
  static cudaq::LinkedLibraryHolder holder;

  mod.doc() = "Python bindings for CUDA Quantum.";

  py::class_<cudaq::RuntimeTarget>(
      mod, "Target",
      "The `cudaq.Target` represents the underlying infrastructure that CUDA "
      "Quantum kernels will execute on. Instances of `cudaq.Target` describe "
      "what simulator they may leverage, the quantum_platform required for "
      "execution, and a description for the target.")
      .def_readonly("name", &cudaq::RuntimeTarget::name,
                    "The name of the `cudaq.Target`.")
      .def_readonly("simulator", &cudaq::RuntimeTarget::simulatorName,
                    "The name of the simulator this `cudaq.Target` leverages. "
                    "This will be empty for physical QPUs.")
      .def_readonly("platform", &cudaq::RuntimeTarget::simulatorName,
                    "The name of the quantum_platform implementation this "
                    "`cudaq.Target` leverages.")
      .def_readonly("description", &cudaq::RuntimeTarget::simulatorName,
                    "A string describing the features for this `cudaq.Target`.")
      .def("num_qpus", &cudaq::RuntimeTarget::num_qpus,
           "Return the number of QPUs available in this `cudaq.Target`.")
      .def(
          "__str__",
          [](cudaq::RuntimeTarget &self) {
            return fmt::format("Target {}\n\tsimulator={}\n\tplatform={}"
                               "\n\tdescription={}\n",
                               self.name, self.simulatorName, self.platformName,
                               self.description);
          },
          "Persist the information in this `cudaq.Target` to a string.");

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
  mod.def(
      "has_target",
      [&](const std::string &name) { return holder.hasTarget(name); },
      "Return true if the `cudaq.Target` with the given name exists.");
  mod.def(
      "reset_target", [&]() { return holder.resetTarget(); },
      "Reset the current `cudaq.Target` to the default.");
  mod.def(
      "get_target",
      [&](const std::string &name) { return holder.getTarget(name); },
      "Return the `cudaq.Target` with the given name. Will raise an exception "
      "if the name is not valid.");
  mod.def(
      "get_target", [&]() { return holder.getTarget(); },
      "Return the `cudaq.Target` with the given name. Will raise an exception "
      "if the name is not valid.");
  mod.def(
      "get_targets", [&]() { return holder.getTargets(); },
      "Return all available `cudaq.Target` instances on the current system.");
  mod.def(
      "set_target",
      [&](const cudaq::RuntimeTarget &target, py::kwargs extraConfig) {
        std::map<std::string, std::string> config;
        for (auto &[key, value] : extraConfig) {
          if (!py::isinstance<py::str>(value))
            throw std::runtime_error(
                "QPU kwargs config value must be a string.");

          config.emplace(key.cast<std::string>(), value.cast<std::string>());
        }
        holder.setTarget(target.name, config);
      },
      "Set the `cudaq.Target` to be used for CUDA Quantum kernel execution. "
      "Can provide optional, target-specific configuration data via Python "
      "kwargs.");
  mod.def(
      "set_target",
      [&](const std::string &name, py::kwargs extraConfig) {
        std::map<std::string, std::string> config;
        for (auto &[key, value] : extraConfig) {
          if (!py::isinstance<py::str>(value))
            throw std::runtime_error(
                "QPU kwargs config value must be a string.");

          config.emplace(key.cast<std::string>(), value.cast<std::string>());
        }
        holder.setTarget(name, config);
      },
      "Set the `cudaq.Target` with given name to be used for CUDA Quantum "
      "kernel execution. Can provide optional, target-specific configuration "
      "data via Python kwargs.");

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
