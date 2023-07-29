/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_runtime_target.h"
#include "LinkedLibraryHolder.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cudaq {

void bindRuntimeTarget(py::module &mod, LinkedLibraryHolder &holder) {

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
          std::string strValue = "";
          if (py::isinstance<py::bool_>(value))
            strValue = value.cast<py::bool_>() ? "true" : "false";
          else if (py::isinstance<py::str>(value))
            strValue = value.cast<std::string>();
          else
            throw std::runtime_error(
                "QPU kwargs config value must be cast-able to a string.");

          config.emplace(key.cast<std::string>(), strValue);
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
          std::string strValue = "";
          if (py::isinstance<py::bool_>(value))
            strValue = value.cast<py::bool_>() ? "true" : "false";
          else if (py::isinstance<py::str>(value))
            strValue = value.cast<std::string>();
          else
            throw std::runtime_error(
                "QPU kwargs config value must be cast-able to a string.");

          config.emplace(key.cast<std::string>(), strValue);
        }
        holder.setTarget(name, config);
      },
      "Set the `cudaq.Target` with given name to be used for CUDA Quantum "
      "kernel execution. Can provide optional, target-specific configuration "
      "data via Python kwargs.");
}

} // namespace cudaq
