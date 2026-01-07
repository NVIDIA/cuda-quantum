/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_runtime_target.h"
#include "LinkedLibraryHolder.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "cudaq/platform.h"
#include "cudaq/target_control.h"
#include <functional>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <shared_mutex>

namespace {
using SetTargetCallbackFnTy = std::function<void(const cudaq::RuntimeTarget &)>;
// List of `set_target` callbacks to be executed during target changes.
// We use a string identifier for so that the subscriber can identify itself,
// e.g., to unsubscribe/replace the callback.
static std::unordered_map<std::string, SetTargetCallbackFnTy> g_callbacks;
static std::shared_mutex g_callbackMutex;

void registerSetTargetCallback(
    std::function<void(const cudaq::RuntimeTarget &)> callback,
    const std::string &id) {
  CUDAQ_INFO("Register set_target callback with id '{}'.", id);
  std::unique_lock<std::shared_mutex> lock(g_callbackMutex);
  g_callbacks[id] = callback;
}

void unregisterSetTargetCallback(const std::string &id) {
  CUDAQ_INFO("Unregister set_target callback with id '{}'.", id);
  std::unique_lock<std::shared_mutex> lock(g_callbackMutex);
  g_callbacks.erase(id);
}

void onTargetChange(const cudaq::RuntimeTarget &newTarget) {
  std::shared_lock<std::shared_mutex> lock(g_callbackMutex);
  for (auto &[name, callback] : g_callbacks) {
    CUDAQ_INFO("Execute set target callback named '{}'", name);
    callback(newTarget);
  }
}
} // namespace

namespace cudaq {

std::map<std::string, std::string>
parseTargetKwArgs(const py::kwargs &extraConfig) {
  if (extraConfig.contains("options"))
    throw std::runtime_error("The keyword `options` argument is not supported "
                             "in cudaq.set_target(). Please use the keyword "
                             "`option` in order to set the target options.");
  std::map<std::string, std::string> config;
  for (auto &[key, value] : extraConfig) {
    std::string strValue = "";
    if (py::isinstance<py::bool_>(value))
      strValue = value.cast<py::bool_>() ? "true" : "false";
    else if (py::isinstance<py::str>(value))
      strValue = value.cast<std::string>();
    else if (py::isinstance<py::int_>(value))
      strValue = std::to_string(value.cast<int>());
    else
      throw std::runtime_error(
          "QPU kwargs config value must be cast-able to a string.");

    // Ignore empty parameter values
    if (!strValue.empty())
      config.emplace(key.cast<std::string>(), strValue);
  }
  return config;
}

void bindRuntimeTarget(py::module &mod, LinkedLibraryHolder &holder) {

  py::enum_<simulation_precision>(
      mod, "SimulationPrecision",
      "Enumeration describing the precision of the underyling simulation.")
      .value("fp32", simulation_precision::fp32)
      .value("fp64", simulation_precision::fp64);

  py::class_<cudaq::RuntimeTarget>(
      mod, "Target",
      "The `cudaq.Target` represents the underlying infrastructure that "
      "CUDA-Q kernels will execute on. Instances of `cudaq.Target` describe "
      "what simulator they may leverage, the quantum_platform required for "
      "execution, and a description for the target.")
      .def_readonly("name", &cudaq::RuntimeTarget::name,
                    "The name of the `cudaq.Target`.")
      .def_readonly("simulator", &cudaq::RuntimeTarget::simulatorName,
                    "The name of the simulator this `cudaq.Target` leverages. "
                    "This will be empty for physical QPUs.")
      .def_readonly("platform", &cudaq::RuntimeTarget::platformName,
                    "The name of the quantum_platform implementation this "
                    "`cudaq.Target` leverages.")
      .def_readonly("description", &cudaq::RuntimeTarget::description,
                    "A string describing the features for this `cudaq.Target`.")
      .def(
          "num_qpus",
          [](cudaq::RuntimeTarget &_) { return cudaq::platform_num_qpus(); },
          "Return the number of QPUs available in this `cudaq.Target`.")
      .def(
          "is_remote",
          [](cudaq::RuntimeTarget &_) { return cudaq::is_remote_platform(); },
          "Returns true if the target consists of a remote REST QPU.")
      .def(
          "is_emulated",
          [](cudaq::RuntimeTarget &_) { return cudaq::is_emulated_platform(); },
          "Returns true if the emulation mode for the target has been "
          "activated.")
      .def(
          "is_remote_simulator",
          [](cudaq::RuntimeTarget &_) {
            return cudaq::is_remote_simulator_platform();
          },
          "Returns true if the target consists of a remote REST Simulator QPU.")
      .def("get_precision", &cudaq::RuntimeTarget::get_precision,
           "Return the simulation precision for the current target.")
      .def(
          "__str__",
          [](cudaq::RuntimeTarget &self) {
            std::string targetInfo = fmt::format(
                "Target {}\n\tsimulator={}\n\tplatform={}"
                "\n\tdescription={}\n\tprecision={}\n",
                self.name, self.simulatorName, self.platformName,
                self.description,
                self.get_precision() == simulation_precision::fp32 ? "fp32"
                                                                   : "fp64");
            const std::string argHelperStr = self.get_target_args_help_string();
            if (!argHelperStr.empty()) {
              targetInfo += ("Supported Arguments:\n" + argHelperStr);
            }
            return targetInfo;
          },
          "Persist the information in this `cudaq.Target` to a string.");

  mod.def(
      "has_target",
      [&](const std::string &name) { return holder.hasTarget(name); },
      "Return true if the `cudaq.Target` with the given name exists.");
  mod.def(
      "reset_target",
      [&]() {
        holder.resetTarget();
        onTargetChange(holder.getTarget());
      },
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
        auto config = parseTargetKwArgs(extraConfig);
        holder.setTarget(target.name, config);
        onTargetChange(target);
      },
      "Set the `cudaq.Target` to be used for CUDA-Q kernel execution. "
      "Can provide optional, target-specific configuration data via Python "
      "kwargs.");
  mod.def(
      "set_target",
      [&](const std::string &name, py::kwargs extraConfig) {
        auto config = parseTargetKwArgs(extraConfig);
        holder.setTarget(name, config);
        onTargetChange(holder.getTarget());
      },
      "Set the `cudaq.Target` with given name to be used for CUDA-Q "
      "kernel execution. Can provide optional, target-specific configuration "
      "data via Python kwargs.");
  mod.def(
      "register_set_target_callback",
      [&](std::function<void(const cudaq::RuntimeTarget &)> callback,
          const std::string &id) {
        // If we're in an environment that disallows target changes, do nothing.
        // For example, this can happen when running C++ with an embedded Python
        // plugin.
        if (!cudaq::__internal__::canModifyTarget())
          return;
        registerSetTargetCallback(callback, id);
        // Execute the callback on the current target
        callback(holder.getTarget());
      },
      "Register a callback function to be executed when the runtime target is "
      "changed. The string `id` can be used to identify the callback for "
      "replacement/removal purposes.");
  mod.def(
      "unregister_set_target_callback",
      [](const std::string &id) {
        // If we're in an environment that disallows target changes, do nothing.
        if (!cudaq::__internal__::canModifyTarget())
          return;
        unregisterSetTargetCallback(id);
      },
      "Unregister a callback identified by the input identifier.");

  py::module_::import("atexit").attr("register")(py::cpp_function([]() {
    // Perform cleanup of registered callbacks, which might be Python objects.
    g_callbacks.clear();
  }));
}

} // namespace cudaq
