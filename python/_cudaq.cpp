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
#include "runtime/cudaq/spin/py_spin_op.h"
#include "utils/LinkedLibraryHolder.h"

#include <pybind11/stl.h>

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
          if (key == "qpu")
            holder.setQPU(value);
          else if (key == "platform")
            holder.setPlatform(value);
        }
      },
      "This function is meant to be called when the cudaq module is loaded and "
      "provides a mechanism for the programmer to change the backend simulator "
      "/ qpu and platform via the command line.");
  mod.def(
      "list_qpus", [&]() { return holder.list_qpus(); },
      "Lists all available backends. "
      "Use set_qpu to execute code on one of these backends.");
  mod.def(
      "set_qpu",
      [&](const std::string &name, py::kwargs extraConfig) {
        std::map<std::string, std::string> config;
        for (auto &[key, value] : extraConfig) {
          if (!py::isinstance<py::str>(value))
            throw std::runtime_error(
                "QPU kwargs config value must be a string.");

          config.emplace(key.cast<std::string>(), value.cast<std::string>());
        }
        holder.setQPU(name, config);
      },
      "Specifies which backend quantum kernels will be executed on. "
      "Possible values can be queried using the list_qpus. "
      "You may also specify the name of an external simulation plugin. "
      "Can specify str:str key value pairs as kwargs to configure the "
      "backend.");
  mod.def(
      "set_platform",
      [&](const std::string &platformName, py::kwargs extraConfig) {
        std::map<std::string, std::string> config;
        for (auto &[key, value] : extraConfig) {
          if (!py::isinstance<py::str>(value))
            throw std::runtime_error(
                "Platform kwargs config value must be a string.");

          config.emplace(key.cast<std::string>(), value.cast<std::string>());
        }
        holder.setPlatform(platformName, config);
      },
      "Set the quantum_platform to use. Can specify str:str key value "
      "pair as kwargs to configure the platform.");
  mod.def(
      "has_qpu", [](const std::string &name) { return holder.hasQPU(name); },
      "Return true if there is a backend simulator with the given name.");

  cudaq::bindBuilder(mod);
  cudaq::bindQuakeValue(mod);
  cudaq::bindObserve(mod);
  cudaq::bindObserveResult(mod);
  cudaq::bindNoiseModel(mod);
  cudaq::bindSample(mod);
  cudaq::bindMeasureCounts(mod);
  cudaq::bindSpinWrapper(mod);
  cudaq::bindOptimizerWrapper(mod);
  cudaq::bindVQE(mod);
  cudaq::bindPyState(mod);
}
