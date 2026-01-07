/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_AnalogHamiltonian.h"
#include "common/AnalogHamiltonian.h"
#include "common/JsonConvert.h"
#include <pybind11/stl.h>

namespace py = pybind11;

namespace cudaq {

/// @brief Binds the `cudaq::ahs` classes.
void bindAnalogHamiltonian(py::module &mod) {

  py::class_<cudaq::ahs::AtomArrangement>(mod, "AtomArrangement")
      .def(py::init<>())
      .def_readwrite("sites", &cudaq::ahs::AtomArrangement::sites)
      .def_readwrite("filling", &cudaq::ahs::AtomArrangement::filling);

  py::class_<cudaq::ahs::Setup>(mod, "SetUp")
      .def(py::init<>())
      .def_readwrite("ahs_register", &cudaq::ahs::Setup::ahs_register);

  py::class_<cudaq::ahs::TimeSeries>(mod, "TimeSeries")
      .def(py::init<>())
      .def(py::init<std::vector<std::pair<double, double>>>())
      .def_readwrite("values", &cudaq::ahs::TimeSeries::values)
      .def_readwrite("times", &cudaq::ahs::TimeSeries::times);

  py::class_<cudaq::ahs::FieldPattern>(mod, "FieldPattern")
      /// NOTE: Other constructors not required from Python interface
      .def(py::init<>())
      .def_readwrite("patternStr", &cudaq::ahs::FieldPattern::patternStr)
      .def_readwrite("patternVals", &cudaq::ahs::FieldPattern::patternVals);

  py::class_<cudaq::ahs::PhysicalField>(mod, "PhysicalField")
      .def(py::init<>())
      .def_readwrite("time_series", &cudaq::ahs::PhysicalField::time_series)
      .def_readwrite("pattern", &cudaq::ahs::PhysicalField::pattern);

  py::class_<cudaq::ahs::DrivingField>(mod, "DrivingField")
      .def(py::init<>())
      .def_readwrite("amplitude", &cudaq::ahs::DrivingField::amplitude)
      .def_readwrite("phase", &cudaq::ahs::DrivingField::phase)
      .def_readwrite("detuning", &cudaq::ahs::DrivingField::detuning);

  py::class_<cudaq::ahs::LocalDetuning>(mod, "LocalDetuning")
      .def(py::init<>())
      .def_readwrite("magnitude", &cudaq::ahs::LocalDetuning::magnitude);

  py::class_<cudaq::ahs::Hamiltonian>(mod, "Hamiltonian")
      .def(py::init<>())
      .def_readwrite("drivingFields", &cudaq::ahs::Hamiltonian::drivingFields)
      .def_readwrite("localDetuning", &cudaq::ahs::Hamiltonian::localDetuning);

  py::class_<cudaq::ahs::Program>(mod, "Program")
      .def(py::init<>())
      .def_readwrite("setup", &cudaq::ahs::Program::setup)
      .def_readwrite("hamiltonian", &cudaq::ahs::Program::hamiltonian)
      .def(
          "to_json",
          [](const cudaq::ahs::Program &p) { return json(p).dump(); },
          "Convert Program to JSON");

  py::class_<cudaq::ahs::ShotMetadata>(mod, "ShotMetadata")
      .def(py::init<>())
      .def_readwrite("shotStatus", &cudaq::ahs::ShotMetadata::shotStatus);

  py::class_<cudaq::ahs::ShotResult>(mod, "ShotResult")
      .def(py::init<>())
      .def_readwrite("preSequence", &cudaq::ahs::ShotResult::preSequence)
      .def_readwrite("postSequence", &cudaq::ahs::ShotResult::postSequence);

  py::class_<cudaq::ahs::ShotMeasurement>(mod, "ShotMeasurement")
      .def(py::init<>())
      .def_readwrite("shotMetadata", &cudaq::ahs::ShotMeasurement::shotMetadata)
      .def_readwrite("shotResult", &cudaq::ahs::ShotMeasurement::shotResult);

  /// TODO: Add other classes if needed
}

} // namespace cudaq
