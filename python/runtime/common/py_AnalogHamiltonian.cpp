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
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/map.h>

namespace py = nanobind;

namespace cudaq {

/// @brief Binds the `cudaq::ahs` classes.
void bindAnalogHamiltonian(py::module_ &mod) {

  py::class_<cudaq::ahs::AtomArrangement>(mod, "AtomArrangement")
      .def(py::init<>())
      .def_rw("sites", &cudaq::ahs::AtomArrangement::sites)
      .def_rw("filling", &cudaq::ahs::AtomArrangement::filling);

  py::class_<cudaq::ahs::Setup>(mod, "SetUp")
      .def(py::init<>())
      .def_rw("ahs_register", &cudaq::ahs::Setup::ahs_register);

  py::class_<cudaq::ahs::TimeSeries>(mod, "TimeSeries")
      .def(py::init<>())
      .def(py::init<std::vector<std::pair<double, double>>>())
      .def_rw("values", &cudaq::ahs::TimeSeries::values)
      .def_rw("times", &cudaq::ahs::TimeSeries::times);

  py::class_<cudaq::ahs::FieldPattern>(mod, "FieldPattern")
      /// NOTE: Other constructors not required from Python interface
      .def(py::init<>())
      .def_rw("patternStr", &cudaq::ahs::FieldPattern::patternStr)
      .def_rw("patternVals", &cudaq::ahs::FieldPattern::patternVals);

  py::class_<cudaq::ahs::PhysicalField>(mod, "PhysicalField")
      .def(py::init<>())
      .def_rw("time_series", &cudaq::ahs::PhysicalField::time_series)
      .def_rw("pattern", &cudaq::ahs::PhysicalField::pattern);

  py::class_<cudaq::ahs::DrivingField>(mod, "DrivingField")
      .def(py::init<>())
      .def_rw("amplitude", &cudaq::ahs::DrivingField::amplitude)
      .def_rw("phase", &cudaq::ahs::DrivingField::phase)
      .def_rw("detuning", &cudaq::ahs::DrivingField::detuning);

  py::class_<cudaq::ahs::LocalDetuning>(mod, "LocalDetuning")
      .def(py::init<>())
      .def_rw("magnitude", &cudaq::ahs::LocalDetuning::magnitude);

  py::class_<cudaq::ahs::Hamiltonian>(mod, "Hamiltonian")
      .def(py::init<>())
      .def_rw("drivingFields", &cudaq::ahs::Hamiltonian::drivingFields)
      .def_rw("localDetuning", &cudaq::ahs::Hamiltonian::localDetuning);

  py::class_<cudaq::ahs::Program>(mod, "Program")
      .def(py::init<>())
      .def_rw("setup", &cudaq::ahs::Program::setup)
      .def_rw("hamiltonian", &cudaq::ahs::Program::hamiltonian)
      .def(
          "to_json",
          [](const cudaq::ahs::Program &p) { return json(p).dump(); },
          "Convert Program to JSON");

  py::class_<cudaq::ahs::ShotMetadata>(mod, "ShotMetadata")
      .def(py::init<>())
      .def_rw("shotStatus", &cudaq::ahs::ShotMetadata::shotStatus);

  py::class_<cudaq::ahs::ShotResult>(mod, "ShotResult")
      .def(py::init<>())
      .def_rw("preSequence", &cudaq::ahs::ShotResult::preSequence)
      .def_rw("postSequence", &cudaq::ahs::ShotResult::postSequence);

  py::class_<cudaq::ahs::ShotMeasurement>(mod, "ShotMeasurement")
      .def(py::init<>())
      .def_rw("shotMetadata", &cudaq::ahs::ShotMeasurement::shotMetadata)
      .def_rw("shotResult", &cudaq::ahs::ShotMeasurement::shotResult);

  /// TODO: Add other classes if needed
}

} // namespace cudaq
