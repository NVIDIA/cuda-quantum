/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_AnalogHamiltonian.h"
#include "common/AnalogHamiltonian.h"
#include "nlohmann/json.hpp"
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

using json = nlohmann::json;

namespace cudaq {

/// @brief Binds the `cudaq::ahs` classes.
void bindAnalogHamiltonian(nanobind::module_ &mod) {

  nanobind::class_<cudaq::ahs::AtomArrangement>(mod, "AtomArrangement")
      .def(nanobind::init<>())
      .def_rw("sites", &cudaq::ahs::AtomArrangement::sites)
      .def_rw("filling", &cudaq::ahs::AtomArrangement::filling);

  nanobind::class_<cudaq::ahs::Setup>(mod, "SetUp")
      .def(nanobind::init<>())
      .def_rw("ahs_register", &cudaq::ahs::Setup::ahs_register);

  nanobind::class_<cudaq::ahs::TimeSeries>(mod, "TimeSeries")
      .def(nanobind::init<>())
      .def(nanobind::init<std::vector<std::pair<double, double>>>())
      .def_rw("values", &cudaq::ahs::TimeSeries::values)
      .def_rw("times", &cudaq::ahs::TimeSeries::times);

  nanobind::class_<cudaq::ahs::FieldPattern>(mod, "FieldPattern")
      /// NOTE: Other constructors not required from Python interface
      .def(nanobind::init<>())
      .def_rw("patternStr", &cudaq::ahs::FieldPattern::patternStr)
      .def_rw("patternVals", &cudaq::ahs::FieldPattern::patternVals);

  nanobind::class_<cudaq::ahs::PhysicalField>(mod, "PhysicalField")
      .def(nanobind::init<>())
      .def_rw("time_series", &cudaq::ahs::PhysicalField::time_series)
      .def_rw("pattern", &cudaq::ahs::PhysicalField::pattern);

  nanobind::class_<cudaq::ahs::DrivingField>(mod, "DrivingField")
      .def(nanobind::init<>())
      .def_rw("amplitude", &cudaq::ahs::DrivingField::amplitude)
      .def_rw("phase", &cudaq::ahs::DrivingField::phase)
      .def_rw("detuning", &cudaq::ahs::DrivingField::detuning);

  nanobind::class_<cudaq::ahs::LocalDetuning>(mod, "LocalDetuning")
      .def(nanobind::init<>())
      .def_rw("magnitude", &cudaq::ahs::LocalDetuning::magnitude);

  nanobind::class_<cudaq::ahs::Hamiltonian>(mod, "Hamiltonian")
      .def(nanobind::init<>())
      .def_rw("drivingFields", &cudaq::ahs::Hamiltonian::drivingFields)
      .def_rw("localDetuning", &cudaq::ahs::Hamiltonian::localDetuning);

  nanobind::class_<cudaq::ahs::Program>(mod, "Program")
      .def(nanobind::init<>())
      .def_rw("setup", &cudaq::ahs::Program::setup)
      .def_rw("hamiltonian", &cudaq::ahs::Program::hamiltonian)
      .def(
          "to_json",
          [](const cudaq::ahs::Program &p) { return json(p).dump(); },
          "Convert Program to JSON");

  nanobind::class_<cudaq::ahs::ShotMetadata>(mod, "ShotMetadata")
      .def(nanobind::init<>())
      .def_rw("shotStatus", &cudaq::ahs::ShotMetadata::shotStatus);

  nanobind::class_<cudaq::ahs::ShotResult>(mod, "ShotResult")
      .def(nanobind::init<>())
      .def_rw("preSequence", &cudaq::ahs::ShotResult::preSequence)
      .def_rw("postSequence", &cudaq::ahs::ShotResult::postSequence);

  nanobind::class_<cudaq::ahs::ShotMeasurement>(mod, "ShotMeasurement")
      .def(nanobind::init<>())
      .def_rw("shotMetadata", &cudaq::ahs::ShotMeasurement::shotMetadata)
      .def_rw("shotResult", &cudaq::ahs::ShotMeasurement::shotResult);

  /// TODO: Add other classes if needed
}

} // namespace cudaq
