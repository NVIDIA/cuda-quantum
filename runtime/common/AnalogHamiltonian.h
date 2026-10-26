/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nlohmann/json_fwd.hpp"
#include "cudaq/runtime/logger/cudaq_fmt.h"
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq {
namespace ahs {
using json = nlohmann::json;

/// @brief Convert a double to a JSON string.
inline std::string doubleAsJsonString(double d) {
  std::string s = cudaq_fmt::format("{:.8f}", d);
  return s;
}

/// @brief Convert a vector of strings to a vector of doubles.
inline std::vector<double>
doubleFromStr(const std::vector<std::string> &stringList) {
  std::vector<double> result;
  result.reserve(stringList.size());
  for (const auto &s : stringList) {
    result.push_back(std::stod(s));
  }
  return result;
}

/// @brief Convert a vector of doubles to a vector of strings.
inline std::vector<std::string>
strFromDouble(const std::vector<double> &doubleList) {
  std::vector<std::string> result;
  result.reserve(doubleList.size());
  for (const auto &d : doubleList) {
    result.push_back(doubleAsJsonString(d));
  }
  return result;
}

/// @brief Represents the arrangement of atoms in terms of coordinates and their
/// filling (filled or vacant).
struct AtomArrangement {
  std::vector<std::vector<double>> sites;
  std::vector<int> filling;
};

void to_json(json &j, const AtomArrangement &p);
void from_json(const json &j, AtomArrangement &p);

/// @brief  Represents the setup of neutral atom registers
struct Setup {
  AtomArrangement ahs_register;
};

void to_json(json &j, const Setup &p);
void from_json(const json &j, Setup &p);

/// @brief Represents control signal time series
struct TimeSeries {
  TimeSeries() = default;
  TimeSeries(const std::vector<std::pair<double, double>> &data) {
    for (const auto &pair : data) {
      values.push_back(pair.first);
      times.push_back(pair.second);
    }
  }
  std::vector<double> values;
  std::vector<double> times;

  bool almostEqual(const TimeSeries &other, double tol = 1e-12) const {
    if (values.size() != other.values.size() ||
        times.size() != other.times.size()) {
      return false;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (std::abs(values[i] - other.values[i]) > tol) {
        return false;
      }
    }
    for (std::size_t i = 0; i < times.size(); ++i) {
      if (std::abs(times[i] - other.times[i]) > tol) {
        return false;
      }
    }
    return true;
  }
};

void to_json(json &j, const TimeSeries &p);
void from_json(const json &j, TimeSeries &p);

/// @brief Represents the pattern of a control field.
// This can be a pattern name, e.g., 'uniform', or a vector of scaling
// coefficients (value between 0.0 and 1.0), one value for each atom site.
struct FieldPattern {
  FieldPattern() : patternStr("uniform") {}
  FieldPattern(const std::string &patternName) : patternStr(patternName) {}
  FieldPattern(const std::vector<double> &patternValues)
      : patternVals(patternValues) {}

  std::string patternStr;
  std::vector<double> patternVals;
  bool operator==(const FieldPattern &other) const {
    return patternStr == other.patternStr && patternVals == other.patternVals;
  }
};

void to_json(json &j, const FieldPattern &p);
void from_json(const json &j, FieldPattern &p);

/// @brief Represents the temporal and spatial dependence of a control parameter
/// affecting the atoms
struct PhysicalField {
  TimeSeries time_series;
  FieldPattern pattern;
};

void to_json(json &j, const PhysicalField &p);
void from_json(const json &j, PhysicalField &p);

/// @brief Represents the global driving field of neutral atom system
struct DrivingField {
  // Omega field
  PhysicalField amplitude;
  // Phi field
  PhysicalField phase;
  // Delta field
  PhysicalField detuning;
};

void to_json(json &j, const DrivingField &p);
void from_json(const json &j, DrivingField &p);

/// @brief Represents the local `detuning`
struct LocalDetuning {
  PhysicalField magnitude;
};

void to_json(json &j, const LocalDetuning &p);
void from_json(const json &j, LocalDetuning &p);

/// @brief Represents the neutral atom Hamiltonian (driven parts)
struct Hamiltonian {
  std::vector<DrivingField> drivingFields;
  std::vector<LocalDetuning> localDetuning = {};
};

void to_json(json &j, const Hamiltonian &p);
void from_json(const json &j, Hamiltonian &p);

/// @brief Represents an Analog Hamiltonian Simulation program
struct Program {
  Setup setup;
  Hamiltonian hamiltonian;
};

void to_json(json &j, const Program &p);
void from_json(const json &j, Program &p);

/// @brief Serialize an Analog Hamiltonian Simulation program to a JSON string.
std::string toJsonString(const Program &program);

///////////////////////////////////////////////////////////////////////////////
// The following classes represent the result of Analog Hamiltonian Simulation
// program for the QuEra backend

/// @brief Represents the metadata of the shot
struct ShotMetadata {
  std::string shotStatus;
};

void to_json(json &j, const ShotMetadata &p);
void from_json(const json &j, ShotMetadata &p);

/// @brief Represents the results of a single shot
struct ShotResult {
  std::optional<std::vector<int>> preSequence;
  std::optional<std::vector<int>> postSequence;
};

void to_json(json &j, const ShotResult &p);
void from_json(const json &j, ShotResult &p);

/// @brief Represents the measurement results of a single shot
struct ShotMeasurement {
  ShotMetadata shotMetadata;
  ShotResult shotResult;
};

void to_json(json &j, const ShotMeasurement &p);
void from_json(const json &j, ShotMeasurement &p);

/// @brief Represents the metadata of a single generic task (not tied to AHS
/// program)
struct TaskMetadata {
  std::string id;
  int shots;
  std::string deviceId;
  std::optional<std::string> deviceParameters;
  std::optional<std::string> createdAt;
  std::optional<std::string> endedAt;
  std::optional<std::string> status;
  std::optional<std::string> failureReason;
};

void to_json(json &j, const TaskMetadata &p);
void from_json(const json &j, TaskMetadata &p);

/// @brief Represents the metadata of QuEra-specific task
struct QueraMetadata {
  int numSuccessfulShots;

  QueraMetadata() = default;
  QueraMetadata(int n) {
    if (n < 0 || n > 1000) {
      throw std::out_of_range("Shots must be between 0 and 1000");
    }
    numSuccessfulShots = n;
  }
};

void to_json(json &j, const QueraMetadata &p);
void from_json(const json &j, QueraMetadata &p);

/// @brief Represents the additional metadata about a task, instead of the
/// generalized form, this class specializes to QuEra and AHS program.
struct AdditionalMetadata {
  Program action;
  QueraMetadata queraMetadata;
};

void to_json(json &j, const AdditionalMetadata &p);
void from_json(const json &j, AdditionalMetadata &p);

/// @brief Represents the task result of Analog Hamiltonian Simulation
struct TaskResult {
  TaskMetadata taskMetadata;
  std::optional<std::vector<ShotMeasurement>> measurements;
  std::optional<AdditionalMetadata> additionalMetadata;
};

void to_json(json &j, const TaskResult &p);
void from_json(const json &j, TaskResult &p);

} // namespace ahs

} // namespace cudaq
