/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/cudaq_fmt.h"
#include "nlohmann/json.hpp"
#include <optional>
#include <string>
#include <vector>

namespace cudaq {
namespace ahs {
using json = nlohmann::json;

// Macros to help reduce redundant field typing for optional fields
#define TO_JSON_OPT_HELPER(field)                                              \
  do {                                                                         \
    if (p.field)                                                               \
      j[#field] = *p.field;                                                    \
  } while (0)

#define FROM_JSON_OPT_HELPER(field)                                            \
  do {                                                                         \
    if (j.contains(#field))                                                    \
      p.field = j[#field];                                                     \
  } while (0)

// Macros to help reduce redundant field typing for non-optional fields
#define TO_JSON_HELPER(field) j[#field] = p.field
#define FROM_JSON_HELPER(field) j[#field].get_to(p.field)

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
  friend void to_json(json &j, const AtomArrangement &p) {
    TO_JSON_HELPER(filling);
    // Note: the schema expects floating point numbers as strings
    std::vector<std::vector<std::string>> floatAsStrings;
    for (const auto &site : p.sites)
      floatAsStrings.push_back(strFromDouble(site));
    j["sites"] = floatAsStrings;
  }

  friend void from_json(const json &j, AtomArrangement &p) {
    FROM_JSON_HELPER(filling);
    std::vector<std::vector<std::string>> floatAsStrings;
    j["sites"].get_to(floatAsStrings);
    for (const auto &row : floatAsStrings)
      p.sites.push_back(doubleFromStr(row));
  }
};

/// @brief  Represents the setup of neutral atom registers
struct Setup {
  AtomArrangement ahs_register;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Setup, ahs_register);
};

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
  friend void to_json(json &j, const TimeSeries &p) {
    j["values"] = strFromDouble(p.values);
    j["times"] = strFromDouble(p.times);
  }

  friend void from_json(const json &j, TimeSeries &p) {
    std::vector<std::string> floatAsStrings;
    j["values"].get_to(floatAsStrings);
    p.values = doubleFromStr(floatAsStrings);
    floatAsStrings.clear();
    j["times"].get_to(floatAsStrings);
    p.times = doubleFromStr(floatAsStrings);
  }
};

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

  friend void to_json(json &j, const FieldPattern &p) {
    if (p.patternStr.empty())
      j = strFromDouble(p.patternVals);
    else
      j = p.patternStr;
  }

  friend void from_json(const json &j, FieldPattern &p) {
    if (j.is_array()) {
      std::vector<std::string> floatAsStrings;
      j.get_to(floatAsStrings);
      p.patternVals = doubleFromStr(floatAsStrings);
      p.patternStr.clear();
    } else {
      j.get_to(p.patternStr);
      p.patternVals.clear();
    }
  }
};

/// @brief Represents the temporal and spatial dependence of a control parameter
/// affecting the atoms
struct PhysicalField {
  TimeSeries time_series;
  FieldPattern pattern;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(PhysicalField, time_series, pattern);
};

/// @brief Represents the global driving field of neutral atom system
struct DrivingField {
  // Omega field
  PhysicalField amplitude;
  // Phi field
  PhysicalField phase;
  // Delta field
  PhysicalField detuning;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DrivingField, amplitude, phase, detuning);
};

/// @brief Represents the local `detuning`
struct LocalDetuning {
  PhysicalField magnitude;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(LocalDetuning, magnitude);
};

/// @brief Represents the neutral atom Hamiltonian (driven parts)
struct Hamiltonian {
  std::vector<DrivingField> drivingFields;
  std::vector<LocalDetuning> localDetuning = {};
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Hamiltonian, drivingFields, localDetuning);
};

/// @brief Represents an Analog Hamiltonian Simulation program
struct Program {
  Setup setup;
  Hamiltonian hamiltonian;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Program, setup, hamiltonian);
};

///////////////////////////////////////////////////////////////////////////////
// The following classes represent the result of Analog Hamiltonian Simulation
// program for the QuEra backend

/// @brief Represents the metadata of the shot
struct ShotMetadata {
  std::string shotStatus;

  friend void to_json(json &j, const ShotMetadata &p) {
    TO_JSON_HELPER(shotStatus);
  }

  friend void from_json(const json &j, ShotMetadata &p) {
    FROM_JSON_HELPER(shotStatus);
  }
};

/// @brief Represents the results of a single shot
struct ShotResult {
  std::optional<std::vector<int>> preSequence;
  std::optional<std::vector<int>> postSequence;

  friend void to_json(json &j, const ShotResult &p) {
    TO_JSON_OPT_HELPER(preSequence);
    TO_JSON_OPT_HELPER(postSequence);
  }

  friend void from_json(const json &j, ShotResult &p) {
    FROM_JSON_OPT_HELPER(preSequence);
    FROM_JSON_OPT_HELPER(postSequence);
  }
};

/// @brief Represents the measurement results of a single shot
struct ShotMeasurement {
  ShotMetadata shotMetadata;
  ShotResult shotResult;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ShotMeasurement, shotMetadata, shotResult);
};

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

  friend void to_json(json &j, const TaskMetadata &p) {
    TO_JSON_HELPER(id);
    TO_JSON_HELPER(shots);
    TO_JSON_HELPER(deviceId);
    TO_JSON_OPT_HELPER(deviceParameters);
    TO_JSON_OPT_HELPER(createdAt);
    TO_JSON_OPT_HELPER(endedAt);
    TO_JSON_OPT_HELPER(status);
    TO_JSON_OPT_HELPER(failureReason);
  }

  friend void from_json(const json &j, TaskMetadata &p) {
    FROM_JSON_HELPER(id);
    FROM_JSON_HELPER(shots);
    FROM_JSON_HELPER(deviceId);
    FROM_JSON_OPT_HELPER(deviceParameters);
    FROM_JSON_OPT_HELPER(createdAt);
    FROM_JSON_OPT_HELPER(endedAt);
    FROM_JSON_OPT_HELPER(status);
    FROM_JSON_OPT_HELPER(failureReason);
  }
};

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

  friend void to_json(json &j, const QueraMetadata &p) {
    TO_JSON_HELPER(numSuccessfulShots);
  }

  friend void from_json(const json &j, QueraMetadata &p) {
    FROM_JSON_HELPER(numSuccessfulShots);
  }
};

/// @brief Represents the additional metadata about a task, instead of the
/// generalized form, this class specializes to QuEra and AHS program.
struct AdditionalMetadata {
  Program action;
  QueraMetadata queraMetadata;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(AdditionalMetadata, action, queraMetadata);
};

/// @brief Represents the task result of Analog Hamiltonian Simulation
struct TaskResult {
  TaskMetadata taskMetadata;
  std::optional<std::vector<ShotMeasurement>> measurements;
  std::optional<AdditionalMetadata> additionalMetadata;

  friend void to_json(json &j, const TaskResult &p) {
    TO_JSON_HELPER(taskMetadata);
    TO_JSON_OPT_HELPER(measurements);
    TO_JSON_OPT_HELPER(additionalMetadata);
  }

  friend void from_json(const json &j, TaskResult &p) {
    FROM_JSON_HELPER(taskMetadata);
    FROM_JSON_OPT_HELPER(measurements);
    FROM_JSON_OPT_HELPER(additionalMetadata);
  }
};

} // namespace ahs

} // namespace cudaq
