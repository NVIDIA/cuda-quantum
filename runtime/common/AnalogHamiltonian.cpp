/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "AnalogHamiltonian.h"
#include "nlohmann/json.hpp"

namespace cudaq {
namespace ahs {

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
// Same as above, but for required fields: throws if the field is missing.
#define FROM_JSON_AT_HELPER(field) j.at(#field).get_to(p.field)

void to_json(json &j, const AtomArrangement &p) {
  TO_JSON_HELPER(filling);
  // Note: the schema expects floating point numbers as strings
  std::vector<std::vector<std::string>> floatAsStrings;
  for (const auto &site : p.sites)
    floatAsStrings.push_back(strFromDouble(site));
  j["sites"] = floatAsStrings;
}

void from_json(const json &j, AtomArrangement &p) {
  FROM_JSON_HELPER(filling);
  std::vector<std::vector<std::string>> floatAsStrings;
  j["sites"].get_to(floatAsStrings);
  for (const auto &row : floatAsStrings)
    p.sites.push_back(doubleFromStr(row));
}

void to_json(json &j, const Setup &p) { TO_JSON_HELPER(ahs_register); }

void from_json(const json &j, Setup &p) { FROM_JSON_AT_HELPER(ahs_register); }

void to_json(json &j, const TimeSeries &p) {
  j["values"] = strFromDouble(p.values);
  j["times"] = strFromDouble(p.times);
}

void from_json(const json &j, TimeSeries &p) {
  std::vector<std::string> floatAsStrings;
  j["values"].get_to(floatAsStrings);
  p.values = doubleFromStr(floatAsStrings);
  floatAsStrings.clear();
  j["times"].get_to(floatAsStrings);
  p.times = doubleFromStr(floatAsStrings);
}

void to_json(json &j, const FieldPattern &p) {
  if (p.patternStr.empty())
    j = strFromDouble(p.patternVals);
  else
    j = p.patternStr;
}

void from_json(const json &j, FieldPattern &p) {
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

void to_json(json &j, const PhysicalField &p) {
  TO_JSON_HELPER(time_series);
  TO_JSON_HELPER(pattern);
}

void from_json(const json &j, PhysicalField &p) {
  FROM_JSON_AT_HELPER(time_series);
  FROM_JSON_AT_HELPER(pattern);
}

void to_json(json &j, const DrivingField &p) {
  TO_JSON_HELPER(amplitude);
  TO_JSON_HELPER(phase);
  TO_JSON_HELPER(detuning);
}

void from_json(const json &j, DrivingField &p) {
  FROM_JSON_AT_HELPER(amplitude);
  FROM_JSON_AT_HELPER(phase);
  FROM_JSON_AT_HELPER(detuning);
}

void to_json(json &j, const LocalDetuning &p) { TO_JSON_HELPER(magnitude); }

void from_json(const json &j, LocalDetuning &p) {
  FROM_JSON_AT_HELPER(magnitude);
}

void to_json(json &j, const Hamiltonian &p) {
  TO_JSON_HELPER(drivingFields);
  TO_JSON_HELPER(localDetuning);
}

void from_json(const json &j, Hamiltonian &p) {
  FROM_JSON_AT_HELPER(drivingFields);
  FROM_JSON_AT_HELPER(localDetuning);
}

void to_json(json &j, const Program &p) {
  TO_JSON_HELPER(setup);
  TO_JSON_HELPER(hamiltonian);
}

void from_json(const json &j, Program &p) {
  FROM_JSON_AT_HELPER(setup);
  FROM_JSON_AT_HELPER(hamiltonian);
}

std::string toJsonString(const Program &program) {
  return json(program).dump();
}

void to_json(json &j, const ShotMetadata &p) { TO_JSON_HELPER(shotStatus); }

void from_json(const json &j, ShotMetadata &p) { FROM_JSON_HELPER(shotStatus); }

void to_json(json &j, const ShotResult &p) {
  TO_JSON_OPT_HELPER(preSequence);
  TO_JSON_OPT_HELPER(postSequence);
}

void from_json(const json &j, ShotResult &p) {
  FROM_JSON_OPT_HELPER(preSequence);
  FROM_JSON_OPT_HELPER(postSequence);
}

void to_json(json &j, const ShotMeasurement &p) {
  TO_JSON_HELPER(shotMetadata);
  TO_JSON_HELPER(shotResult);
}

void from_json(const json &j, ShotMeasurement &p) {
  FROM_JSON_AT_HELPER(shotMetadata);
  FROM_JSON_AT_HELPER(shotResult);
}

void to_json(json &j, const TaskMetadata &p) {
  TO_JSON_HELPER(id);
  TO_JSON_HELPER(shots);
  TO_JSON_HELPER(deviceId);
  TO_JSON_OPT_HELPER(deviceParameters);
  TO_JSON_OPT_HELPER(createdAt);
  TO_JSON_OPT_HELPER(endedAt);
  TO_JSON_OPT_HELPER(status);
  TO_JSON_OPT_HELPER(failureReason);
}

void from_json(const json &j, TaskMetadata &p) {
  FROM_JSON_HELPER(id);
  FROM_JSON_HELPER(shots);
  FROM_JSON_HELPER(deviceId);
  FROM_JSON_OPT_HELPER(deviceParameters);
  FROM_JSON_OPT_HELPER(createdAt);
  FROM_JSON_OPT_HELPER(endedAt);
  FROM_JSON_OPT_HELPER(status);
  FROM_JSON_OPT_HELPER(failureReason);
}

void to_json(json &j, const QueraMetadata &p) {
  TO_JSON_HELPER(numSuccessfulShots);
}

void from_json(const json &j, QueraMetadata &p) {
  FROM_JSON_HELPER(numSuccessfulShots);
}

void to_json(json &j, const AdditionalMetadata &p) {
  TO_JSON_HELPER(action);
  TO_JSON_HELPER(queraMetadata);
}

void from_json(const json &j, AdditionalMetadata &p) {
  FROM_JSON_AT_HELPER(action);
  FROM_JSON_AT_HELPER(queraMetadata);
}

void to_json(json &j, const TaskResult &p) {
  TO_JSON_HELPER(taskMetadata);
  TO_JSON_OPT_HELPER(measurements);
  TO_JSON_OPT_HELPER(additionalMetadata);
}

void from_json(const json &j, TaskResult &p) {
  FROM_JSON_HELPER(taskMetadata);
  FROM_JSON_OPT_HELPER(measurements);
  FROM_JSON_OPT_HELPER(additionalMetadata);
}

} // namespace ahs

} // namespace cudaq
