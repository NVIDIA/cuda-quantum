/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2026 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "QuantumComputationParameters.h"
#include "cudaq/runtime/logger/logger.h"

using json = nlohmann::json;
using namespace cudaq::qio;

QuantumComputationParameters::QuantumComputationParameters(
    std::size_t shots, json options)
    : m_shots(shots), m_options(options) {}

json QuantumComputationParameters::toJson() const {
  return {{"shots", m_shots}, {"options", m_options}};
}

QuantumComputationParameters
QuantumComputationParameters::fromJson(json j) {
  CUDAQ_INFO("from json");

  auto s = j.value("shots", 0);

  CUDAQ_INFO("from json hey {}", s);

  auto o = j.value("options", {});

  CUDAQ_INFO("from json 2");

  auto k = QuantumComputationParameters(s,o);

  CUDAQ_INFO("from json 3");

  return k
}

nlohmann::json QuantumComputationParameters::options() {
  return m_options;
}