/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PasqalRemoteRESTQPU.h"
#include "common/AnalogRydberg.h"
#include "nlohmann/json.hpp"

void cudaq::PasqalRemoteRESTQPU::setTargetBackend(const std::string &backend) {
  AnalogRemoteRESTQPU::setTargetBackend(backend);
  rydbergC6 = ahs::fresnelCan.rydbergC6;
  auto it = backendConfig.find("emulation_rydberg_c6");
  if (!emulate || it == backendConfig.end())
    return;
  std::size_t parsed = 0;
  try {
    rydbergC6 = std::stod(it->second, &parsed);
  } catch (const std::exception &) {
    parsed = 0;
  }
  if (parsed == 0 || parsed != it->second.size())
    throw std::invalid_argument("Invalid `emulation_rydberg_c6` value '" +
                                it->second +
                                "': expected a number in rad m^6 / s.");
}

cudaq::sample_result
cudaq::PasqalRemoteRESTQPU::emulateJob(const std::string &payload,
                                       std::size_t shots, std::size_t seed,
                                       analog::Engine &engine) {
  const ahs::DeviceSpecification device{rydbergC6};
  auto counts = ServerMessage::object();
  for (const auto &[bits, count] :
       ahs::emulate(payload, shots, seed, device, engine))
    counts[bits] = count;
  // Parse a completed Pasqal Cloud job exactly as for remote execution.
  ServerMessage response;
  response["data"]["status"] = "DONE";
  response["data"]["result"] = ServerMessage::array({counts});
  std::string jobId = "emulated";
  return serverHelper->processResults(response, jobId);
}

cudaq::PasqalRemoteRESTQPU::~PasqalRemoteRESTQPU() = default;

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::PasqalRemoteRESTQPU, pasqal)
