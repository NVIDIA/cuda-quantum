/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/AnalogEmulation.h"
#include "common/AnalogHamiltonian.h"

namespace cudaq::ahs {

/// @brief Physics parameters for ideal Rydberg emulation, independent of a
/// transport. C6 includes `1/hbar` and is measured in rad m^6 / s. Hardware
/// constraints and calibration data are not part of this specification.
struct DeviceSpecification {
  double rydbergC6;
};

/// @brief FRESNEL_CAN1 cloud specification (2026-09-11): Rb-87, 60S.
/// Pulser's level-60 coefficient is 865723.02 rad us^-1 um^6.
inline constexpr DeviceSpecification fresnelCan{865723.02e-30};

/// @brief Rydberg Hamiltonian of an AHS program, following Pulser's convention:
/// |g> = |0>, |r> = |1>, and Omega/2 (cos(phi) X + sin(phi) Y) - Delta n.
/// Only occupied sites are emulated. Amplitude and detuning are linear; phase
/// is constant between time points.
analog::Model makeRydbergModel(const Program &program,
                               const DeviceSpecification &device);

/// @brief Emulate a serialized AHS program. Output strings list every trap site
/// in register order: vacant or ground=0, Rydberg=1.
inline sample_result emulate(const std::string &payload, std::size_t shots,
                             std::size_t seed,
                             const DeviceSpecification &device,
                             analog::Engine &engine) {
  const auto program = fromJsonString(payload);
  return analog::sampleStateVector(
      engine.evolveFromGround(makeRydbergModel(program, device)), shots, seed,
      program.setup.ahs_register.filling);
}

} // namespace cudaq::ahs
