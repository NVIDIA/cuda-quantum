/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "nvqir/photonics/PhotonicCircuitSimulator.h"
#include "common/PluginUtils.h"

thread_local nvqir::PhotonicCircuitSimulator *photonic_simulator;
inline static constexpr std::string_view GetPhotonicCircuitSimulatorSymbol =
    "getPhotonicCircuitSimulator";

/// @brief Provide a holder for externally created
/// PhotonicCircuitSimulator pointers (like from Python) that
/// will invoke clone on the simulator when requested, which
/// in turn will create the simulator if there isn't one on the
/// current thread, otherwise it will reuse the existing one
struct ExternallyProvidedPhotonicSimGenerator {
  nvqir::PhotonicCircuitSimulator *simulator;
  ExternallyProvidedPhotonicSimGenerator(nvqir::PhotonicCircuitSimulator *sim)
      : simulator(sim) {}
  auto operator()() { return simulator->clone(); }
};
static std::unique_ptr<ExternallyProvidedPhotonicSimGenerator>
    externPhotonicSimGenerator;

extern "C" {
void __nvqir__setPhotonicCircuitSimulator(
    nvqir::PhotonicCircuitSimulator *sim) {
  photonic_simulator = sim;
  // If we had been given one before, reset the holder
  if (externPhotonicSimGenerator) {
    auto ptr = externPhotonicSimGenerator.release();
    delete ptr;
  }
  externPhotonicSimGenerator =
      std::make_unique<ExternallyProvidedPhotonicSimGenerator>(sim);
  cudaq::info("[runtime] Setting the photonic circuit simulator to {}.",
              sim->name());
}
}

namespace nvqir {

/// @brief Return the single simulation backend pointer, create if not created
/// already.
/// @return
PhotonicCircuitSimulator *getPhotonicCircuitSimulatorInternal() {
  if (photonic_simulator)
    return photonic_simulator;

  if (externPhotonicSimGenerator) {
    photonic_simulator = (*externPhotonicSimGenerator)();
    return photonic_simulator;
  }
  photonic_simulator = cudaq::getUniquePluginInstance<PhotonicCircuitSimulator>(
      GetPhotonicCircuitSimulatorSymbol);
  cudaq::info("Creating the {} backend.", photonic_simulator->name());
  return photonic_simulator;
};

void setPhotonicRandomSeed(std::size_t seed) {
  getPhotonicCircuitSimulatorInternal()->setRandomSeed(seed);
}

} // namespace nvqir
