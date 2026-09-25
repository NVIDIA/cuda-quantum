/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogEmulation.h"
#include "common/EvolveResult.h"
#include "common/SimulationState.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/schedule.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>
#include <mutex>

namespace cudaq::analog {
namespace {
// RK4's final stage must use the left limit of a waveform interval. The next
// integrate() call starts at the same time with the new (right-continuous)
// coefficients.
class PiecewiseRungeKutta : public integrators::runge_kutta {
  std::shared_ptr<double> intervalEnd;

public:
  PiecewiseRungeKutta(double step, std::shared_ptr<double> end)
      : integrators::runge_kutta(4, step), intervalEnd(std::move(end)) {}

  void integrate(double targetTime) override {
    *intervalEnd = targetTime;
    integrators::runge_kutta::integrate(targetTime);
  }
};

void requireGpu() {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    throw std::runtime_error(
        "Analog emulation requires a CUDA-capable GPU for the CUDA-Q dynamics "
        "backend.");
}
} // namespace

// Emulations share the device's cuDensityMat context, which is not safe for
// concurrent evolution.
class DynamicsEngine : public Engine {
public:
  std::vector<std::complex<double>>
  evolveFromGround(const Model &model) override {
    requireGpu();
    if (model.dimensions.empty())
      return {1.0};
    static std::mutex evolutionMutex;
    std::lock_guard<std::mutex> lock(evolutionMutex);
    auto intervalEnd = std::make_shared<double>(0.0);
    cudaq::schedule schedule(
        std::vector<std::complex<double>>(model.times.begin(),
                                          model.times.end()),
        {"t"},
        [intervalEnd](const std::string &, const std::complex<double> &time) {
          return std::complex<double>(std::min(
              time.real(),
              std::nextafter(*intervalEnd,
                             -std::numeric_limits<double>::infinity())));
        });
    PiecewiseRungeKutta integrator(model.maxStep, intervalEnd);
    auto result = cudaq::detail::evolveSingle(
        model.hamiltonian, model.dimensions, schedule,
        cudaq::InitialState::ZERO, integrator, {}, {},
        cudaq::IntermediateResultSave::None, std::nullopt);
    const auto &state = result.states.value().back();
    std::vector<std::complex<double>> amplitudes(1UL << state.get_num_qubits());
    state.to_host(amplitudes.data(), amplitudes.size());
    return amplitudes;
  }
};

CUDAQ_REGISTER_TYPE(Engine, DynamicsEngine, dynamics)

} // namespace cudaq::analog
