/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogHamiltonian.h"
#include "common/EvolveResult.h"
#include "common/Logger.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <random>
#include <sstream>
#include <string>

namespace cudaq::__internal__ {

evolve_result evolveSingle(const cudaq::rydberg_hamiltonian &hamiltonian,
                           const cudaq::schedule &schedule,
                           std::optional<int> shots_count = std::nullopt) {
  auto amp = hamiltonian.get_amplitude();
  auto ph = hamiltonian.get_phase();
  auto dg = hamiltonian.get_delta_global();
  std::vector<std::pair<double, double>> amp_ts;
  std::vector<std::pair<double, double>> ph_ts;
  std::vector<std::pair<double, double>> dg_ts;
  for (const auto &step : schedule) {
    auto amp_res = amp.evaluate({{"t", step}});
    amp_ts.push_back(std::make_pair(amp_res.real(), step.real()));

    auto ph_res = ph.evaluate({{"t", step}});
    ph_ts.push_back(std::make_pair(ph_res.real(), step.real()));

    auto dg_res = dg.evaluate({{"t", step}});
    dg_ts.push_back(std::make_pair(dg_res.real(), step.real()));
  }

  auto atoms = cudaq::ahs::AtomArrangement();
  for (auto pair : hamiltonian.get_atom_sites())
    atoms.sites.push_back({pair.first, pair.second});
  atoms.filling = hamiltonian.get_atom_filling();

  auto omega = cudaq::ahs::PhysicalField();
  omega.time_series = cudaq::ahs::TimeSeries(amp_ts);

  auto phi = cudaq::ahs::PhysicalField();
  phi.time_series = cudaq::ahs::TimeSeries(ph_ts);

  auto delta = cudaq::ahs::PhysicalField();
  delta.time_series = cudaq::ahs::TimeSeries(dg_ts);

  auto drive = cudaq::ahs::DrivingField();
  drive.amplitude = omega;
  drive.phase = phi;
  drive.detuning = delta;

  auto program = cudaq::ahs::Program();
  program.setup.ahs_register = atoms;
  program.hamiltonian.drivingFields = {drive};
  program.hamiltonian.localDetuning = {};

  std::ostringstream programName;
  programName << "__analog_hamiltonian_kernel__" << []() {
    const char chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const auto length = sizeof(chars) / sizeof(char);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> distribution(0, length - 1);
    std::string result;
    result.reserve(10);
    for (int i = 0; i < 10; ++i)
      result += chars[distribution(generator)];
    return result;
  }();

  auto programJson = nlohmann::json(program);

  auto &platform = cudaq::get_platform();
  auto ctx =
      std::make_unique<ExecutionContext>("sample", shots_count.value_or(100));
  ctx->asyncExec = false;
  platform.set_exec_ctx(ctx.get());

  auto programString = programJson.dump();
  CUDAQ_DBG("Program JSON: {}", programString);

  auto dynamicResult = cudaq::altLaunchKernel(
      programName.str().c_str(), KernelThunkType(nullptr),
      (void *)(const_cast<char *>(programString.c_str())), 0, 0);

  auto sampleResults = ctx->result;
  platform.reset_exec_ctx();

  return evolve_result(sampleResults);
}

} // namespace cudaq::__internal__
