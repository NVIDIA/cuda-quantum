/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogRydberg.h"
#include <algorithm>
#include <cmath>

namespace cudaq {
namespace {

double interpolate(const cudaq::ahs::TimeSeries &series, double time,
                   bool constant = false) {
  if (series.times.empty())
    return 0.0;
  auto next = std::upper_bound(series.times.begin(), series.times.end(), time);
  if (next == series.times.begin())
    return series.values.front();
  const auto i = next - series.times.begin() - 1;
  if (constant || next == series.times.end())
    return series.values[i];
  return std::lerp(series.values[i], series.values[i + 1],
                   (time - series.times[i]) /
                       (series.times[i + 1] - series.times[i]));
}

std::vector<double> uniqueTimes(const cudaq::ahs::Program &program) {
  std::vector<double> times;
  for (const auto &drive : program.hamiltonian.drivingFields) {
    times.insert(times.end(), drive.amplitude.time_series.times.begin(),
                 drive.amplitude.time_series.times.end());
    times.insert(times.end(), drive.phase.time_series.times.begin(),
                 drive.phase.time_series.times.end());
    times.insert(times.end(), drive.detuning.time_series.times.begin(),
                 drive.detuning.time_series.times.end());
  }
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  return times;
}

double distance(const std::vector<double> &a, const std::vector<double> &b) {
  const auto dx = a[0] - b[0];
  const auto dy = a[1] - b[1];
  return std::sqrt(dx * dx + dy * dy);
}

double siteScale(const cudaq::ahs::PhysicalField &field, std::size_t site) {
  return field.pattern.patternStr == "uniform"
             ? 1.0
             : field.pattern.patternVals[site];
}

cudaq::scalar_operator fieldCoefficient(const cudaq::ahs::PhysicalField &field,
                                        std::size_t site) {
  return cudaq::scalar_operator(
      [series = field.time_series,
       scale = siteScale(field, site)](const cudaq::parameter_map &parameters) {
        return scale * interpolate(series, parameters.at("t").real());
      },
      {{"t", "time"}});
}

cudaq::scalar_operator driveCoefficient(const cudaq::ahs::DrivingField &drive,
                                        std::size_t site,
                                        double (*quadrature)(double)) {
  return cudaq::scalar_operator(
      [amplitude = drive.amplitude.time_series, phase = drive.phase.time_series,
       amplitudeScale = 0.5 * siteScale(drive.amplitude, site),
       phaseScale = siteScale(drive.phase, site),
       quadrature](const cudaq::parameter_map &parameters) {
        const auto time = parameters.at("t").real();
        return amplitudeScale * interpolate(amplitude, time) *
               quadrature(phaseScale * interpolate(phase, time, true));
      },
      {{"t", "time"}});
}

auto rydbergPopulation(std::size_t site) {
  return cudaq::matrix_op::number(site);
}

cudaq::sum_op<cudaq::matrix_handler>
rydbergHamiltonian(const cudaq::ahs::Program &program,
                   const cudaq::ahs::DeviceSpecification &device,
                   const std::vector<std::size_t> &occupied) {
  auto hamiltonian = cudaq::matrix_op::empty();
  for (const auto &drive : program.hamiltonian.drivingFields) {
    for (std::size_t i = 0; i < occupied.size(); ++i) {
      const auto site = occupied[i];
      hamiltonian +=
          driveCoefficient(drive, site, std::cos) * cudaq::spin_op::x(i);
      hamiltonian +=
          driveCoefficient(drive, site, std::sin) * cudaq::spin_op::y(i);
      hamiltonian -=
          fieldCoefficient(drive.detuning, site) * rydbergPopulation(i);
    }
  }

  const auto &sites = program.setup.ahs_register.sites;
  for (std::size_t i = 0; i < occupied.size(); ++i)
    for (std::size_t j = i + 1; j < occupied.size(); ++j)
      hamiltonian +=
          (device.rydbergC6 /
           std::pow(distance(sites[occupied[i]], sites[occupied[j]]), 6)) *
          rydbergPopulation(i) * rydbergPopulation(j);
  return hamiltonian;
}

void validateProgram(const cudaq::ahs::Program &program) {
  if (!program.hamiltonian.localDetuning.empty())
    throw std::invalid_argument(
        "Local detuning is not supported by AHS emulation.");
  const auto &atoms = program.setup.ahs_register;
  if (atoms.sites.size() != atoms.filling.size())
    throw std::invalid_argument(
        "AHS sites and filling must have equal lengths.");
  for (std::size_t i = 0; i < atoms.sites.size(); ++i) {
    if (atoms.sites[i].size() != 2 || !std::isfinite(atoms.sites[i][0]) ||
        !std::isfinite(atoms.sites[i][1]))
      throw std::invalid_argument(
          "AHS sites must contain two finite coordinates.");
    if (atoms.filling[i] != 0 && atoms.filling[i] != 1)
      throw std::invalid_argument("AHS filling must contain only 0 or 1.");
    for (std::size_t j = 0; j < i; ++j)
      if (atoms.filling[i] && atoms.filling[j] &&
          atoms.sites[i] == atoms.sites[j])
        throw std::invalid_argument(
            "Occupied AHS sites must have distinct coordinates.");
  }
  auto validateField = [&atoms](const cudaq::ahs::PhysicalField &field) {
    const auto &series = field.time_series;
    if (series.times.size() != series.values.size())
      throw std::invalid_argument(
          "AHS times and values must have equal lengths.");
    for (std::size_t i = 0; i < series.times.size(); ++i)
      if (!std::isfinite(series.times[i]) || !std::isfinite(series.values[i]) ||
          series.times[i] < 0.0 ||
          (i && series.times[i] <= series.times[i - 1]))
        throw std::invalid_argument("AHS waveforms require finite values and "
                                    "strictly increasing nonnegative times.");
    const auto &pattern = field.pattern;
    if (pattern.patternStr != "uniform" &&
        (!pattern.patternStr.empty() ||
         pattern.patternVals.size() != atoms.sites.size()))
      throw std::invalid_argument(
          "AHS patterns must be uniform or have one scale per trap site.");
    for (auto scale : pattern.patternVals)
      if (!std::isfinite(scale) || scale < 0.0 || scale > 1.0)
        throw std::invalid_argument(
            "AHS spatial scales must be between 0 and 1.");
  };
  for (const auto &drive : program.hamiltonian.drivingFields) {
    validateField(drive.amplitude);
    validateField(drive.phase);
    validateField(drive.detuning);
  }
}

} // namespace

analog::Model ahs::makeRydbergModel(const Program &program,
                                    const DeviceSpecification &device) {
  validateProgram(program);
  if (!std::isfinite(device.rydbergC6))
    throw std::invalid_argument(
        "AHS emulation requires a finite C6 coefficient.");
  dimension_map dimensions;
  const auto &sites = program.setup.ahs_register.sites;
  std::vector<std::size_t> occupied;
  for (std::size_t i = 0; i < sites.size(); ++i)
    if (program.setup.ahs_register.filling[i]) {
      dimensions[occupied.size()] = 2;
      occupied.push_back(i);
    }

  const auto maxAbs = [](const TimeSeries &series) {
    double value = 0.0;
    for (auto v : series.values)
      value = std::max(value, std::abs(v));
    return value;
  };
  double bound = 0.0;
  for (const auto &drive : program.hamiltonian.drivingFields)
    for (auto site : occupied)
      bound +=
          0.5 * siteScale(drive.amplitude, site) *
              maxAbs(drive.amplitude.time_series) +
          siteScale(drive.detuning, site) * maxAbs(drive.detuning.time_series);
  for (std::size_t i = 0; i < occupied.size(); ++i)
    for (std::size_t j = i + 1; j < occupied.size(); ++j)
      bound += std::abs(device.rydbergC6) /
               std::pow(distance(sites[occupied[i]], sites[occupied[j]]), 6);

  if (!std::isfinite(bound))
    throw std::invalid_argument("AHS Hamiltonian norm bound must be finite.");

  // Limit the RK4 step by the Hamiltonian norm as well as the waveform scale.
  // A fixed 1 ns step is unstable for tightly spaced atoms.
  const auto maxStep = bound > 0.0 ? std::min(1e-9, 0.1 / bound) : 1e-9;
  auto times = uniqueTimes(program);
  // Catch mis-scaled inputs (e.g. atom spacing or C6 units) before they turn
  // into an effectively endless integration.
  constexpr double maxSteps = 1e7;
  if (times.size() > 1 && (times.back() - times.front()) / maxStep > maxSteps)
    throw std::invalid_argument(
        "AHS emulation would need more than 1e7 integration steps; check atom "
        "spacing and C6 units.");
  return {rydbergHamiltonian(program, device, occupied), std::move(dimensions),
          std::move(times), maxStep};
}

} // namespace cudaq
