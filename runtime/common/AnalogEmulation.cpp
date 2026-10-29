/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogEmulation.h"
#include "cudaq/utils/cudaq_utils.h"
#include <bit>
#include <dlfcn.h>
#include <filesystem>
#include <random>

CUDAQ_INSTANTIATE_REGISTRY(cudaq::analog::Engine::RegistryType)

std::unique_ptr<cudaq::analog::Engine>
cudaq::analog::loadEngine(const std::string &name) {
  if (!registry::isRegistered<Engine>(name)) {
#if defined(__APPLE__) && defined(__MACH__)
    const std::string suffix = ".dylib";
#else
    const std::string suffix = ".so";
#endif
    // Loading the library runs its static registration.
    const auto library =
        std::filesystem::path(cudaq::getCUDAQLibraryPath()).parent_path() /
        ("libnvqir-" + name + suffix);
    if (std::filesystem::exists(library) &&
        !dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL)) {
      const char *reason = dlerror();
      throw std::runtime_error("Failed to load the analog emulation engine '" +
                               name + "' from " + library.string() + ": " +
                               (reason ? reason : "unknown error"));
    }
  }
  return registry::get<Engine>(name);
}

cudaq::sample_result
cudaq::analog::sampleStateVector(const std::vector<std::complex<double>> &state,
                                 std::size_t shots, std::size_t seed,
                                 const std::vector<int> &filling) {
  const std::size_t numSites =
      filling.empty() ? std::bit_width(state.size()) - 1 : filling.size();
  std::vector<double> probabilities;
  probabilities.reserve(state.size());
  for (auto amplitude : state)
    probabilities.push_back(std::norm(amplitude));
  std::mt19937 generator(seed ? seed : std::random_device{}());
  std::discrete_distribution<std::size_t> distribution(probabilities.begin(),
                                                       probabilities.end());
  CountsDictionary counts;
  for (std::size_t shot = 0; shot < shots; ++shot) {
    const auto basis = distribution(generator);
    std::string bits(numSites, '0');
    std::size_t atom = 0;
    for (std::size_t site = 0; site < numSites; ++site)
      if (filling.empty() || filling[site])
        bits[site] = ((basis >> atom++) & 1) ? '1' : '0';
    counts[bits]++;
  }
  return sample_result(ExecutionResult(counts));
}
