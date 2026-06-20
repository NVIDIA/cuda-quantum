/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(MKLQ_HAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

using complexd = std::complex<double>;

namespace {

constexpr int maxOpenMpThreads = 4;

struct Options {
  std::vector<std::string> variants;
  std::vector<std::size_t> qubits;
  int repeats = 3;
  int warmups = 1;
  std::size_t seed = 13;
};

struct Workspace {
  std::vector<double> probabilities;
  std::vector<double> splitReal;
  std::vector<double> splitImag;

  explicit Workspace(std::size_t dimension)
      : probabilities(dimension, 0.0), splitReal(dimension, 0.0),
        splitImag(dimension, 0.0) {}
};

std::vector<std::string> parseCsv(std::string_view value) {
  std::vector<std::string> result;
  std::stringstream stream{std::string(value)};
  std::string item;
  while (std::getline(stream, item, ','))
    if (!item.empty())
      result.push_back(item);
  if (result.empty())
    throw std::runtime_error("expected at least one CSV item");
  return result;
}

std::vector<std::size_t> parseSizeCsv(std::string_view value) {
  std::vector<std::size_t> result;
  for (const auto &item : parseCsv(value)) {
    const auto parsed = std::stoull(item);
    if (parsed < 1 || parsed >= 63)
      throw std::runtime_error("qubit count must be in [1, 62]");
    result.push_back(parsed);
  }
  return result;
}

Options parseArgs(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg{argv[index]};
    auto requireValue = [&](std::string_view name) -> std::string_view {
      if (index + 1 >= argc)
        throw std::runtime_error(std::string(name) + " requires a value");
      return argv[++index];
    };

    if (arg == "--variants") {
      options.variants = parseCsv(requireValue(arg));
    } else if (arg == "--qubits") {
      options.qubits = parseSizeCsv(requireValue(arg));
    } else if (arg == "--repeats") {
      options.repeats = std::stoi(std::string(requireValue(arg)));
    } else if (arg == "--warmups") {
      options.warmups = std::stoi(std::string(requireValue(arg)));
    } else if (arg == "--seed") {
      options.seed = std::stoull(std::string(requireValue(arg)));
    } else {
      throw std::runtime_error("unknown argument: " + std::string(arg));
    }
  }

  if (options.variants.empty())
    throw std::runtime_error("--variants is required");
  if (options.qubits.empty())
    throw std::runtime_error("--qubits is required");
  if (options.repeats < 1 || options.warmups < 0)
    throw std::runtime_error("repeats must be positive and warmups non-negative");
  return options;
}

std::vector<complexd> makeState(std::size_t dimension, std::size_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<complexd> state(dimension);
  for (auto &amplitude : state)
    amplitude = {dist(rng), dist(rng)};
  return state;
}

void scalarNorm(const std::vector<complexd> &state, Workspace &workspace) {
  std::transform(state.begin(), state.end(), workspace.probabilities.begin(),
                 [](const auto &amplitude) { return std::norm(amplitude); });
}

void scalarSplit(const std::vector<complexd> &state, Workspace &workspace) {
  for (std::size_t index = 0; index < state.size(); ++index) {
    const auto real = state[index].real();
    const auto imag = state[index].imag();
    workspace.probabilities[index] = real * real + imag * imag;
  }
}

int openmpThreadCount() {
#if defined(_OPENMP)
  return std::max(1, std::min(omp_get_max_threads(), maxOpenMpThreads));
#else
  return 1;
#endif
}

bool openmpSplit(const std::vector<complexd> &state, Workspace &workspace) {
#if defined(_OPENMP)
  const auto threadCount = openmpThreadCount();
#pragma omp parallel for schedule(static) num_threads(threadCount)
  for (std::size_t index = 0; index < state.size(); ++index) {
    const auto real = state[index].real();
    const auto imag = state[index].imag();
    workspace.probabilities[index] = real * real + imag * imag;
  }
  return true;
#else
  (void)state;
  (void)workspace;
  return false;
#endif
}

bool accelerateInterleaved(const std::vector<complexd> &state,
                           Workspace &workspace) {
#if defined(MKLQ_HAS_ACCELERATE)
  const auto *interleaved = reinterpret_cast<const double *>(state.data());
  const auto length = static_cast<vDSP_Length>(state.size());
  vDSP_vsqD(interleaved, 2, workspace.probabilities.data(), 1, length);
  vDSP_vmaD(interleaved + 1, 2, interleaved + 1, 2,
            workspace.probabilities.data(), 1, workspace.probabilities.data(),
            1, length);
  return true;
#else
  (void)state;
  (void)workspace;
  return false;
#endif
}

bool accelerateVdsp(const std::vector<complexd> &state, Workspace &workspace) {
#if defined(MKLQ_HAS_ACCELERATE)
  DSPDoubleSplitComplex split{workspace.splitReal.data(),
                              workspace.splitImag.data()};
  vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex *>(state.data()), 2,
             &split, 1, state.size());
  vDSP_zvmagsD(&split, 1, workspace.probabilities.data(), 1, state.size());
  return true;
#else
  (void)state;
  (void)workspace;
  return false;
#endif
}

bool runVariant(std::string_view variant, const std::vector<complexd> &state,
                Workspace &workspace) {
  if (variant == "scalar-norm") {
    scalarNorm(state, workspace);
    return true;
  }
  if (variant == "scalar-split") {
    scalarSplit(state, workspace);
    return true;
  }
  if (variant == "openmp-split")
    return openmpSplit(state, workspace);
  if (variant == "accelerate-interleaved")
    return accelerateInterleaved(state, workspace);
  if (variant == "accelerate-vdsp")
    return accelerateVdsp(state, workspace);
  throw std::runtime_error("unknown variant: " + std::string(variant));
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if (values.size() % 2)
    return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

double maxAbsDiff(const std::vector<double> &actual,
                  const std::vector<double> &expected) {
  double diff = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index)
    diff = std::max(diff, std::abs(actual[index] - expected[index]));
  return diff;
}

double checksum(const std::vector<double> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0);
}

void emitJsonEscaped(std::ostream &os, std::string_view value) {
  os << '"';
  for (const char ch : value) {
    if (ch == '"' || ch == '\\')
      os << '\\' << ch;
    else
      os << ch;
  }
  os << '"';
}

void emitRow(std::ostream &os, bool &first, std::string_view variant,
             std::size_t qubits, std::size_t dimension, std::string_view status,
             const std::vector<double> &timings, double diff, double sum,
             int openmpThreads, std::string_view error = "") {
  if (!first)
    os << ",\n";
  first = false;

  os << "    {\"variant\": ";
  emitJsonEscaped(os, variant);
  os << ", \"qubits\": " << qubits << ", \"dimension\": " << dimension
     << ", \"status\": ";
  emitJsonEscaped(os, status);
  if (!error.empty()) {
    os << ", \"error\": ";
    emitJsonEscaped(os, error);
  }
  os << ", \"metrics\": {";
  if (!timings.empty()) {
    const auto minmax = std::minmax_element(timings.begin(), timings.end());
    os << "\"elapsed_seconds_min\": " << *minmax.first
       << ", \"elapsed_seconds_median\": " << median(timings)
       << ", \"elapsed_seconds_max\": " << *minmax.second
       << ", \"state_amplitudes_per_second\": " << dimension / median(timings)
       << ", \"max_abs_diff_vs_scalar_norm\": " << diff
       << ", \"probability_checksum\": " << sum
       << ", \"openmp_threads\": " << openmpThreads;
  }
  os << "}}";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parseArgs(argc, argv);
    std::cout << "{\n  \"results\": [\n";
    bool first = true;

    for (const auto qubits : options.qubits) {
      const auto dimension = 1ULL << qubits;
      const auto state = makeState(dimension, options.seed + qubits);
      Workspace referenceWorkspace(dimension);
      scalarNorm(state, referenceWorkspace);
      const auto reference = referenceWorkspace.probabilities;

      for (const auto &variant : options.variants) {
        Workspace workspace(dimension);
        if (!runVariant(variant, state, workspace)) {
          emitRow(std::cout, first, variant, qubits, dimension, "unsupported",
                  {}, 0.0, 0.0, openmpThreadCount(),
                  "variant not available in this build");
          continue;
        }

        for (int warmup = 0; warmup < options.warmups; ++warmup)
          (void)runVariant(variant, state, workspace);

        std::vector<double> timings;
        timings.reserve(options.repeats);
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
          const auto start = std::chrono::steady_clock::now();
          (void)runVariant(variant, state, workspace);
          const auto end = std::chrono::steady_clock::now();
          timings.push_back(std::chrono::duration<double>(end - start).count());
        }

        const auto openmpThreads =
            variant == "openmp-split" ? openmpThreadCount() : 0;
        emitRow(std::cout, first, variant, qubits, dimension, "ok", timings,
                maxAbsDiff(workspace.probabilities, reference),
                checksum(workspace.probabilities), openmpThreads);
      }
    }

    std::cout << "\n  ]\n}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
