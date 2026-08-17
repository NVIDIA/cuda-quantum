/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/AnalogHamiltonian.h"
#include "common/EvolveResult.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/algorithms/observe.h"
#include "cudaq/host_config.h"
#include "cudaq/operators.h"
#include "cudaq/platform.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/schedule.h"
#include <exception>

namespace cudaq {
class base_integrator;

/// @brief Return type for asynchronous `evolve_async`.
using async_evolve_result = std::future<evolve_result>;

namespace detail {
// Internal methods for evolve implementation on circuit simulators.

/// @brief Evolve from an initial state to the final state, no intermediate
/// states.
template <typename QuantumKernel>
evolve_result evolve(state initial_state, QuantumKernel &&kernel,
                     const std::vector<spin_op> &observables = {},
                     int shots_count = -1) {
  state final_state =
      get_state(std::forward<QuantumKernel>(kernel), initial_state);
  if (observables.size() == 0)
    return evolve_result(final_state);

  with_platform_in_library_mode libraryMode(cudaq::get_platform());
  auto prepare_state = [final_state]() { auto qs = qvector<2>(final_state); };
  std::vector<observe_result> final_expectations;
  for (auto observable : observables) {
    shots_count <= 0
        ? final_expectations.push_back(observe(prepare_state, observable))
        : final_expectations.push_back(
              observe(shots_count, prepare_state, observable));
  }
  return evolve_result(final_state, final_expectations);
}

/// @brief Evolve from an initial state to the final state and gather
/// intermediate states.
// Step evolution is provided as `kernels`.
template <typename QuantumKernel>
evolve_result evolve(state initial_state, std::vector<QuantumKernel> &kernels,
                     const std::vector<std::vector<spin_op>> &observables = {},
                     int shots_count = -1,
                     bool save_intermediate_states = true) {
  std::vector<state> intermediate_states = {};
  std::vector<std::vector<observe_result>> expectation_values = {};
  int step_idx = -1;
  for (auto kernel : kernels) {
    if (intermediate_states.size() == 0) {
      intermediate_states.push_back(get_state(kernel, initial_state));
    } else {
      auto new_state = get_state(kernel, intermediate_states.back());
      if (save_intermediate_states) {
        intermediate_states.push_back(new_state);
      } else {
        // If we are not saving intermediate results, we just update the last
        // state.
        std::swap(intermediate_states.back(), new_state);
      }
    }
    if (observables.size() > 0) {
      with_platform_in_library_mode libraryMode(cudaq::get_platform());
      std::vector<observe_result> expectations = {};
      auto prepare_state = [intermediate_states]() {
        auto qs = qvector<2>(intermediate_states.back());
      };
      for (auto observable : observables[++step_idx]) {
        shots_count <= 0
            ? expectations.push_back(observe(prepare_state, observable))
            : expectations.push_back(
                  observe(shots_count, prepare_state, observable));
      }
      expectation_values.push_back(expectations);
    }
  }
  if (step_idx < 0)
    return evolve_result(intermediate_states);
  return evolve_result(intermediate_states, expectation_values);
}

template <typename QuantumKernel>
async_evolve_result
evolve_async(state initial_state, QuantumKernel &&kernel,
             const std::vector<spin_op> &observables = {},
             std::size_t qpu_id = 0,
             std::optional<cudaq::noise_model> noise_model = std::nullopt,
             int shots_count = -1) {
  auto &platform = cudaq::get_platform();
  std::promise<evolve_result> promise;
  auto f = promise.get_future();

  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), func = std::forward<QuantumKernel>(kernel),
       initial_state, observables, noise_model, shots_count,
       &platform]() mutable {
        try {
          if (noise_model.has_value())
            platform.set_noise(&noise_model.value());
          with_platform_in_library_mode libraryMode(platform);
          auto result = evolve(initial_state, func, observables, shots_count);
          if (noise_model.has_value())
            platform.set_noise(nullptr);
          p.set_value(std::move(result));
        } catch (...) {
          if (noise_model.has_value())
            platform.set_noise(nullptr);
          p.set_exception(std::current_exception());
        }
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return f;
}

template <typename QuantumKernel>
async_evolve_result
evolve_async(state initial_state, std::vector<QuantumKernel> kernels,
             const std::vector<std::vector<spin_op>> &observables = {},
             std::size_t qpu_id = 0,
             std::optional<cudaq::noise_model> noise_model = std::nullopt,
             int shots_count = -1, bool save_intermediate_states = true) {
  auto &platform = cudaq::get_platform();
  std::promise<evolve_result> promise;
  auto f = promise.get_future();

  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), kernels, initial_state, observables, noise_model,
       shots_count, &platform, save_intermediate_states]() mutable {
        try {
          if (noise_model.has_value())
            platform.set_noise(&noise_model.value());
          with_platform_in_library_mode libraryMode(platform);
          auto result = evolve(initial_state, kernels, observables, shots_count,
                               save_intermediate_states);
          if (noise_model.has_value())
            platform.set_noise(nullptr);
          p.set_value(std::move(result));
        } catch (...) {
          if (noise_model.has_value())
            platform.set_noise(nullptr);
          p.set_exception(std::current_exception());
        }
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return f;
}

inline async_evolve_result
evolve_async(std::function<evolve_result()> evolveFunctor,
             std::size_t qpu_id = 0) {
  auto &platform = cudaq::get_platform();
  if (qpu_id >= platform.num_qpus()) {
    throw std::invalid_argument("Provided qpu_id " + std::to_string(qpu_id) +
                                " is invalid (must be < " +
                                std::to_string(platform.num_qpus()) +
                                " i.e. platform.num_qpus())");
  }
  std::promise<evolve_result> promise;
  auto f = promise.get_future();

  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), evolveFunctor]() mutable {
        try {
          p.set_value(evolveFunctor());
        } catch (...) {
          p.set_exception(std::current_exception());
        }
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return f;
}

// Helper to migrate an input state to the current device if necessary
state migrateState(const state &inputState);

evolve_result evolveSingle(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensions, const schedule &schedule,
    const state &initial_state, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapse_operators = {},
    const std::vector<sum_op<cudaq::matrix_handler>> &observables = {},
    IntermediateResultSave store_intermediate_results =
        IntermediateResultSave::None,
    std::optional<int> shots_count = std::nullopt);

evolve_result evolveSingle(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensions, const schedule &schedule,
    InitialState initial_state, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapse_operators = {},
    const std::vector<sum_op<cudaq::matrix_handler>> &observables = {},
    IntermediateResultSave store_intermediate_results =
        IntermediateResultSave::None,
    std::optional<int> shots_count = std::nullopt);

std::vector<evolve_result> evolveBatched(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensions, const schedule &schedule,
    const std::vector<state> &initial_states, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapse_operators = {},
    const std::vector<sum_op<cudaq::matrix_handler>> &observables = {},
    IntermediateResultSave store_intermediate_results =
        IntermediateResultSave::None,
    std::optional<int> shots_count = std::nullopt);

evolve_result
evolveSingle(const super_op &superOp, const cudaq::dimension_map &dimensionsMap,
             const schedule &schedule, const state &initialState,
             base_integrator &integrator,
             const std::vector<sum_op<cudaq::matrix_handler>> &observables = {},
             IntermediateResultSave store_intermediate_results =
                 IntermediateResultSave::None,
             std::optional<int> shotsCount = std::nullopt);

evolve_result
evolveSingle(const super_op &superOp, const cudaq::dimension_map &dimensionsMap,
             const schedule &schedule, InitialState initialState,
             base_integrator &integrator,
             const std::vector<sum_op<cudaq::matrix_handler>> &observables = {},
             IntermediateResultSave store_intermediate_results =
                 IntermediateResultSave::None,
             std::optional<int> shotsCount = std::nullopt);

std::vector<evolve_result> evolveBatched(
    const super_op &superOp, const cudaq::dimension_map &dimensions,
    const schedule &schedule, const std::vector<state> &initial_states,
    base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &observables = {},
    IntermediateResultSave store_intermediate_results =
        IntermediateResultSave::None,
    std::optional<int> shots_count = std::nullopt);

std::vector<evolve_result> evolveBatched(
    const std::vector<sum_op<cudaq::matrix_handler>> &hamiltonians,
    const cudaq::dimension_map &dimensions, const schedule &schedule,
    const std::vector<state> &initial_states, base_integrator &integrator,
    const std::vector<std::vector<sum_op<cudaq::matrix_handler>>>
        &collapse_operators = {},
    const std::vector<sum_op<cudaq::matrix_handler>> &observables = {},
    IntermediateResultSave store_intermediate_results =
        IntermediateResultSave::None,
    std::optional<int> batch_size = std::nullopt);

std::vector<evolve_result> evolveBatched(
    const std::vector<super_op> &superOps,
    const cudaq::dimension_map &dimensions, const schedule &schedule,
    const std::vector<state> &initial_states, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &observables = {},
    IntermediateResultSave store_intermediate_results =
        IntermediateResultSave::None,
    std::optional<int> batch_size = std::nullopt);

// Internal methods for evolve implementation on circuit simulators.
inline sample_result launchAnalogKernel(const std::string &kernel_name,
                                        const std::string &program,
                                        std::size_t shots_count,
                                        std::size_t qpu_id = 0) {
  if (!cudaq::detail::isAnalogHamiltonianKernel(kernel_name))
    throw std::runtime_error("Unexpected type of kernel.");

  auto &platform = cudaq::get_platform();
  sample_policy policy;
  policy.options.shots = shots_count;
  policy.kernelName = kernel_name;
  ExecutionContext ctx(sample_policy::name, shots_count, qpu_id);
  return detail::launch(policy, qpu_id, ctx, platform, [&]() {
    [[maybe_unused]] auto dynamicResult = cudaq::altLaunchKernel(
        kernel_name.c_str(), KernelThunkType(nullptr),
        const_cast<char *>(program.c_str()), program.size(), 0);
  });
}

inline async_sample_result
launchAnalogKernelAsync(const std::string &kernel_name,
                        const std::string &program, std::size_t shots_count,
                        std::size_t qpu_id = 0) {
  if (!cudaq::detail::isAnalogHamiltonianKernel(kernel_name))
    throw std::runtime_error("Unexpected type of kernel.");

  auto &platform = cudaq::get_platform();
  async_sample_policy policy;
  policy.inner.options.shots = shots_count;
  policy.inner.kernelName = kernel_name;
  ExecutionContext ctx(sample_policy::name, shots_count, qpu_id);
  return detail::launch(policy, qpu_id, ctx, platform, [&]() {
    [[maybe_unused]] auto dynamicResult = cudaq::altLaunchKernel(
        kernel_name.c_str(), KernelThunkType(nullptr),
        const_cast<char *>(program.c_str()), program.size(), 0);
  });
}

inline evolve_result
evolveSingle(const cudaq::rydberg_hamiltonian &hamiltonian,
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
  auto programString = programJson.dump();
  CUDAQ_DBG("Program JSON: {}", programString);

  auto sampleResults = launchAnalogKernel(programName.str(), programString,
                                          shots_count.value_or(100), 0);

  return evolve_result(sampleResults);
}

} // namespace detail
} // namespace cudaq
