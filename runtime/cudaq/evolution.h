/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EvolveResult.h"
#include "cudaq/base_integrator.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include "cudaq/utils/tensor.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cudaq {

evolve_result evolve_single(
    const operator_sum<cudaq::matrix_operator> &hamiltonian,
    const std::map<int, int> &dimensions, const Schedule &schedule,
    const state &initial_state, BaseIntegrator &integrator,
    const std::vector<operator_sum<cudaq::matrix_operator>>
        &collapse_operators = {},
    const std::vector<operator_sum<cudaq::matrix_operator>> &observables = {},
    bool store_intermediate_results = false,
    std::optional<int> shots_count = std::nullopt);
// class Evolution {
// public:
//   /// Computes the Taylor series expansion of the matrix exponential.
//   static matrix_2 taylor_series_expm(const matrix_2 &op_matrix, int order =
//   20);

//   /// Computes the evolution step matrix
//   static matrix_2 compute_step_matrix(
//       const operator_sum &hamiltonian, const std::map<int, int> &dimensions,
//       const std::map<std::string, std::complex<double>> &parameters, double
//       dt, bool use_gpu = false);

//   /// Adds noise channels based on collapse operators.
//   static void add_noise_channel_for_step(
//       const std::string &step_kernel_name, cudaq::noise_model &noise_model,
//       const std::vector<operator_sum> &collapse_operators,
//       const std::map<int, int> &dimensions,
//       const std::map<std::string, std::complex<double>> &parameters, double
//       dt);

//   /// Launches an analog Hamiltonian kernel for quantum simulations.
//   static evolve_result launch_analog_hamiltonian_kernel(
//       const std::string &target_name, const rydberg_hamiltonian &hamiltonian,
//       const Schedule &schedule, int shots_count, bool is_async = false);

//   /// Generates evolution kernels for the simulation.
//   static std::vector<std::string> evolution_kernel(
//       int num_qubits,
//       const std::function<
//           matrix_2(const std::map<std::string, std::complex<double>> &,
//           double)> &compute_step_matrix,
//       const std::vector<double> tlist,
//       const std::vector<std::map<std::string, std::complex<double>>>
//           &schedule_parameters);

//   /// Evolves a single quantum state under a given `hamiltonian`.
//   static evolve_result
//   evolve_single(const operator_sum &hamiltonian,
//                 const std::map<int, int> &dimensions,
//                 const std::shared_ptr<Schedule> &schedule, state
//                 initial_state, const std::vector<operator_sum>
//                 &collapse_operators = {}, const std::vector<operator_sum>
//                 &observables = {}, bool store_intermediate_results = false,
//                 std::shared_ptr<BaseIntegrator<state>> integrator = nullptr,
//                 std::optional<int> shots_count = std::nullopt);

//   /// Evolves a single or multiple quantum states under a given
//   `hamiltonian`.
//   /// Run only for dynamics target else throw error
//   static std::vector<evolve_result>
//   evolve(const operator_sum &hamiltonian, const std::map<int, int>
//   &dimensions,
//          const std::shared_ptr<Schedule> &schedule,
//          const std::vector<state> &initial_states,
//          const std::vector<operator_sum> &collapse_operators = {},
//          const std::vector<operator_sum> &observables = {},
//          bool store_intermediate_results = false,
//          std::shared_ptr<BaseIntegrator<state>> integrator = nullptr,
//          std::optional<int> shots_count = std::nullopt);
// };
} // namespace cudaq