/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/evolution.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <stdexcept>

namespace cudaq {

// Can be removed
matrix_2 taylor_series_expm(const matrix_2 &op_matrix, int order = 20) {
  matrix_2 result = matrix_2(op_matrix.get_rows(), op_matrix.get_columns());
  matrix_2 op_matrix_n =
      matrix_2(op_matrix.get_rows(), op_matrix.get_columns());

  for (size_t i = 0; i < op_matrix.get_rows(); i++) {
    result[{i, i}] = std::complex<double>(1.0, 0.0);
    op_matrix_n[{i, i}] = std::complex<double>(1.0, 0.0);
  }

  double factorial = 1.0;
  for (int n = 1; n <= order; n++) {
    op_matrix_n *= op_matrix;
    factorial *= n;
    result += std::complex<double>(1.0 / factorial, 0.0) * op_matrix_n;
  }

  return result;
}

matrix_2 compute_step_matrix(
    const operator_sum &hamiltonian, const std::map<int, int> &dimensions,
    const std::map<std::string, std::complex<double>> &parameters, double dt,
    bool use_gpu) {
  matrix_2 op_matrix = hamiltonian.to_matrix(dimensions, parameters);
  op_matrix = dt * std::complex<double>(0, -1) * op_matrix;

  if (use_gpu) {
    // TODO: Implement GPU matrix exponential using CuPy or cuQuantum
    throw std::runtime_error(
        "GPU-based matrix exponentiation not implemented.");
  } else {
    return taylor_series_expm(op_matrix);
  }
}

void add_noise_channel_for_step(
    const std::string &step_kernel_name, cudaq::noise_model &noise_model,
    const std::vector<operator_sum> &collapse_operators,
    const std::map<int, int> &dimensions,
    const std::map<std::string, std::complex<double>> &parameters, double dt) {
  for (const auto &collapse_op : collapse_operators) {
    matrix_2 L = collapse_op.to_matrix(dimensions, parameters);
    matrix_2 G = std::complex<double>(-0.5, 0.0) * (L * L);

    // Kraus operators
    matrix_2 M0 = (dt * G) + matrix_2(L.get_rows(), L.get_columns());
    matrix_2 M1 = std::sqrt(dt) * L;

    try {
      noise_model.add_all_qubit_channel(
          step_kernel_name, kraus_channel({std::move(M0), std::move(M1)}));
    } catch (const std::exception &e) {
      std::cerr << "Error adding noise channel: " << e.what() << std::endl;
      throw;
    }
  }
}

// evolve_result launch_analog_hamiltonian_kernel(const std::string
// &target_name,
//                                    const rydberg_hamiltonian &hamiltonian,
//                                    const Schedule &schedule,
//                                    int shots_count, bool is_async = false) {
//     // Generate the time series
//     std::vector<std::pair<double, double>> amp_ts, ph_ts, dg_ts;

//     auto current_schedule = schedule;
//     current_schedule.reset();

//     while(auto t = current_schedule.current_step()) {
//         std::map<std::string, std::complex<double>> parameters = {{"time",
//         t.value()}};

//         amp_ts.emplace_back(hamiltonian.get_amplitude().evaluate(parameters).real(),
//         t.value().real());
//         ph_ts.emplace_back(hamiltonian.get_phase().evaluate(parameters).real(),
//         t.value().real());
//         dg_ts.emplace_back(hamiltonian.get_delta_global().evaluate(parameters).real(),
//         t.value().real());

//         ++schedule;
//     }

//     // Atom arrangement and physical fields
//     cudaq::ahs::AtomArrangement atoms;

// }
} // namespace cudaq