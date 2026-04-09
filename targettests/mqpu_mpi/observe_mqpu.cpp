/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: mpi
// clang-format off
// RUN: nvq++ --target qpp-cpu --platform mqpu %s -o %t && mpiexec -np 4 --allow-run-as-root %t 
// clang-format on

#include <cudaq.h>

int main() {
  cudaq::mpi::initialize();

  // Get the quantum_platform singleton
  auto &platform = cudaq::get_platform();

  // Query the number of QPUs in the system
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);

  assert(num_qpus == cudaq::mpi::num_ranks() &&
         "Number of QPUs must match number of MPI ranks for this test.");

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  // A simple linear search (mimicking particle swarm) to solve the
  // optimization. 0.0 to 1.0 with step of 0.01
  const double step = 0.01;
  const std::vector<double> thetas = cudaq::linspace(0.0, 1.0, 101);
  // Split the thetas across the QPUs.
  const auto numThetas = thetas.size();
  const auto thetasPerQpu =
      (numThetas + num_qpus - 1) / num_qpus; // Round up division
  std::vector<std::vector<double>> qpuThetaAssignments(num_qpus);
  for (size_t i = 0; i < numThetas; ++i) {
    const int qpuId = i / thetasPerQpu;
    qpuThetaAssignments[qpuId].push_back(thetas[i]);
  }

  std::vector<std::vector<double>> qpuResults(num_qpus);
  for (int qpu = 0; qpu < num_qpus; qpu++) {
    for (const auto &theta : qpuThetaAssignments[qpu])
      qpuResults[qpu].push_back(
          cudaq::observe_async(qpu, ansatz, h, theta).get());
  }
  for (int qpu = 0; qpu < num_qpus; qpu++)
    cudaq::mpi::broadcast(qpuResults[qpu], qpu);

  // Find the minimum result across all QPUs and the corresponding theta value.
  double globalMin = std::numeric_limits<double>::max();
  double optimalTheta = 0.0;
  for (int qpu = 0; qpu < num_qpus; qpu++) {
    for (size_t i = 0; i < qpuResults[qpu].size(); ++i) {
      if (qpuResults[qpu][i] < globalMin) {
        globalMin = qpuResults[qpu][i];
        optimalTheta = qpuThetaAssignments[qpu][i];
      }
    }
  }
  printf("Minimum energy found: %lf at theta = %lf\n", globalMin, optimalTheta);
  assert(std::abs(globalMin + 1.7487) < 1e-2 &&
         "Minimum energy does not match expected value.");
  assert(std::abs(optimalTheta - 0.59) < 1e-2 &&
         "Optimal theta does not match expected value.");
  cudaq::mpi::finalize();
  return 0;
}
