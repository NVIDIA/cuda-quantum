/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Demonstrates gradient computation (parameter-shift rule) where each `QPU`
// group runs a multi-GPU observe() call for its assigned gradient component.
//
// Each observe() is distributed across `ranks_per_qpu` GPUs via the MPI
// sub-communicator, enabling circuits too large for one GPU.
//
// Build and run (4 ranks → 2 QPUs of 2 ranks/GPUs each):
// clang-format off
// ```
//   nvq++ --target tensornet -o observe_gradient_cudaq_mpi observe_gradient_cudaq_mpi.cpp
//   mpirun -n 4 ./observe_gradient_cudaq_mpi
// ```
// clang-format on

// [Begin Documentation]
#include <cmath>
#include <cstdio>
#include <cudaq.h>
#include <vector>

// Dummy ansatz for demonstration: replace with your VQE circuit.
// Ry(theta0) on even qubits, Ry(theta1) on odd qubits — product state,
// no entanglement. With H = sum_i Z(i) this gives an analytically
// verifiable gradient: grad[k] = -n/2 * sin(theta_k).
__qpu__ void ansatz(int n, double theta0, double theta1) {
  cudaq::qvector q(n);
  for (int i = 0; i < n; i++) {
    if (i % 2 == 0)
      ry(theta0, q[i]);
    else
      ry(theta1, q[i]);
  }
}

int main() {
  cudaq::mpi::initialize();

  const int world_rank = cudaq::mpi::rank();
  const int world_size = cudaq::mpi::num_ranks();

  const int ranks_per_qpu = 2;
  if (world_size % ranks_per_qpu != 0) {
    if (world_rank == 0)
      fprintf(stderr, "World size must be a multiple of %d.\n", ranks_per_qpu);
    cudaq::mpi::finalize();
    return 1;
  }
  const int num_qpus = world_size / ranks_per_qpu;

  // Assign each rank to a `QPU` group and split the communicator.
  // `QPU` group g computes the gradient for parameter theta_g.
  //
  // This example uses 2 parameters and 2 QPU groups. To scale to N parameters,
  // launch with N * `ranks_per_qpu` total MPI ranks and provide N initial
  // `params`. Each additional QPU group adds `ranks_per_qpu` ranks and handles
  // one more gradient component in parallel — no other code changes required.
  //
  // ```
  //  MPI_COMM_WORLD
  //  +----------+----------+----------+----------+
  //  |  rank 0  |  rank 1  |  rank 2  |  rank 3  |
  //  +----------+----------+----------+----------+
  //  |        QPU 0        |        QPU 1        |
  //  | grad[0]=(E+-E-)/2   | grad[1]=(E+-E-)/2   |
  //  +---------------------+---------------------+
  // ```
  const int qpu_id = world_rank / ranks_per_qpu;
  void *qpu_comm = cudaq::mpi::split_communicator(qpu_id);

  // Each `QPU` group uses `ranks_per_qpu` GPUs for every cudaq::observe() call.
  cudaq::mpi::set_communicator(qpu_comm);

  // Dummy Hamiltonian (sum of Z on all qubits) for demonstration:
  // replace with your physical Hamiltonian, e.g. generated from `PySCF`.
  const int n_qubits = 40;
  auto H = cudaq::spin::z(0) + cudaq::spin::z(1);
  for (int i = 2; i < n_qubits; i++)
    H = H + cudaq::spin::z(i);
  const std::vector<double> params = {0.5, 0.3}; // theta_0, theta_1
  const double shift = M_PI / 2.0;

  // Each `QPU` group applies +/-shift to its assigned parameter and runs two
  // observe() calls. Both calls use all `ranks_per_qpu` GPUs via the
  // sub-communicator, enabling multi-GPU tensor-network contraction.
  //
  // In a VQE optimization loop, this block executes every iteration:
  // the optimizer supplies updated `params`, each `QPU` group re-evaluates its
  // two shifted energies, and the gathered gradient drives the next step.
  auto p_plus = params, p_minus = params;
  p_plus[qpu_id] += shift;
  p_minus[qpu_id] -= shift;

  const double e_plus =
      cudaq::observe(ansatz, H, n_qubits, p_plus[0], p_plus[1]).expectation();
  const double e_minus =
      cudaq::observe(ansatz, H, n_qubits, p_minus[0], p_minus[1]).expectation();
  const double local_grad = (e_plus - e_minus) / 2.0;

  // Gather local_grad from every world rank. All ranks within a `QPU` group
  // hold the same value (collective result), so sample every `ranks_per_qpu-th`
  // entry to reconstruct the full gradient vector.
  std::vector<double> all_grads(world_size);
  cudaq::mpi::all_gather(all_grads, std::vector<double>{local_grad});

  if (world_rank == 0) {
    printf("Gradient: [");
    for (int i = 0; i < num_qpus; i++)
      printf("%s%.6f", i > 0 ? ", " : "", all_grads[i * ranks_per_qpu]);
    printf("]\n");
  }

  cudaq::mpi::finalize();
  return 0;
}
// [End Documentation]
