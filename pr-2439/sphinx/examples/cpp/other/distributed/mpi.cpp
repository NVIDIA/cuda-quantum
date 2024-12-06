/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ mpi.cpp -DCUDAQ_ENABLE_MPI_EXAMPLE=1 -o mpi.x && mpiexec -np 4 ./mpi.x
// ```

// This example demonstrates CUDA-Q MPI support.

#ifndef CUDAQ_ENABLE_MPI_EXAMPLE
#define CUDAQ_ENABLE_MPI_EXAMPLE 0
#endif

#include <cudaq.h>

int main(int argc, char **argv) {
#if CUDAQ_ENABLE_MPI_EXAMPLE == 0
  return 0;
#else
  // Initialize MPI
  cudaq::mpi::initialize();

  if (cudaq::mpi::rank() == 0)
    printf("Running MPI example with %d processes.\n", cudaq::mpi::num_ranks());

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  // In addition to the built-in `MQPU` platform, users can construct MPI
  // application directly using CUDA-Q MPI support.
  const auto allParams =
      cudaq::random_vector(-M_PI, M_PI, cudaq::mpi::num_ranks());

  // For example, each MPI process can run `cudaq::observe` for a different
  // parameter.
  const double rankParam = allParams[cudaq::mpi::rank()];
  const double rankResult = cudaq::observe(ansatz, h, rankParam);
  printf("[Process %d]: Energy(%lf) = %lf.\n", cudaq::mpi::rank(), rankParam,
         rankResult);
  // Then, using `cudaq::mpi::all_gather` to collect all the results.
  std::vector<double> gatherData(cudaq::mpi::num_ranks());
  cudaq::mpi::all_gather(gatherData, {rankResult});
  if (cudaq::mpi::rank() == 0) {
    printf("Gathered data from all ranks: \n");
    for (const auto &x : gatherData)
      printf("%lf\n", x);
  }

  // Verify that the data has been assembled as expected.
  if (std::abs(gatherData[cudaq::mpi::rank()] - rankResult) > 1e-12)
    return -1;
  cudaq::mpi::finalize();
  return 0;
#endif
}
