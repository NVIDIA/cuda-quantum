/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cudaq.h>

TEST(MPITester, checkSimple) {
  cudaq::mpi::initialize();
  std::cout << "My rank = " << cudaq::mpi::rank() << "\n";
  {
    std::vector<double> data(10);
    if (cudaq::mpi::rank() == 0) {
      for (auto &x : data)
        x = 123;
    }
    cudaq::mpi::broadcast(data, 0);

    for (const auto &x : data)
      std::cout << "Broadcast: Rank " << cudaq::mpi::rank() << ": " << x
                << "\n";
  }

  {
    std::vector<double> gather_data(cudaq::mpi::num_ranks());
    const std::vector<double> rank_data{1.0 * cudaq::mpi::rank()};
    cudaq::mpi::all_gather(gather_data, rank_data);
    for (const auto &x : gather_data)
      std::cout << "Gather: Rank " << cudaq::mpi::rank() << ": " << x << "\n";
  }

  {
    std::vector<double> sum_data(2);
    const std::vector<double> rank_data{1.0 * cudaq::mpi::rank(),
                                        10.0 * cudaq::mpi::rank()};
    cudaq::mpi::all_reduce(sum_data, rank_data);
    for (const auto &x : sum_data)
      std::cout << "Reduce: Rank " << cudaq::mpi::rank() << ": " << x << "\n";
  }

  cudaq::mpi::finalize();
}