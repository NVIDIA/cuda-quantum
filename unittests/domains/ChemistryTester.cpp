/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"

#include "cudaq/algorithm.h"
#include "cudaq/domains/chemistry.h"
#include "cudaq/optimizers.h"

CUDAQ_TEST(GenerateExcitationsTester, checkSimple) {

  auto [singles, doubles] = cudaq::generateExcitations(2, 4);

  EXPECT_EQ(2, singles.size());
  EXPECT_EQ(1, doubles.size());

  EXPECT_EQ(0, singles[0][0]);
  EXPECT_EQ(2, singles[0][1]);
  EXPECT_EQ(1, singles[1][0]);
  EXPECT_EQ(3, singles[1][1]);

  EXPECT_EQ(0, doubles[0][0]);
  EXPECT_EQ(1, doubles[0][1]);
  EXPECT_EQ(2, doubles[0][2]);
  EXPECT_EQ(3, doubles[0][3]);
}

CUDAQ_TEST(H2MoleculeTester, checkHamiltonian) {
  {
    cudaq::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                       {"H", {0., 0., .7474}}};
    auto molecule = cudaq::create_molecule(geometry, "sto-3g", 1, 0);
    molecule.hamiltonian.dump();
    auto groundEnergy =
        molecule.hamiltonian.to_matrix().minimal_eigenvalue().real();
    EXPECT_NEAR(-1.137, groundEnergy, 1e-3);
  }

  {
    cudaq::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                       {"H", {0., 0., .7474}}};
    auto molecule = cudaq::create_molecule(geometry, "6-31g", 1, 0);
    molecule.hamiltonian.dump();
    auto groundEnergy =
        molecule.hamiltonian.to_matrix().minimal_eigenvalue().real();
    EXPECT_NEAR(-1.1516, groundEnergy, 1e-3);
  }
}

CUDAQ_TEST(H2MoleculeTester, checkUCCSD) {
  {
    cudaq::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                       {"H", {0., 0., .7474}}};
    auto molecule = cudaq::create_molecule(geometry, "sto-3g", 1, 0);
    auto ansatz = [&](std::vector<double> thetas) __qpu__ {
      cudaq::qreg q(2 * molecule.n_orbitals);
      x(q[0]);
      x(q[1]);
      cudaq::uccsd(q, thetas, molecule.n_electrons);
    };

    cudaq::optimizers::cobyla optimizer;
    auto res = cudaq::vqe(ansatz, molecule.hamiltonian, optimizer,
                          cudaq::uccsd_num_parameters(molecule.n_electrons,
                                                      2 * molecule.n_orbitals));
    EXPECT_NEAR(-1.137, std::get<0>(res), 1e-3);

    // Get the true ground state eigenvector
    auto matrix = molecule.hamiltonian.to_matrix();
    auto eigenVectors = matrix.eigenvectors();

    // Map it to a cudaq::state
    std::vector<std::complex<double>> expectedData(eigenVectors.rows());
    for (std::size_t i = 0; i < eigenVectors.rows(); i++)
      expectedData[i] = eigenVectors(i, 0);
    auto groundState = cudaq::get_state(ansatz, std::get<1>(res));
    cudaq::state expectedState(std::make_tuple(
        std::vector<std::size_t>{eigenVectors.rows()}, expectedData));

    // Make sure our UCCSD state at the optimal parameters is the ground state
    EXPECT_NEAR(1.0, groundState.overlap(expectedState), 1e-6);
  }
}
