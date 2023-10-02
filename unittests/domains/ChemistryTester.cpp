/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <random>

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

CUDAQ_TEST(H2MoleculeTester, checkExpPauli) {
  auto kernel = [](double theta) __qpu__ {
    cudaq::qreg q(4);
    x(q[0]);
    x(q[1]);
    exp_pauli(theta, q, "XXXY");
  };

  cudaq::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                     {"H", {0., 0., .7474}}};
  auto molecule = cudaq::create_molecule(geometry, "sto-3g", 1, 0);
  cudaq::observe(kernel, molecule.hamiltonian, 1.1);

  cudaq::optimizers::cobyla optimizer;
  auto [e, opt] = optimizer.optimize(1, [&](std::vector<double> x) -> double {
    double e = cudaq::observe(kernel, molecule.hamiltonian, x[0]);
    printf("E = %lf\n", e);
    return e;
  });

  EXPECT_NEAR(-1.137, e, 1e-3);
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

  {
    // Test dynamic molecular_geometry generation
    const auto gen_random_h2_geometry = []() -> std::vector<cudaq::atom> {
      std::vector<cudaq::atom> geom;
      geom.emplace_back(cudaq::atom{"H", {0.0, 0.0, 0.0}});
      static std::random_device rd;
      static std::uniform_real_distribution<> dist(0.1, 1); // range [0.1, 1)
      geom.emplace_back(cudaq::atom{"H", {0.0, 0.0, dist(rd)}});
      return geom;
    };

    cudaq::molecular_geometry geometry(gen_random_h2_geometry());
    auto molecule = cudaq::create_molecule(geometry, "sto-3g", 1, 0);
    auto ansatz = [&](const std::vector<double> &thetas) __qpu__ {
      cudaq::qreg q(2 * molecule.n_orbitals);
      for (std::size_t qId = 0; qId < molecule.n_orbitals; ++qId) {
        x(q[qId]);
      }
      cudaq::uccsd(q, thetas, molecule.n_electrons);
    };

    cudaq::optimizers::cobyla optimizer;
    auto res = cudaq::vqe(ansatz, molecule.hamiltonian, optimizer,
                          cudaq::uccsd_num_parameters(molecule.n_electrons,
                                                      2 * molecule.n_orbitals));

    // Get the true ground state eigenvalue
    auto matrix = molecule.hamiltonian.to_matrix();
    const auto min_eigenvalue = matrix.minimal_eigenvalue();
    EXPECT_NEAR(min_eigenvalue.real(), std::get<0>(res), 1e-3);
    EXPECT_NEAR(min_eigenvalue.imag(), 0.0, 1e-9);
  }
}

CUDAQ_TEST(H2MoleculeTester, checkHWE) {

  cudaq::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                     {"H", {0., 0., .7474}}};
  auto molecule = cudaq::create_molecule(geometry, "sto-3g", 1, 0);

  auto H = molecule.hamiltonian;
  std::size_t numQubits = H.num_qubits();
  std::size_t numLayers = 4;
  auto numParams = cudaq::num_hwe_parameters(numQubits, numLayers);
  EXPECT_EQ(40, numParams);
  cudaq::optimizers::cobyla optimizer;
  optimizer.max_eval = 1000;

  std::vector<double> optParams;
  {
    auto ansatz = [&](std::vector<double> thetas) __qpu__ {
      cudaq::qvector q(numQubits);
      x(q[0]);
      x(q[1]);
      cudaq::hwe(q, numLayers, thetas);
    };

    std::vector<double> params =
        cudaq::random_vector(-3.0, 3.0, numParams, std::mt19937::default_seed);
    std::vector<double> zeros(numParams);
    auto res2 = cudaq::observe(ansatz, H, zeros);
    std::cout << "HF = " << std::setprecision(16) << res2.exp_val_z() << "\n";
    EXPECT_NEAR(-1.116, res2.exp_val_z(), 1e-3);

    auto res = cudaq::vqe(ansatz, H, optimizer, numParams);
    std::cout << "opt result = " << std::setprecision(16) << std::get<0>(res)
              << "\n";
    EXPECT_NEAR(-1.137, std::get<0>(res), 1e-3);
    optParams = std::get<1>(res);
  }

  {
    auto ansatz = [&](std::vector<double> thetas) __qpu__ {
      cudaq::qvector q(numQubits);
      x(q[0]);
      x(q[1]);
      cudaq::hwe(q, numLayers, thetas, {{0, 1}, {1, 2}, {2, 3}});
    };

    double res = cudaq::observe(ansatz, H, optParams);
    EXPECT_NEAR(-1.137, res, 1e-3);
  }
}