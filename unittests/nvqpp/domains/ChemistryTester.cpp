/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <random>

#include "cudaq/algorithm.h"
#include "cudaq/domains/chemistry/molecule.h"
#include "cudaq/domains/chemistry/uccsd.h"
#include "cudaq/optimizers.h"

namespace chemistry_tester {
// nvq++ does not support cudaq::cnot_coupling (see hwe.h), so inline the HWE
// gate sequence here instead of calling cudaq::hwe from __qpu__ kernels.
// noinline: nvq++ cannot inline one __qpu__ function into another.
__attribute__((noinline)) __qpu__ void
hwe_default(cudaq::qview<> qubits, std::size_t numLayers,
            const std::vector<double> &parameters) {
  std::size_t numQubits = qubits.size();
  std::size_t thetaCounter = 0;
  for (std::size_t i = 0; i < numQubits; i++) {
    ry(parameters[thetaCounter], qubits[i]);
    rz(parameters[thetaCounter + 1], qubits[i]);
    thetaCounter += 2;
  }

  for (std::size_t layer = 0; layer < numLayers; layer++) {
    for (std::size_t q = 0; q + 1 < numQubits; q++)
      x<cudaq::ctrl>(qubits[q], qubits[q + 1]);

    for (std::size_t q = 0; q < numQubits; q++) {
      ry(parameters[thetaCounter], qubits[q]);
      rz(parameters[thetaCounter + 1], qubits[q]);
      thetaCounter += 2;
    }
  }
}

__attribute__((noinline)) __qpu__ void
hwe_custom_chain(cudaq::qview<> qubits, std::size_t numLayers,
                 const std::vector<double> &parameters) {
  std::size_t numQubits = qubits.size();
  std::size_t thetaCounter = 0;
  for (std::size_t i = 0; i < numQubits; i++) {
    ry(parameters[thetaCounter], qubits[i]);
    rz(parameters[thetaCounter + 1], qubits[i]);
    thetaCounter += 2;
  }

  for (std::size_t layer = 0; layer < numLayers; layer++) {
    x<cudaq::ctrl>(qubits[0], qubits[1]);
    x<cudaq::ctrl>(qubits[1], qubits[2]);
    x<cudaq::ctrl>(qubits[2], qubits[3]);

    for (std::size_t q = 0; q < numQubits; q++) {
      ry(parameters[thetaCounter], qubits[q]);
      rz(parameters[thetaCounter + 1], qubits[q]);
      thetaCounter += 2;
    }
  }
}
} // namespace chemistry_tester

CUDAQ_TEST(GenerateExcitationsTester, checkSimple) {
  {
    std::size_t numElectrons = 2;
    std::size_t numQubits = 4;

    auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
        cudaq::get_uccsd_excitations(numElectrons, numQubits);
    EXPECT_TRUE(doublesAlpha.empty());
    EXPECT_TRUE(doublesBeta.empty());
    EXPECT_TRUE(singlesAlpha.size() == 1);
    EXPECT_EQ(singlesAlpha[0][0], 0);
    EXPECT_EQ(singlesAlpha[0][1], 2);
    EXPECT_EQ(singlesBeta[0][0], 1);
    EXPECT_EQ(singlesBeta[0][1], 3);
    EXPECT_EQ(doublesMixed[0][0], 0);
    EXPECT_EQ(doublesMixed[0][1], 1);
    EXPECT_EQ(doublesMixed[0][2], 3);
    EXPECT_EQ(doublesMixed[0][3], 2);
    EXPECT_TRUE(singlesBeta.size() == 1);
    EXPECT_TRUE(doublesMixed.size() == 1);
  }
  {
    std::size_t numElectrons = 4;
    std::size_t numQubits = 8;

    auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
        cudaq::get_uccsd_excitations(numElectrons, numQubits);

    cudaq::excitation_list expectedSinglesAlpha{{0, 4}, {0, 6}, {2, 4}, {2, 6}};
    cudaq::excitation_list expectedSinglesBeta{{1, 5}, {1, 7}, {3, 5}, {3, 7}};
    cudaq::excitation_list expectedDoublesMixed{
        {0, 1, 5, 4}, {0, 1, 5, 6}, {0, 1, 7, 4}, {0, 1, 7, 6},
        {0, 3, 5, 4}, {0, 3, 5, 6}, {0, 3, 7, 4}, {0, 3, 7, 6},
        {2, 1, 5, 4}, {2, 1, 5, 6}, {2, 1, 7, 4}, {2, 1, 7, 6},
        {2, 3, 5, 4}, {2, 3, 5, 6}, {2, 3, 7, 4}, {2, 3, 7, 6}};
    cudaq::excitation_list expectedDoublesAlpha{{0, 2, 4, 6}},
        expectedDoublesBeta{{1, 3, 5, 7}};
    EXPECT_TRUE(singlesAlpha.size() == 4);
    EXPECT_EQ(singlesAlpha, expectedSinglesAlpha);
    EXPECT_TRUE(singlesBeta.size() == 4);
    EXPECT_EQ(singlesBeta, expectedSinglesBeta);
    EXPECT_TRUE(doublesMixed.size() == 16);
    EXPECT_EQ(doublesMixed, expectedDoublesMixed);
    EXPECT_TRUE(doublesAlpha.size() == 1);
    EXPECT_EQ(doublesAlpha, expectedDoublesAlpha);
    EXPECT_TRUE(doublesBeta.size() == 1);
    EXPECT_EQ(doublesBeta, expectedDoublesBeta);
  }
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
    cudaq::qvector q(4);
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
  // nvq++ cannot lower cudaq::uccsd's __qpu__ overload (cudaq::excitations is
  // an unhandled type). Use the kernel_builder overload instead.
  {
    cudaq::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                       {"H", {0., 0., .7474}}};
    auto molecule = cudaq::create_molecule(geometry, "sto-3g", 1, 0);
    auto numQubits = 2 * molecule.n_orbitals;
    auto [ansatz, thetas] = cudaq::make_kernel<std::vector<double>>();
    auto q = ansatz.qalloc(numQubits);
    ansatz.x(q[0]);
    ansatz.x(q[1]);
    cudaq::uccsd(ansatz, q, thetas, molecule.n_electrons, numQubits);

    cudaq::optimizers::cobyla optimizer;
    auto nParams = cudaq::uccsd_num_parameters(molecule.n_electrons, numQubits);
    auto res = cudaq::vqe(ansatz, molecule.hamiltonian, optimizer, nParams);
    EXPECT_NEAR(-1.137, std::get<0>(res), 1e-3);

    // Get the true ground state eigenvector
    auto matrix = molecule.hamiltonian.to_matrix();
    auto eigenVectors = matrix.eigenvectors();

    // Map it to a cudaq::state
    std::vector<cudaq::complex> expectedData(eigenVectors.cols());
    for (std::size_t i = 0; i < eigenVectors.cols(); i++)
      expectedData[i] = eigenVectors(0, i);
    auto groundState = cudaq::get_state(ansatz, std::get<1>(res));

    // Make sure our UCCSD state at the optimal parameters is the ground state
    auto expectedState = cudaq::state::from_data(expectedData);
    EXPECT_NEAR(1.0, groundState.overlap(expectedState).real(), 1e-6);
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
    auto numQubits = 2 * molecule.n_orbitals;
    auto [ansatz, thetas] = cudaq::make_kernel<std::vector<double>>();
    auto q = ansatz.qalloc(numQubits);
    for (std::size_t qId = 0; qId < molecule.n_orbitals; ++qId)
      ansatz.x(q[qId]);
    cudaq::uccsd(ansatz, q, thetas, molecule.n_electrons, numQubits);

    cudaq::optimizers::cobyla optimizer;
    auto nParams = cudaq::uccsd_num_parameters(molecule.n_electrons, numQubits);
    auto res = cudaq::vqe(ansatz, molecule.hamiltonian, optimizer, nParams);

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
  // Host locals cannot be captured into __qpu__ lambdas; use macros for the
  // sto-3g H2 sizes referenced inside kernels below.
#define HWE_NUM_QUBITS 4
#define HWE_NUM_LAYERS 4
  EXPECT_EQ(HWE_NUM_QUBITS, H.num_qubits());
  auto numParams = 2 * HWE_NUM_QUBITS * (1 + HWE_NUM_LAYERS);
  EXPECT_EQ(40, numParams);
  cudaq::optimizers::cobyla optimizer;
  optimizer.max_eval = 1000;

  std::vector<double> optParams;
  {
    auto ansatz = [](std::vector<double> thetas) __qpu__ {
      cudaq::qvector q(HWE_NUM_QUBITS);
      x(q[0]);
      x(q[1]);
      chemistry_tester::hwe_default(q, HWE_NUM_LAYERS, thetas);
    };

    std::vector<double> params =
        cudaq::random_vector(-3.0, 3.0, numParams, std::mt19937::default_seed);
    std::vector<double> zeros(numParams);
    auto res2 = cudaq::observe(ansatz, H, zeros);
    std::cout << "HF = " << std::setprecision(16) << res2.expectation() << "\n";
    EXPECT_NEAR(-1.116, res2.expectation(), 1e-3);

    auto res = cudaq::vqe(ansatz, H, optimizer, numParams);
    std::cout << "opt result = " << std::setprecision(16) << std::get<0>(res)
              << "\n";
    EXPECT_NEAR(-1.137, std::get<0>(res), 1e-3);
    optParams = std::get<1>(res);
  }

  {
    auto ansatz = [](std::vector<double> thetas) __qpu__ {
      cudaq::qvector q(HWE_NUM_QUBITS);
      x(q[0]);
      x(q[1]);
      chemistry_tester::hwe_custom_chain(q, HWE_NUM_LAYERS, thetas);
    };

    double res = cudaq::observe(ansatz, H, optParams);
    EXPECT_NEAR(-1.137, res, 1e-3);
  }
#undef HWE_NUM_QUBITS
#undef HWE_NUM_LAYERS
}
