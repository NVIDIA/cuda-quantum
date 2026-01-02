/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved. *
 * *
 * This source code and the accompanying materials are made available under *
 * the terms of the Apache License 2.0 which accompanies this distribution. *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <gtest/gtest.h>

using namespace cudaq;

TEST(RydbergHamiltonianTest, ConstructorValidInputs) {
  // Valid atom sites
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  // Valid atom filling
  rydberg_hamiltonian hamiltonian(atom_sites, amplitude, phase, delta_global);

  EXPECT_EQ(hamiltonian.get_atom_sites().size(), atom_sites.size());
  EXPECT_EQ(hamiltonian.get_atom_filling().size(), atom_sites.size());
  EXPECT_EQ(hamiltonian.get_amplitude().evaluate({}),
            std::complex<double>(1.0, 0.0));
  EXPECT_EQ(hamiltonian.get_phase().evaluate({}),
            std::complex<double>(0.0, 0.0));
  EXPECT_EQ(hamiltonian.get_delta_global().evaluate({}),
            std::complex<double>(-0.5, 0.0));
}

TEST(RydbergHamiltonianTest, ConstructorWithAtomFilling) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  // Valid atom filling
  std::vector<int> atom_filling = {1, 0, 1};

  rydberg_hamiltonian hamiltonian(atom_sites, amplitude, phase, delta_global,
                                  atom_filling);

  EXPECT_EQ(hamiltonian.get_atom_sites().size(), atom_sites.size());
  EXPECT_EQ(hamiltonian.get_atom_filling(), atom_filling);
}

TEST(RydbergHamiltonianTest, InvalidAtomFillingSize) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  // Invalid atom filling size
  std::vector<int> atom_filling = {1, 0};

  EXPECT_ANY_THROW(rydberg_hamiltonian(atom_sites, amplitude, phase,
                                       delta_global, atom_filling));
}

TEST(RydbergHamiltonianTest, UnsupportedLocalDetuning) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  // Invalid delta_local
  auto delta_local =
      std::make_pair(scalar_operator(0.5), std::vector<double>{0.1, 0.2, 0.3});

  EXPECT_ANY_THROW(rydberg_hamiltonian(atom_sites, amplitude, phase,
                                       delta_global, {}, delta_local));
}

TEST(RydbergHamiltonianTest, Accessors) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  rydberg_hamiltonian hamiltonian(atom_sites, amplitude, phase, delta_global);

  EXPECT_EQ(hamiltonian.get_atom_sites(), atom_sites);
  EXPECT_EQ(hamiltonian.get_amplitude().evaluate({}),
            std::complex<double>(1.0, 0.0));
  EXPECT_EQ(hamiltonian.get_phase().evaluate({}),
            std::complex<double>(0.0, 0.0));
  EXPECT_EQ(hamiltonian.get_delta_global().evaluate({}),
            std::complex<double>(-0.5, 0.0));
}

TEST(RydbergHamiltonianTest, DefaultAtomFilling) {
  std::vector<rydberg_hamiltonian::coordinate> atom_sites = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};

  // Valid operators
  scalar_operator amplitude(1.0);
  scalar_operator phase(0.0);
  scalar_operator delta_global(-0.5);

  rydberg_hamiltonian hamiltonian(atom_sites, amplitude, phase, delta_global);

  std::vector<int> expected_filling(atom_sites.size(), 1);
  EXPECT_EQ(hamiltonian.get_atom_filling(), expected_filling);
}
