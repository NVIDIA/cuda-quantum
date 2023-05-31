/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/spin_op.h"

namespace cudaq {

/// @brief An atom represents an atom from the
/// periodic table, it has a name and a coordinate in 3D space.
struct atom {
  const std::string name;
  const double coordinates[3];
};

/// @brief The molecular_geometry encapsulates a vector
/// of atoms, each describing their name and location in
/// 3D space.
class molecular_geometry {
private:
  std::vector<atom> atoms;

public:
  molecular_geometry(std::initializer_list<atom> &&args)
      : atoms(args.begin(), args.end()) {}
  molecular_geometry(const std::vector<atom> &args) : atoms(args) {}
  std::size_t size() const { return atoms.size(); }
  auto begin() { return atoms.begin(); }
  auto end() { return atoms.end(); }
  auto begin() const { return atoms.cbegin(); };
  auto end() const { return atoms.cend(); }
  std::string name() const;
};

/// @brief The `one_body_integrals` provide simple holder type
/// for the `h_pq` coefficients for the second quantized molecular Hamiltonian.
class one_body_integrals {
private:
  std::unique_ptr<std::complex<double>> data;

public:
  std::vector<std::size_t> shape;
  one_body_integrals(const std::vector<std::size_t> &shape);
  std::complex<double> &operator()(std::size_t i, std::size_t j);
  void dump();
};

/// @brief The `two_body_integrals` provide simple holder type
/// for the `h_pqrs` coefficients for the second quantized molecular
/// Hamiltonian.
class two_body_integals {
private:
  std::unique_ptr<std::complex<double>> data;

public:
  std::vector<std::size_t> shape;
  two_body_integals(const std::vector<std::size_t> &shape);
  std::complex<double> &operator()(std::size_t p, std::size_t q, std::size_t r,
                                   std::size_t s);
  void dump();
};

/// @brief The `molecular_hamiltonian` type holds all the pertinent
/// data for a molecule created by CUDA Quantum from its geometry and
/// other metadata.
struct molecular_hamiltonian {
  spin_op hamiltonian;
  one_body_integrals one_body;
  two_body_integals two_body;
  std::size_t n_electrons;
  std::size_t n_orbitals;
  double nuclear_repulsion;
  double hf_energy;
  double fci_energy;
};

/// @brief Given a molecular structure and other metadata,
/// construct the Hamiltonian for the molecule as a `cudaq::spin_op`
molecular_hamiltonian create_molecule(const molecular_geometry &geometry,
                                      const std::string &basis,
                                      int multiplicity, int charge,
                                      std::string driver = "pyscf");

/// @brief Given a molecular structure and other metadata,
/// construct the Hamiltonian for the molecule as a `cudaq::spin_op`.
/// Describe the active space via `n_active_electrons` and `n_active_orbitals`.
molecular_hamiltonian
create_molecule(const molecular_geometry &geometry, const std::string &basis,
                int multiplicity, int charge, std::size_t n_active_electrons,
                std::size_t n_active_orbitals, std::string driver = "pyscf");
} // namespace cudaq
