/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Registry.h"
#include "cudaq/domains/chemistry/molecule.h"

namespace cudaq {

/// @brief MoleculePackageDriver provides an extensible interface for
/// generating molecular Hamiltonians and associated metadata.
class MoleculePackageDriver
    : public registry::RegisteredType<MoleculePackageDriver> {
public:
  /// @brief Return a `molecular_hamiltonian` described by the given
  /// geometry, basis set, multiplicity, and charge. Optionally
  /// restrict the active space.
  virtual molecular_hamiltonian
  createMolecule(const molecular_geometry &geometry, const std::string &basis,
                 int multiplicity, int charge,
                 std::optional<std::size_t> nActiveElectrons = std::nullopt,
                 std::optional<std::size_t> nActiveOrbitals = std::nullopt) = 0;

  /// Virtual destructor needed when deleting an instance of a derived class
  /// via a pointer to the base class.
  virtual ~MoleculePackageDriver(){};
};
} // namespace cudaq
