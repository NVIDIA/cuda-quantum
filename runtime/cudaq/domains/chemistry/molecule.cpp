/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "molecule.h"
#include "MoleculePackageDriver.h"
#include "cudaq/utils/cudaq_utils.h"

#include <xtensor/xadapt.hpp>

LLVM_INSTANTIATE_REGISTRY(cudaq::MoleculePackageDriver::RegistryType)

namespace cudaq {

std::string molecular_geometry::name() const {
  std::string ret = "";
  for (auto &a : atoms)
    ret += a.name;

  return ret;
}

one_body_integrals::one_body_integrals(const std::vector<std::size_t> &shape)
    : shape(shape) {
  assert(shape.size() == 2);
  data = std::unique_ptr<std::complex<double>>(
      new std::complex<double>[shape[0] * shape[1]]);
}

std::complex<double> &one_body_integrals::operator()(std::size_t p,
                                                     std::size_t q) {
  return xt::adapt(data.get(), shape[0] * shape[1], xt::no_ownership(),
                   shape)(p, q);
}

void one_body_integrals::dump() {
  // Manual print to avoid xio.hpp which triggers clang 17-18 template
  // ambiguity with svector's rebind_container (LLVM #91504).
  auto arr =
      xt::adapt(data.get(), shape[0] * shape[1], xt::no_ownership(), shape);
  std::cerr << "{";
  for (std::size_t i = 0; i < shape[0]; ++i) {
    std::cerr << (i > 0 ? " {" : "{");
    for (std::size_t j = 0; j < shape[1]; ++j)
      std::cerr << (j > 0 ? ", " : "") << arr(i, j);
    std::cerr << "}" << (i + 1 < shape[0] ? ",\n" : "");
  }
  std::cerr << "}\n";
}

two_body_integals::two_body_integals(const std::vector<std::size_t> &shape)
    : shape(shape) {
  assert(shape.size() == 4);
  data = std::unique_ptr<std::complex<double>>(
      new std::complex<double>[shape[0] * shape[1] * shape[2] * shape[3]]);
}

std::complex<double> &two_body_integals::operator()(std::size_t p,
                                                    std::size_t q,
                                                    std::size_t r,
                                                    std::size_t s) {
  return xt::adapt(data.get(), shape[0] * shape[1] * shape[2] * shape[3],
                   xt::no_ownership(), shape)(p, q, r, s);
}

void two_body_integals::dump() {
  // Manual print to avoid xio.hpp which triggers clang 17-18 template
  // ambiguity with svector's rebind_container (LLVM #91504).
  std::size_t total = shape[0] * shape[1] * shape[2] * shape[3];
  auto arr = xt::adapt(data.get(), total, xt::no_ownership(), shape);
  std::cerr << "{";
  for (std::size_t i = 0; i < shape[0]; ++i) {
    std::cerr << (i > 0 ? " {" : "{");
    for (std::size_t j = 0; j < shape[1]; ++j) {
      std::cerr << (j > 0 ? " {" : "{");
      for (std::size_t k = 0; k < shape[2]; ++k) {
        std::cerr << (k > 0 ? " {" : "{");
        for (std::size_t l = 0; l < shape[3]; ++l)
          std::cerr << (l > 0 ? ", " : "") << arr(i, j, k, l);
        std::cerr << "}" << (k + 1 < shape[2] ? "," : "");
      }
      std::cerr << "}" << (j + 1 < shape[1] ? "," : "");
    }
    std::cerr << "}" << (i + 1 < shape[0] ? ",\n" : "");
  }
  std::cerr << "}\n";
}

molecular_hamiltonian create_molecule(const molecular_geometry &geometry,
                                      const std::string &basis,
                                      int multiplicity, int charge,
                                      std::string driver) {
  auto packageDriver = registry::get<MoleculePackageDriver>("pyscf");
  if (!packageDriver)
    throw std::runtime_error("Invalid molecule package driver (" + driver +
                             ").");
  return packageDriver->createMolecule(geometry, basis, multiplicity, charge);
}

molecular_hamiltonian
create_molecule(const molecular_geometry &geometry, const std::string &basis,
                int multiplicity, int charge, std::size_t n_active_electrons,
                std::size_t n_active_orbitals, std::string driver) {
  auto packageDriver = registry::get<MoleculePackageDriver>("pyscf");
  if (!packageDriver)
    throw std::runtime_error("Invalid molecule package driver (" + driver +
                             ").");
  return packageDriver->createMolecule(geometry, basis, multiplicity, charge,
                                       n_active_electrons, n_active_orbitals);
}
} // namespace cudaq
