/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <sstream>
#include <stdexcept>

namespace cudaq {
template <typename HandlerTy>
rydberg_hamiltonian<HandlerTy>::rydberg_hamiltonian(
    const std::vector<Coordinate> &atom_sites, const scalar_operator &amplitude,
    const scalar_operator &phase, const scalar_operator &delta_global,
    const std::vector<int> &atom_filling,
    const std::optional<std::pair<scalar_operator, std::vector<double>>>
        &delta_local)
    : atom_sites(atom_sites), amplitude(amplitude), phase(phase),
      delta_global(delta_global), delta_local(delta_local) {
  if (atom_filling.empty()) {
    this->atom_filling = std::vector<int>(atom_sites.size(), 1);
  } else if (atom_sites.size() != atom_filling.size()) {
    throw std::invalid_argument(
        "Size of `atom_sites` and `atom_filling` must be equal.");
  } else {
    this->atom_filling = atom_filling;
  }

  if (delta_local.has_value()) {
    throw std::runtime_error(
        "Local detuning is an experimental feature not yet supported.");
  }
}

template <typename HandlerTy>
const std::vector<typename rydberg_hamiltonian<HandlerTy>::Coordinate> &
rydberg_hamiltonian<HandlerTy>::get_atom_sites() const {
  return atom_sites;
}

template <typename HandlerTy>
const std::vector<int> &
rydberg_hamiltonian<HandlerTy>::get_atom_filling() const {
  return atom_filling;
}

template <typename HandlerTy>
const scalar_operator &rydberg_hamiltonian<HandlerTy>::get_amplitude() const {
  return amplitude;
}

template <typename HandlerTy>
const scalar_operator &rydberg_hamiltonian<HandlerTy>::get_phase() const {
  return phase;
}

template <typename HandlerTy>
const scalar_operator &
rydberg_hamiltonian<HandlerTy>::get_delta_global() const {
  return delta_global;
}
} // namespace cudaq