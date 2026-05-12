/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Synthesis/AgSynth.h"

#include <optional>
#include <ranges>

using namespace cudaq::synth;

CliffordCircuit cudaq::synth::ag_synth_inverse(const Tableau &tab) {
  Tableau work = tab;
  const size_t n = work.num_qubits();
  CliffordCircuit circ(n);

  for (size_t i = 0; i < n; ++i) {
    // --- Step 1: bring X_i into the x[j] rows ---
    // Find any j >= i such that x[j].get(i) == 1. If found, CX-chain to
    // concentrate that bit onto x[i].
    auto find_x_pivot = [&]() -> std::optional<size_t> {
      for (size_t j = i; j < n; ++j)
        if (work.x()[j].get(i))
          return j;
      return std::nullopt;
    };

    if (auto pivot = find_x_pivot()) {
      size_t index = *pivot;
      // Eliminate all other rows that have x[j].get(i) = 1
      for (size_t j = i + 1; j < n; ++j) {
        if (work.x()[j].get(i) && j != index) {
          work.append_cx(index, j);
          circ.push_back(CliffordGate::cx(index, j));
        }
      }
      // If the pivot row also has z[index].get(i), apply S to clear it
      if (work.z()[index].get(i)) {
        work.append_s(index);
        circ.push_back(CliffordGate::s(index));
      }
      // Apply H to convert the X pivot to a Z pivot
      work.append_h(index);
      circ.push_back(CliffordGate::h(index));
    }

    // --- Step 2: ensure z[i].get(i) == 1 ---
    if (!work.z()[i].get(i)) {
      // Find another row with z[j].get(i) == 1 and CX into row i
      size_t index = 0;
      for (size_t j = 0; j < n; ++j) {
        if (work.z()[j].get(i)) {
          index = j;
          break;
        }
      }
      work.append_cx(i, index);
      circ.push_back(CliffordGate::cx(i, index));
    }

    // --- Step 3: clear z[j].get(i) for all j != i ---
    for (size_t j = 0; j < n; ++j) {
      if (work.z()[j].get(i) && j != i) {
        work.append_cx(j, i);
        circ.push_back(CliffordGate::cx(j, i));
      }
    }

    // --- Step 4: clear x[j].get(i + n) for all j != i ---
    for (size_t j = 0; j < n; ++j) {
      if (work.x()[j].get(i + n) && j != i) {
        work.append_cx(i, j);
        circ.push_back(CliffordGate::cx(i, j));
      }
    }

    // --- Step 5: clear z[j].get(i + n) for all j != i ---
    for (size_t j = 0; j < n; ++j) {
      if (work.z()[j].get(i + n) && j != i) {
        work.append_cx(i, j);
        circ.push_back(CliffordGate::cx(i, j));
        work.append_s(j);
        circ.push_back(CliffordGate::s(j));
        work.append_cx(i, j);
        circ.push_back(CliffordGate::cx(i, j));
      }
    }

    // --- Step 6: clear diagonal z[i].get(i + n) ---
    if (work.z()[i].get(i + n)) {
      work.append_s(i);
      circ.push_back(CliffordGate::s(i));
    }

    // --- Step 7: fix signs ---
    if (work.signs().get(i)) {
      work.append_x(i);
      circ.push_back(CliffordGate::x(i));
    }
    if (work.signs().get(i + n)) {
      work.append_z(i);
      circ.push_back(CliffordGate::z(i));
    }
  }

  return circ;
}

CliffordCircuit cudaq::synth::ag_synth(const Tableau &tab) {
  CliffordCircuit inv = ag_synth_inverse(tab);
  CliffordCircuit fwd(tab.num_qubits());
  // Reverse and invert: most gates are self-inverse; S^{-1} = Z*S.
  for (const CliffordGate &it : std::ranges::reverse_view(inv)) {
    fwd.push_back(it);
    if (it.kind == CliffordGateKind::S)
      fwd.push_back(CliffordGate::z(it.qubit0));
  }
  return fwd;
}
