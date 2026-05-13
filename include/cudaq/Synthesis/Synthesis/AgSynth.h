/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Circuit/CliffordCircuit.h"
#include "cudaq/Synthesis/Circuit/Tableau.h"

namespace cudaq::synth {

/// Synthesize the inverse circuit using the `Aaronson-Gottesman` algorithm:
/// the circuit that maps \p tab to the identity. Produced by Gaussian
/// elimination on a working copy of the tableau.
CliffordCircuit ag_synth_inverse(const Tableau &tab);

/// Synthesize the forward circuit using the `Aaronson-Gottesman` algorithm:
/// the circuit that produces \p tab from the identity. Internally reverses
/// ag_synth_inverse and adjusts S -> S*Z (since S^{-1} = Z*S).
CliffordCircuit ag_synth(const Tableau &tab);

} // namespace cudaq::synth
