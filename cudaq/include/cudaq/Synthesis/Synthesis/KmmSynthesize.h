/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Math/Unitary.h"

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Exact KMM synthesis
//===----------------------------------------------------------------------===//

/// Decompose a DOmegaUnitary into a Clifford+T Circuit in Matsumoto-Amano
/// normal form.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, sec. 7.3 step 3, using the
/// exact synthesis algorithm of Kliuchnikov, Maslov, Mosca [10].
///
/// Three-phase algorithm:
///   1. Repeatedly call `reduce_denomexp` to peel a single gate (or a
///      short HT / SHT syllable) off the left, dropping the denominator
///      exponent k by one per iteration.
///   2. At k = 0 the residue is a Clifford; unwind it as X / W / S gates
///      via the (a, b, c, d) Clifford parametrisation.
///   3. Normalise the accumulated gate list with `normalize_gates` to land
///      in canonical Matsumoto-Amano form with minimum T-count.
///
/// Total over its input domain. Every DOmegaUnitary produced by
/// gridsynth_unitary is valid input, and normalize_gates is total over
/// Circuit values.
Circuit kmm_synthesize(DOmegaUnitary unitary);

} // namespace cudaq::synth
