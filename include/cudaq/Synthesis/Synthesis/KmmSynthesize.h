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

/// Decomposes a DOmegaUnitary into a Clifford+T Circuit in Matsumoto-Amano
/// normal form.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §7.3 step 3.
/// Uses the exact synthesis algorithm of [10].
///
/// Algorithm:
/// 1. Repeatedly call reduce_denomexp to peel off one gate (or short syllable)
///    from the left, reducing the denominator exponent k by 1 each iteration.
/// 2. When k = 0, the remaining matrix is a Clifford; decompose it into
///    X/W/S gates via the normal-form parametrization.
/// 3. Normalize the accumulated Circuit via normalize_gates() to produce
///    the canonical Matsumoto-Amano form with minimum T-count.
///
/// The function cannot fail: all inputs from gridsynth_unitary are valid
/// DOmegaUnitaries, and normalize_gates is total over Circuit (no unknown
/// gate characters are possible).
///
Circuit kmm_synthesize(DOmegaUnitary unitary);

} // namespace cudaq::synth
