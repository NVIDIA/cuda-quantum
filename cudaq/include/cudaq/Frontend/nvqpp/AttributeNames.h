/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {

/// Name of the annotation attribute attached to CUDA-Q kernels
static constexpr const char kernelAnnotation[] = "quantum";

/// Name of the attribute attached to entry point functions.
static constexpr const char entryPointAttrName[] = "cudaq-entrypoint";

/// Name of the attribute attached to CUDA-Q kernels.
static constexpr const char kernelAttrName[] = "cudaq-kernel";

/// Name of the attribute attached to device call functions.
static constexpr const char deviceCallAttrName[] = "cudaq-devicecall";

/// Name of the annotation attribute attached to unitary generator function for
/// user-defined custom operations
static constexpr const char generatorAnnotation[] =
    "user_custom_quantum_operation";

/// Name of the annotation attribute attached to the declaration of a function
/// that works with quantum types without being a kernel: a function that is
/// implemented outside of CUDA-Q (the backend, another language), or a
/// primitive operation of an execution manager. Attach via `__qpu_intrinsic__`.
/// A function that is not a kernel and is not marked this way cannot take a
/// quantum type.
static constexpr const char intrinsicAnnotation[] = "quantum_intrinsic";

/// Name of the annotation attribute that disables quantum optimizations on a
/// kernel. Attach via `__disable_quantum_optimization__`.
static constexpr const char disableQuantumOptAnnotation[] =
    "disable_quantum_optimization";

/// Name of the annotation attached to atomic quantum region definitions.
static constexpr const char atomicQuantumRegionAnnotation[] =
    "atomic_quantum_region";

} // namespace cudaq
