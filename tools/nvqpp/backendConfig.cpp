/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// This file is meant to be used by the nvq++ driver script, the
/// NVQPP_TARGET_BACKEND_CONFIG string must be replaced (e.g. with sed)
/// with the actual target backend string.

// TODO: Replace this file with a compiler generated constant string and cleanup
// the driver.
namespace cudaq {
void set_target_backend(const char *);
}

static constexpr const char targetBackendName[] = NVQPP_TARGET_BACKEND_CONFIG;

__attribute__((constructor)) void setTargetBackend() {
  cudaq::set_target_backend(targetBackendName);
}
