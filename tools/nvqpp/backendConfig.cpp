/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// This file is meant to be used by the nvq++ driver script, the
/// NVQPP_QPUD_BACKEND_CONFIG string must be replaced (e.g. with sed)
/// with the actual target backend string.
namespace cudaq {
void set_target_backend(const char *);
}
constexpr static const char ____targetBackend[] = NVQPP_TARGET_BACKEND_CONFIG;
__attribute__((constructor)) void setTargetBackend() {
  cudaq::set_target_backend(____targetBackend);
}
