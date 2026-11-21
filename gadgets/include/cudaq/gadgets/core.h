/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Numeric compatibility revisions for v0.x development, not major releases. */
#define CUDAQ_GADGETS_ABI_VERSION 1u

uint32_t cudaq_gadgets_abi_version(void);

#ifdef __cplusplus
}
#endif
