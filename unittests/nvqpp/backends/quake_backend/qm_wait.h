/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cudaq.h>

// A quantum operation the backend implements, as a vendor would ship it.
// `extern "C"` keeps the symbol verbatim in the payload.
extern "C" void __qm__wait_function(double duration, cudaq::qubit &q);
