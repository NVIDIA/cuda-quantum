/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq_internal/compiler/RuntimeMLIR.h"

// Pass registration is done through the 'register_dialect' python call.
// The native target initialization is built into the MLIR python extension.
void cudaq_internal::compiler::initializeLangMLIR() {}
