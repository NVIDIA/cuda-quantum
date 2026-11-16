/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "Trivial/TrivialOps.h"

using namespace mlir;

// TableGen-generated op definitions.
#define GET_OP_CLASSES
#include "Trivial/TrivialOps.cpp.inc"
