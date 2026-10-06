/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "Trivial/TrivialDialect.h"
#include "Trivial/TrivialOps.h"

using namespace mlir;

// TableGen-generated dialect definitions (constructor, destructor, type id).
#include "Trivial/TrivialDialect.cpp.inc"

void trivial::TrivialDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Trivial/TrivialOps.cpp.inc"
      >();
}
