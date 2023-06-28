/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdio>
#include <cstdlib>

// The TODO macro is to be used to mark parts of the runtime that are not yet
// implemented. Instead of the runtime crashing or generating bad code for
// features that have not been implemented, we want the runtime to generate a
// NYI error message and exit. This TODO in the runtime code does not depend
// upon any LLVM project libraries.
#define TODO(msg)                                                              \
  {                                                                            \
    fprintf(stderr, __FILE__ ":%d: not yet implemented %s\n", __LINE__, msg);  \
    abort();                                                                   \
  }
