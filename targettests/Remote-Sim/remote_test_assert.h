/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// llvm-lit tests for remote-mqpu tend to hang rather than abort if they hit an
// assertion error using the regular `assert()` call. Use this instead to
// cleanly exit/abort from your main function while printing an error message
// about where the error occurred.
#define REMOTE_TEST_ASSERT(x)                                                  \
  do {                                                                         \
    if (!(x)) {                                                                \
      printf("Assertion failure in %s:%d\nExiting\n", __FILE__, __LINE__);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)
