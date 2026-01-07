/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// llvm-lit tests for remote-mqpu hang rather than abort if they hit an
// assertion error using the regular `assert()` call. The reason is that the
// AutoLaunchRestServerProcess destructor is not called when the assertion
// fails, so the `cudaq-qpud` process stays running in the background, and the
// llvm-lit wrapper thinks the test has not completed, so it just hangs waiting
// for completion.
// Note: this is similar to what would happen if abort() were called.
// E.g., https://man7.org/linux/man-pages/man3/abort.3.html says:
// "As with other cases of abnormal termination the functions registered with
// atexit(3) and on_exit(3) are not called."

// Use this macro instead to cleanly exit/abort from your main function while
// printing an error message about where the error occurred.
#define REMOTE_TEST_ASSERT(x)                                                  \
  do {                                                                         \
    if (!(x)) {                                                                \
      printf("Assertion failure in %s:%d\nExiting\n", __FILE__, __LINE__);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)
