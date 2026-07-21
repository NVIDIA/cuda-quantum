/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <vector>

// expected-note@* 0+ {{}}

// A negative or overflowing `observable_index` is only diagnosed at compile
// time when it is a compile-time constant; runtime values are checked when
// the kernel executes.

struct NegativeObservableIndex {
  void operator()() __qpu__ {
    cudaq::qvector qs(3);
    auto h = mz(qs);
    // clang-format off
    // expected-error@+2{{`cudaq::logical_observable` `observable_index` must be in the range [0, 2^63 - 1]}}
    // expected-error@+1{{statement not supported in qpu kernel}}
    cudaq::logical_observable(h, static_cast<std::size_t>(-1));
    // clang-format on
  }
};

struct OverflowObservableIndex {
  void operator()() __qpu__ {
    cudaq::qvector qs(3);
    auto h = mz(qs);
    // clang-format off
    // expected-error@+2{{`cudaq::logical_observable` `observable_index` must be in the range [0, 2^63 - 1]}}
    // expected-error@+1{{statement not supported in qpu kernel}}
    cudaq::logical_observable(h, 9223372036854775808ULL);
    // clang-format on
  }
};
