/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

/// @file cpu_relax.h
/// @brief CPU pause/yield hint for busy-poll spin loops.

#ifndef CUDAQ_REALTIME_CPU_RELAX
#if defined(__x86_64__)
#include <xmmintrin.h> // _mm_pause()
#define CUDAQ_REALTIME_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define CUDAQ_REALTIME_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define CUDAQ_REALTIME_CPU_RELAX()                                             \
  do {                                                                         \
  } while (0)
#endif
#endif
