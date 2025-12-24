/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file domains.hpp
/// @brief Domain identifiers for logical component grouping
///
/// This header provides domain identifiers for logical component grouping.
/// These are macro identifiers for zero-overhead preprocessor-based dispatch.
///
/// Usage:
///   NVQLINK_TRACE_FULL(DOMAIN_DAEMON, "Something happened")
///   NVQLINK_LOG_INFO(DOMAIN_DAEMON, "Log message")

#define DOMAIN_DAEMON daemon
#define DOMAIN_DISPATCHER dispatcher
#define DOMAIN_MEMORY memory
#define DOMAIN_CHANNEL channel
#define DOMAIN_USER user
#define DOMAIN_GPU gpu
