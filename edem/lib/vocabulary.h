/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#pragma once

#include "cudaq/edem/core.h"
#include <cstdint>
#include <iterator>
// EDEM's GF2 type vocabulary: code types first, in name order, then the free
// types in this fixed order.
namespace cudaq_edem_private {
constexpr char code_prefix[] = "edem.code/";
constexpr uint64_t code_prefix_size = sizeof(code_prefix) - 1;
constexpr uint32_t code_wire_count = 3; // V, S, L
constexpr const char *free_names[] = {"edem.detector", "edem.fault",
                                      "edem.observable", "edem.success"};
constexpr cudaq_edem_port_kind free_kinds[] = {
    CUDAQ_EDEM_DETECTOR, CUDAQ_EDEM_FAULT, CUDAQ_EDEM_OBSERVABLE,
    CUDAQ_EDEM_SUCCESS};
constexpr uint32_t free_wire_count = 1;
constexpr uint32_t free_type_count = std::size(free_names);
static_assert(std::size(free_kinds) == free_type_count);
// Position of a free kind among the free types, after the code types.
constexpr uint32_t free_type(cudaq_edem_port_kind kind) {
  uint32_t i = 0;
  while (free_kinds[i] != kind)
    ++i;
  return i;
}
} // namespace cudaq_edem_private
