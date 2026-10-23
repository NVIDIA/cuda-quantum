/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#include "cudaq/edem/core.h"
#include "cudaq/gf2/core.h"

#include <type_traits>

static_assert(std::is_standard_layout_v<cudaq_edem_description>);
static_assert(std::is_trivially_copyable_v<cudaq_gf2_csc_view>);
static_assert(std::is_standard_layout_v<cudaq_gf2_owned_buffer>);
static_assert(std::is_trivially_copyable_v<cudaq_gf2_memory>);

static_assert(std::is_trivially_copyable_v<cudaq_edem_view>);
static_assert(!std::is_convertible_v<cudaq_gf2_map_view *, cudaq_edem_view *>);
static_assert(!std::is_invocable_v<decltype(&cudaq_edem_get_code),
                                   const cudaq_gf2_map_view *, uint32_t,
                                   cudaq_edem_code_presentation *>);
static_assert(!std::is_invocable_v<decltype(&cudaq_edem_get_group),
                                   const cudaq_gf2_map_view *, uint32_t,
                                   uint32_t, cudaq_edem_fault_group *>);

int main() {
  return cudaq_edem_abi_version() != CUDAQ_EDEM_ABI_VERSION ||
         cudaq_gf2_abi_version() != CUDAQ_GF2_ABI_VERSION;
}
