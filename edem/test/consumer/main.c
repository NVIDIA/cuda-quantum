/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/edem/core.h"
#include "cudaq/gf2/bump_allocator.h"
#include "cudaq/gf2/core.h"
#include <stddef.h>

int main(void) {
  if (cudaq_edem_abi_version() != CUDAQ_EDEM_ABI_VERSION ||
      cudaq_gf2_abi_version() != CUDAQ_GF2_ABI_VERSION)
    return 1;
  cudaq_edem_description empty = {0};
  uint64_t storage[256], scratch[256];
  cudaq_gf2_bump_allocator scratch_arena = {scratch, sizeof(scratch), 0};
  cudaq_gf2_bump_allocator result_arena = {storage, sizeof(storage), 0};
  cudaq_gf2_memory memory = {cudaq_gf2_bump_resource(&scratch_arena),
                             cudaq_gf2_bump_resource(&result_arena)};
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view value;
  if (cudaq_edem_construct(&empty, &memory, &owner, &value) != CUDAQ_EDEM_OK)
    return 2;
  cudaq_gf2_bump_reset(&scratch_arena);
  if (cudaq_edem_read(owner.data, owner.size, &memory, &value) !=
          CUDAQ_EDEM_OK ||
      cudaq_gf2_map_apply(&value.map, 0, 0, 0, 0) != CUDAQ_GF2_OK)
    return 2;
  cudaq_gf2_owned_buffer_release(&owner);
  if (owner.data || owner.size || owner.deallocate || owner.context)
    return 3;
  return 0;
}
