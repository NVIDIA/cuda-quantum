/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#pragma once

#include <cstdint>

/* Private host loop for already-validated matrices and vectors. */
static inline void cudaq_gf2_csc_xor_kernel(std::uint32_t columns,
                                            const std::uint64_t *column_offsets,
                                            const std::uint32_t *row_indices,
                                            const std::uint8_t *input,
                                            std::uint8_t *output) {
  for (std::uint64_t column = 0; column < columns; ++column)
    if (input[column])
      for (std::uint64_t p = column_offsets[column];
           p < column_offsets[column + 1]; ++p)
        output[row_indices[p]] ^= 1;
}
