/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#ifndef CUDAQ_GF2_CSC_XOR_H
#define CUDAQ_GF2_CSC_XOR_H

#include <stdint.h>

/* Private host loop for already-validated matrices and vectors. */
static inline void cudaq_gf2_csc_xor_kernel(uint32_t columns,
                                            const uint64_t *column_offsets,
                                            const uint32_t *row_indices,
                                            const uint8_t *input,
                                            uint8_t *output) {
  for (uint64_t column = 0; column < columns; ++column)
    if (input[column])
      for (uint64_t p = column_offsets[column]; p < column_offsets[column + 1];
           ++p)
        output[row_indices[p]] ^= 1;
}

#endif
