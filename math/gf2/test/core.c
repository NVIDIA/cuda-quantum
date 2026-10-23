/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#include "cudaq/gf2/core.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define OK(call) CHECK((int)(call) == 0)

static void test_gf2(void) {
  /* Unordered input, with a cancelling pair and a tripled entry. */
  cudaq_gf2_entry entries[] = {{1, 2}, {0, 0}, {2, 0}, {0, 0},
                               {1, 2}, {1, 2}, {1, 1}};
  uint64_t offsets[4];
  uint32_t indices[7];
  cudaq_gf2_entry scratch[7];
  cudaq_gf2_csc_view m;
  CHECK(cudaq_gf2_abi_version() == CUDAQ_GF2_ABI_VERSION);
  CHECK(cudaq_gf2_csc_construct(3, 3, entries, 7, offsets, 4, indices, 7,
                                scratch, sizeof(scratch), &m) == CUDAQ_GF2_OK);
  CHECK(m.nonzeros == 3 && indices[0] == 2 && indices[1] == 1 &&
        indices[2] == 1);
  CHECK(offsets[0] == 0 && offsets[1] == 1 && offsets[2] == 2 &&
        offsets[3] == 3);
  CHECK(cudaq_gf2_csc_validate(&m) == CUDAQ_GF2_OK);
  uint8_t x[] = {1, 1, 1}, y[3], shift[] = {1, 1, 0};
  CHECK(cudaq_gf2_csc_apply(&m, x, 3, shift, y, 3) == CUDAQ_GF2_OK);
  CHECK(y[0] == 1 && y[1] == 1 && y[2] == 1);
  x[0] = 2;
  CHECK(cudaq_gf2_csc_apply(&m, x, 3, NULL, y, 3) ==
        CUDAQ_GF2_INVALID_ARGUMENT);
  CHECK(cudaq_gf2_csc_construct(3, 3, entries, 7, offsets, 4, indices, 7,
                                scratch, 0,
                                &m) == CUDAQ_GF2_INSUFFICIENT_CAPACITY);
  entries[0].row = 3;
  CHECK(cudaq_gf2_csc_construct(3, 3, entries, 7, offsets, 4, indices, 7,
                                scratch, sizeof(scratch),
                                &m) == CUDAQ_GF2_INVALID_CONTENT);
  offsets[2] = 8;
  CHECK(cudaq_gf2_csc_validate(&m) == CUDAQ_GF2_INVALID_CONTENT);
  uint64_t empty_offsets = 0;
  CHECK(cudaq_gf2_csc_construct(0, 0, NULL, 0, &empty_offsets, 1, NULL, 0, NULL,
                                0, &m) == CUDAQ_GF2_OK);
  CHECK(cudaq_gf2_csc_apply(&m, NULL, 0, NULL, NULL, 0) == CUDAQ_GF2_OK);
}

int main(void) {
  test_gf2();
  return 0;
}
