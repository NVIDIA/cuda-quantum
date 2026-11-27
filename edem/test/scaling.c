/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#define _POSIX_C_SOURCE 200809L
#include "cudaq/edem/core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define OK(call) CHECK((int)(call) == 0)

static double seconds(void) {
  struct timespec t;
  CHECK(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static void matrix_roundtrip(void) {
  /* Exercise sorting and interning on a large description with repeated and
   * zero matrices, reordered blocks, and remapped code IDs. */
  const uint32_t n = 3000;
  cudaq_edem_matrix_block *blocks = malloc(n * sizeof(*blocks));
  uint32_t *detectors = malloc(n * sizeof(*detectors));
  CHECK(blocks && detectors);
  uint64_t offsets[16][65], zero_offsets[65] = {0};
  uint32_t ids[64], rows[32];
  double probabilities[64];
  for (uint32_t j = 0; j < 16; ++j)
    for (uint32_t c = 0; c <= 64; ++c)
      offsets[j][c] = c > j;
  for (uint32_t i = 0; i < 64; ++i) {
    ids[i] = i;
    probabilities[i] = 0.001;
    if (i < 32)
      rows[i] = i;
  }
  for (uint32_t i = 0; i < n; ++i) {
    detectors[i] = 64;
    cudaq_gf2_csc_view matrix = {64, 64, 1, offsets[(i / 32) % 16],
                                 rows + i % 32};
    if (i % 17 == 0)
      matrix = (cudaq_gf2_csc_view){64, 64, 0, zero_offsets, NULL};
    blocks[i] = (cudaq_edem_matrix_block){
        {CUDAQ_EDEM_DETECTOR, n - i - 1}, {CUDAQ_EDEM_FAULT, 0}, matrix};
  }
  char names[300][24];
  cudaq_edem_code_presentation codes[300];
  for (uint32_t i = 0; i < 300; ++i) {
    int length =
        snprintf(names[i], sizeof(names[i]), "presentation-%u", 299 - i);
    CHECK(length > 0 && (size_t)length < sizeof(names[i]));
    codes[i] =
        (cudaq_edem_code_presentation){names[i], (uint64_t)length, 0, 0, 0};
  }
  cudaq_edem_fault_group group = {CUDAQ_EDEM_INDEPENDENT, 64, ids,
                                  probabilities};
  cudaq_edem_fault_distribution fault = {64, 1, &group};
  uint32_t input_code = 299, output_code = 0;
  cudaq_edem_description d = {0};
  d.shape = (cudaq_edem_shape){1, 1, 1, n, 0, 0};
  d.code_table = (cudaq_edem_code_table){300, codes};
  d.block_count = n;
  d.input_code_indices = &input_code;
  d.output_code_indices = &output_code;
  d.fault_ports_distributions = &fault;
  d.detector_ports_sizes = detectors;
  d.blocks = blocks;
  cudaq_gf2_owned_buffer owner = {0}, canonical = {0};
  cudaq_edem_view value;
  OK(cudaq_edem_construct(&d, NULL, &owner, &value));
  for (uint32_t i = 0; i < n / 2; ++i) {
    cudaq_edem_matrix_block tmp = blocks[i];
    blocks[i] = blocks[n - i - 1];
    blocks[n - i - 1] = tmp;
  }
  OK(cudaq_edem_construct(&d, NULL, &canonical, &value));
  CHECK(owner.size == canonical.size &&
        memcmp(owner.data, canonical.data, (size_t)owner.size) == 0);
  OK(cudaq_edem_read(owner.data, owner.size, NULL, &value));
  cudaq_gf2_owned_buffer_release(&canonical);
  cudaq_gf2_owned_buffer_release(&owner);
  free(detectors);
  free(blocks);
}

int main(void) {
  matrix_roundtrip();
  const uint32_t counts[] = {400, 800, 3200, 12800, 51200, 200000};
  puts("groups,packed_bytes,construct_ms,read_ms");
  for (unsigned trial = 0; trial < sizeof(counts) / sizeof(counts[0]);
       ++trial) {
    uint32_t n = counts[trial];
    uint32_t *ids = malloc(3 * n * sizeof(*ids));
    double *probabilities = malloc(3 * n * sizeof(*probabilities));
    cudaq_edem_fault_group *groups = malloc(n * sizeof(*groups));
    CHECK(ids && probabilities && groups);
    for (uint32_t i = 0; i < 3 * n; ++i) {
      ids[i] = i;
      probabilities[i] = 0.001;
    }
    for (uint32_t i = 0; i < n; ++i)
      groups[i] = (cudaq_edem_fault_group){CUDAQ_EDEM_EXCLUSIVE, 3, ids + 3 * i,
                                           probabilities + 3 * i};
    uint32_t random = 391;
    for (uint32_t i = n; i > 1; --i) {
      random = random * 1664525u + 1013904223u;
      uint32_t j = random % i;
      cudaq_edem_fault_group tmp = groups[i - 1];
      groups[i - 1] = groups[j];
      groups[j] = tmp;
    }
    cudaq_edem_fault_distribution fault = {3 * n, n, groups};
    cudaq_edem_description d = {0};
    d.shape.fault_ports_count = 1;
    d.fault_ports_distributions = &fault;
    cudaq_gf2_owned_buffer owner = {0}, canonical = {0};
    cudaq_edem_view value;
    double start = seconds();
    OK(cudaq_edem_construct(&d, NULL, &owner, &value));
    double construction = seconds() - start;
    uint64_t bytes = owner.size;
    start = seconds();
    OK(cudaq_edem_read(owner.data, bytes, NULL, &value));
    double reading = seconds() - start;
    /* Canonical and shuffled input orders produce the same bytes. */
    for (uint32_t i = 0; i < n; ++i)
      groups[i] = (cudaq_edem_fault_group){CUDAQ_EDEM_EXCLUSIVE, 3, ids + 3 * i,
                                           probabilities + 3 * i};
    OK(cudaq_edem_construct(&d, NULL, &canonical, &value));
    CHECK(canonical.size == bytes);
    CHECK(memcmp(canonical.data, owner.data, (size_t)bytes) == 0);
    OK(cudaq_edem_read(canonical.data, bytes, NULL, &value));
    /* A duplicate in the final groups stays locally sorted and is rejected
     * by the exact-coverage check. */
    ids[3 * n - 3] = 3 * n - 5;
    CHECK(cudaq_edem_validate(&d, NULL) == CUDAQ_EDEM_INVALID_CONTENT);
    cudaq_edem_fault_group last;
    OK(cudaq_edem_get_group(&value, 0, n - 1, &last));
    ((uint32_t *)last.fault_ids)[0] = 3 * n - 5;
    CHECK(cudaq_edem_read(canonical.data, bytes, NULL, &value) ==
          CUDAQ_EDEM_INVALID_CONTENT);
    printf("%u,%llu,%.3f,%.3f\n", n, (unsigned long long)bytes,
           construction * 1e3, reading * 1e3);
    cudaq_gf2_owned_buffer_release(&canonical);
    cudaq_gf2_owned_buffer_release(&owner);
    free(groups);
    free(probabilities);
    free(ids);
  }
  return 0;
}
