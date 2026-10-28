/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#ifndef CUDAQ_EDEM_FORMAT_H
#define CUDAQ_EDEM_FORMAT_H
#include "cudaq/gf2/format.h"
/* EDEM is a GF2 linear map with these two required sections. All references
 * inside a section are relative to that section. Probabilities are IEEE
 * binary64. These schemas belong to EDEM; GF2 does not interpret them. */
#define CUDAQ_EDEM_SECTION_PRESENTATIONS UINT32_C(0x00010000)
#define CUDAQ_EDEM_SECTION_DISTRIBUTIONS UINT32_C(0x00010001)
typedef struct cudaq_edem_packed_code {
  uint32_t type, syndrome_count, virtual_outcome_count, logical_qubit_count;
} cudaq_edem_packed_code;
typedef struct cudaq_edem_packed_fault {
  uint32_t bundle, size, group_count, reserved;
  uint64_t groups;
} cudaq_edem_packed_fault;
typedef struct cudaq_edem_packed_group {
  uint32_t kind, size;
  uint64_t fault_ids, probabilities;
} cudaq_edem_packed_group;
#endif
