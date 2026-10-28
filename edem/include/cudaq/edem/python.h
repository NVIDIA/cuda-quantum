/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#ifndef CUDAQ_EDEM_PYTHON_H
#define CUDAQ_EDEM_PYTHON_H
#include "cudaq/gf2/memory.h"

/* Unstable v0.x native-to-Python adoption protocol (no Python headers
 * needed). A named PyCapsule points to a stable cudaq_gf2_owned_buffer record.
 * The producer owns the record storage and keeps callback/context dependencies
 * alive for the capsule's lifetime. EDEM.adopt validates before transfer,
 * clears the source record, marks the capsule used, and retains it as a
 * dependency to keep it alive. Validation failure leaves the capsule/owner
 * usable. Adoption is single-use. Producer destructors release any remaining
 * owner even when the name has changed, then dispose of record storage. Checked
 * native views do not cross this boundary.
 */
#define CUDAQ_EDEM_OWNED_BUFFER_CAPSULE "cudaq_edem.owned_buffer.v0"
#define CUDAQ_EDEM_USED_BUFFER_CAPSULE "cudaq_edem.owned_buffer.v0.used"
#endif
