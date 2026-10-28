/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#ifndef CUDAQ_EDEM_COMPOSITION_H
#define CUDAQ_EDEM_COMPOSITION_H
#include "cudaq/edem/core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* A borrowed checked closed EDEM. Its fault inputs are original indicator
 * coordinates; its only outputs are detectors, observables and success
 * responses. It uses the existing packed EDEM format, not a second packed
 * representation. The backing packed allocation must outlive the view. */
typedef struct cudaq_edem_dem_view {
  cudaq_edem_view edem;
} cudaq_edem_dem_view;

/* Structural closure only: no input or output code bundles, including
 * zero-width bundles. Success outputs and exclusive laws do not prevent it.
 * On error, the output is unchanged. */
cudaq_edem_status cudaq_edem_is_dem(const cudaq_edem_view *value,
                                    uint32_t *result);

/* Compose a nonempty serial chain. Every output code bundle of occurrence i
 * connects to the same ordinal input code bundle of occurrence i+1. Counts
 * and code presentations must match; there are no implicit closures or zeros.
 * The first input and last output boundaries remain exposed.
 *
 * Each occurrence retains its own fault indicators and exact laws, even when
 * the same definition view occurs repeatedly. Fault, detector, observable and
 * success ports follow occurrence order, then local port order.
 * This ordering supplies occurrence coordinates: (occurrence, local port,
 * local indicator or row). Repeated definitions have distinct occurrences.
 *
 * This first implementation evaluates basis columns through the existing
 * typed-map apply operation. It is a monolithic reference operation, not an
 * incremental/window algorithm. Temporary storage uses memory->scratch;
 * result storage uses memory->result. No source or scratch pointer survives.
 * NULL memory selects heap resources. Owner must be empty; failure preserves
 * owner/view outputs, but need not roll back arena consumption. */
cudaq_edem_status cudaq_edem_compose_chain(const cudaq_edem_view *occurrences,
                                           uint32_t count,
                                           const cudaq_gf2_memory *memory,
                                           cudaq_gf2_owned_buffer *owner,
                                           cudaq_edem_view *result);

/* Strict projection: reject either exposed code boundary, never discard it.
 * The current semantic EDEM format already expresses original fault
 * indicators, so projection retains its blocks, ports and exact laws
 * without changing basis, merging columns or approximating distributions.
 * The independent owned result remains valid after the source is released.
 * Response matrices are the FAULT -> DETECTOR/OBSERVABLE/SUCCESS blocks of
 * result->edem; global columns follow fault-port, then indicator order.
 * Ownership and memory rules are the same as compose_chain. */
cudaq_edem_status cudaq_edem_project_dem(const cudaq_edem_view *value,
                                         const cudaq_gf2_memory *memory,
                                         cudaq_gf2_owned_buffer *owner,
                                         cudaq_edem_dem_view *result);

#ifdef __cplusplus
}
#endif
#endif
