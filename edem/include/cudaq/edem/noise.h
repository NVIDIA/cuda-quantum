/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#ifndef CUDAQ_EDEM_NOISE_H
#define CUDAQ_EDEM_NOISE_H

#include "cudaq/edem/composition.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact conversion to independent binary mechanisms. Independent groups pass
 * through unchanged. Exclusive groups must have 3 or 15 equally probable
 * alternatives whose complete response columns, together with identity, form
 * a uniform linear image of the one-/two-qubit Pauli group. The total channel
 * probability must be <= 3/4 or 15/16, respectively. Other distributions return
 * UNSUPPORTED_DISTRIBUTION; no approximation or renormalization is performed.
 *
 * Response matrices, partitions, and column coordinates are preserved. A
 * converted coordinate labels an independent mechanism with the same response,
 * not the original physical event. The joint detector/observable/success law
 * is preserved. Retain the source DEM for its original event laws and bindings.
 * The source must be a checked closed DEM. Result memory/ownership and failure
 * rules are those of project_dem; owner and result are unchanged on failure. */
cudaq_edem_status cudaq_edem_into_independent_faults_dem(
    const cudaq_edem_dem_view *value, const cudaq_gf2_memory *memory,
    cudaq_gf2_owned_buffer *owner, cudaq_edem_dem_view *result);

/* Coalesce independent mechanisms with equal complete response columns. All
 * groups must be independent; cudaq_edem_into_independent_faults_dem is an
 * explicit preceding step when needed. Probabilities combine by parity,
 * p+q-2pq, with stable evaluation. Zero responses and zero-probability combined
 * mechanisms are removed.
 *
 * Global rows retain detector, observable, then success order, each in original
 * partition/row order. Result partitions consolidate to one fault partition
 * and at most one partition per output kind. Distinct nonzero columns are
 * ordered lexicographically by their sorted global row IDs. This operation
 * uses sparse storage and O(F log F) column comparisons, not dense F-by-row
 * workspaces or a Cartesian product of occurrence partitions.
 *
 * column_map owns a native uint64_t array with one entry per input column, in
 * fault-partition/indicator order. Entries are result column IDs, or UINT64_MAX
 * for discarded mechanisms. This maps response coordinates, not realizations
 * of the original categorical events. Release it with
 * cudaq_gf2_owned_buffer_release. Both owners must be empty; all output records
 * must be disjoint. Failure preserves both owners and result. No input or
 * scratch pointer survives. NULL memory selects the default heap resources. */
cudaq_edem_status cudaq_edem_coalesce_dem_columns(
    const cudaq_edem_dem_view *value, const cudaq_gf2_memory *memory,
    cudaq_gf2_owned_buffer *owner, cudaq_edem_dem_view *result,
    cudaq_gf2_owned_buffer *column_map);

#ifdef __cplusplus
}
#endif
#endif
