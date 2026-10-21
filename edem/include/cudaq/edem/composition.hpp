/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#ifndef CUDAQ_EDEM_COMPOSITION_HPP
#define CUDAQ_EDEM_COMPOSITION_HPP

#include "cudaq/edem/composition.h"
#include "cudaq/edem/owned.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define CUDAQ_EDEM_COMPOSITION_CPP_LOCAL __attribute__((visibility("hidden")))
#else
#define CUDAQ_EDEM_COMPOSITION_CPP_LOCAL
#endif

namespace cudaq::edem {
inline namespace v0 {

/// C++17 conveniences over the C operations. Results use the same move-only,
/// consumer-local owner as construction. Failure preserves result; allocation
/// errors are status codes, and optional resources can reuse caller scratch.
/// Exchange the C ownership record, rather than these helpers' C++ types,
/// across package boundaries.
[[nodiscard]] CUDAQ_EDEM_COMPOSITION_CPP_LOCAL inline cudaq_edem_status
compose_chain(const cudaq_edem_view *occurrences, uint32_t count,
              PackedEDEM &result,
              const cudaq_gf2_memory *memory = nullptr) noexcept {
  cudaq_gf2_owned_buffer owner{};
  cudaq_edem_view view{};
  auto status =
      cudaq_edem_compose_chain(occurrences, count, memory, &owner, &view);
  if (status == CUDAQ_EDEM_OK)
    status = PackedEDEM::adopt(owner, result, memory);
  cudaq_gf2_owned_buffer_release(&owner);
  return status;
}

/// Strictly project a closed EDEM, retaining original columns, exact laws and
/// success responses. The result is an independent packed EDEM owner; closure
/// and the decoder-facing interpretation are established by the C operation.
[[nodiscard]] CUDAQ_EDEM_COMPOSITION_CPP_LOCAL inline cudaq_edem_status
project_DEM(const PackedEDEM &value, PackedEDEM &result,
            const cudaq_gf2_memory *memory = nullptr) noexcept {
  cudaq_gf2_owned_buffer owner{};
  cudaq_edem_dem_view view{};
  auto status = cudaq_edem_project_dem(&value.view(), memory, &owner, &view);
  if (status == CUDAQ_EDEM_OK)
    status = PackedEDEM::adopt(owner, result, memory);
  cudaq_gf2_owned_buffer_release(&owner);
  return status;
}

} // namespace v0
} // namespace cudaq::edem

#undef CUDAQ_EDEM_COMPOSITION_CPP_LOCAL
#endif
