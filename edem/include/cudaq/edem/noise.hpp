/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/edem/noise.h"
#include "cudaq/edem/owned.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define CUDAQ_EDEM_NOISE_CPP_LOCAL __attribute__((visibility("hidden")))
#else
#define CUDAQ_EDEM_NOISE_CPP_LOCAL
#endif

namespace cudaq::edem {
inline namespace v0 {

/// Consumer-local C++17 owner for a response-column mapping. The C ownership
/// record, rather than this helper's C++ type, may cross package boundaries.
class CUDAQ_EDEM_NOISE_CPP_LOCAL ColumnMapping {
  cudaq_gf2_owned_buffer storage_{};
  friend cudaq_edem_status
  coalesce_dem_columns(const PackedEDEM &, PackedEDEM &, ColumnMapping &,
                       const cudaq_gf2_memory *) noexcept;

public:
  ColumnMapping() noexcept = default;
  ColumnMapping(const ColumnMapping &) = delete;
  ColumnMapping &operator=(const ColumnMapping &) = delete;
  ColumnMapping(ColumnMapping &&other) noexcept
      : storage_(std::exchange(other.storage_, {})) {}
  ColumnMapping &operator=(ColumnMapping &&other) noexcept {
    if (this != &other) {
      cudaq_gf2_owned_buffer_release(&storage_);
      storage_ = std::exchange(other.storage_, {});
    }
    return *this;
  }
  ~ColumnMapping() { cudaq_gf2_owned_buffer_release(&storage_); }
  uint64_t size() const noexcept { return storage_.size / sizeof(uint64_t); }
  const uint64_t *data() const & noexcept {
    return static_cast<const uint64_t *>(storage_.data);
  }
  const uint64_t *data() const && = delete;
  cudaq_gf2_owned_buffer release() noexcept {
    return std::exchange(storage_, {});
  }
};

/// Exact independent-mechanism conversion; response column coordinates stay
/// unchanged. The source retains the original event laws. No exceptions.
[[nodiscard]] CUDAQ_EDEM_NOISE_CPP_LOCAL inline cudaq_edem_status
factorize_dem(const PackedEDEM &value, PackedEDEM &result,
              const cudaq_gf2_memory *memory = nullptr) noexcept {
  cudaq_edem_dem_view input{value.view()}, view{};
  cudaq_gf2_owned_buffer owner{};
  auto status =
      cudaq_edem_into_independent_faults_dem(&input, memory, &owner, &view);
  if (status == CUDAQ_EDEM_OK)
    status = PackedEDEM::adopt(owner, result, memory);
  cudaq_gf2_owned_buffer_release(&owner);
  return status;
}

/// Merge independent response columns. Both outputs are replaced only on
/// success. UINT64_MAX mapping entries identify discarded mechanisms.
[[nodiscard]] CUDAQ_EDEM_NOISE_CPP_LOCAL inline cudaq_edem_status
coalesce_dem_columns(const PackedEDEM &value, PackedEDEM &result,
                     ColumnMapping &mapping,
                     const cudaq_gf2_memory *memory = nullptr) noexcept {
  cudaq_edem_dem_view input{value.view()}, view{};
  cudaq_gf2_owned_buffer owner{}, columns{};
  auto status =
      cudaq_edem_coalesce_dem_columns(&input, memory, &owner, &view, &columns);
  PackedEDEM next;
  if (status == CUDAQ_EDEM_OK)
    status = PackedEDEM::adopt(owner, next, memory);
  if (status == CUDAQ_EDEM_OK) {
    result = std::move(next);
    cudaq_gf2_owned_buffer_release(&mapping.storage_);
    mapping.storage_ = std::exchange(columns, {});
  }
  cudaq_gf2_owned_buffer_release(&owner);
  cudaq_gf2_owned_buffer_release(&columns);
  return status;
}

} // namespace v0
} // namespace cudaq::edem
#undef CUDAQ_EDEM_NOISE_CPP_LOCAL
