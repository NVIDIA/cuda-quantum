/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#ifndef CUDAQ_EDEM_OWNED_HPP
#define CUDAQ_EDEM_OWNED_HPP

#include "cudaq/edem/core.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define CUDAQ_EDEM_CPP_LOCAL __attribute__((visibility("hidden")))
#else
#define CUDAQ_EDEM_CPP_LOCAL
#endif

namespace cudaq::edem {
inline namespace v0 {

/// C++17 ownership convenience over the C interface, with no exceptions or
/// RTTI.
///
/// Native consumers, including compiler integrations, can keep a PackedEDEM
/// internally and translate its C status codes into their own error
/// conventions. Applications may also use this installed header directly. A
/// library with a public C++ interface can reuse this helper inside its own
/// model implementation; its public model supplies that library's domain
/// operations and lifetime policy.
///
/// This header is compiled in each consumer. Its type has hidden visibility and
/// is not a cross-library C++ ABI: exchange packed bytes and the C ownership
/// record from release(), then validate with adopt() in the receiving consumer.
/// No semantic builder, allocator state, or cache is added by this helper.
///
/// Allocator callbacks and contexts must outlive this owner and any record
/// released from it. The helper does not retain allocator or module keepalives;
/// consumers must destroy payload owners before their callback dependencies.
class CUDAQ_EDEM_CPP_LOCAL PackedEDEM final {
  cudaq_gf2_owned_buffer owner_{};
  cudaq_edem_view view_{};

public:
  PackedEDEM() noexcept = default;
  PackedEDEM(const PackedEDEM &) = delete;
  PackedEDEM &operator=(const PackedEDEM &) = delete;
  PackedEDEM(PackedEDEM &&other) noexcept
      : owner_(std::exchange(other.owner_, {})),
        view_(std::exchange(other.view_, {})) {}
  PackedEDEM &operator=(PackedEDEM &&other) noexcept {
    if (this != &other) {
      reset();
      owner_ = std::exchange(other.owner_, {});
      view_ = std::exchange(other.view_, {});
    }
    return *this;
  }
  ~PackedEDEM() noexcept { reset(); }

  [[nodiscard]] bool empty() const noexcept { return owner_.data == nullptr; }
  [[nodiscard]] uint64_t size() const noexcept { return owner_.size; }
  [[nodiscard]] const void *data() const & noexcept { return owner_.data; }
  const void *data() const && = delete;

  /// Requires a nonempty owner. The returned reference belongs to this object;
  /// copy the C view token if it must survive a move. Its backing buffer
  /// remains valid until the destination owner releases it. Views of
  /// temporaries are forbidden, and moved-from/reset/released objects contain
  /// an empty token.
  [[nodiscard]] const cudaq_edem_view &view() const & noexcept { return view_; }
  const cudaq_edem_view &view() const && = delete;

  void reset() noexcept {
    view_ = {};
    cudaq_gf2_owned_buffer_release(&owner_);
  }

  /// Construct through the semantic C API. All setup operations preserve result
  /// on failure and replace it on success. The optional per-call memory
  /// provider supports reusable scratch; neither the provider nor its scratch
  /// is retained.
  [[nodiscard]] static cudaq_edem_status
  construct(const cudaq_edem_description &description, PackedEDEM &result,
            const cudaq_gf2_memory *memory = nullptr) noexcept {
    PackedEDEM candidate;
    auto status = cudaq_edem_construct(&description, memory, &candidate.owner_,
                                       &candidate.view_);
    if (status == CUDAQ_EDEM_OK)
      result = std::move(candidate);
    return status;
  }

  /// Validate an aligned, uniquely owned C record before transferring it.
  /// Success clears source without copying its payload. Failure preserves both
  /// source and result. The record and result must not alias the same
  /// ownership.
  [[nodiscard]] static cudaq_edem_status
  adopt(cudaq_gf2_owned_buffer &source, PackedEDEM &result,
        const cudaq_gf2_memory *memory = nullptr) noexcept {
    if (!source.data || !source.size || !source.deallocate)
      return CUDAQ_EDEM_INVALID_ARGUMENT;
    PackedEDEM candidate;
    auto status =
        cudaq_edem_read(source.data, source.size, memory, &candidate.view_);
    if (status == CUDAQ_EDEM_OK) {
      candidate.owner_ = std::exchange(source, {});
      result = std::move(candidate);
    }
    return status;
  }

  /// Make an independent allocation and obtain its checked view through a fresh
  /// validating read. Cloning into this same object is supported.
  [[nodiscard]] cudaq_edem_status
  clone(PackedEDEM &result,
        const cudaq_gf2_memory *memory = nullptr) const noexcept {
    if (empty())
      return CUDAQ_EDEM_INVALID_ARGUMENT;
    if (owner_.size > SIZE_MAX)
      return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
    auto resource = memory ? memory->result : cudaq_gf2_default_allocator();
    if (!resource.allocate || !resource.deallocate)
      return CUDAQ_EDEM_INVALID_ARGUMENT;
    void *data = resource.allocate(resource.context, owner_.size,
                                   CUDAQ_EDEM_PACKED_ALIGNMENT);
    if (!data)
      return CUDAQ_EDEM_ALLOCATION_FAILED;
    PackedEDEM candidate;
    candidate.owner_ = {data, owner_.size, resource.deallocate,
                        resource.context};
    if (reinterpret_cast<std::uintptr_t>(data) % CUDAQ_EDEM_PACKED_ALIGNMENT)
      return CUDAQ_EDEM_INVALID_ARGUMENT;
    std::memcpy(data, owner_.data, static_cast<std::size_t>(owner_.size));
    auto status = cudaq_edem_read(data, owner_.size, memory, &candidate.view_);
    if (status == CUDAQ_EDEM_OK)
      result = std::move(candidate);
    return status;
  }

  /// Transfer the C ownership record to another package or language binding.
  /// The caller must eventually release or adopt it, keeping callbacks alive.
  [[nodiscard]] cudaq_gf2_owned_buffer release() noexcept {
    view_ = {};
    return std::exchange(owner_, {});
  }
};

} // namespace v0
} // namespace cudaq::edem

#undef CUDAQ_EDEM_CPP_LOCAL
#endif
