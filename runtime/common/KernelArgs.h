/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/NamedVariantStore.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace cudaq {

/// Runtime arguments for a kernel launch.
///
/// Kernel arguments are carried across `launchKernel`, `launchModule`, and
/// `compileModule` in two conventions:
///
/// - **Packed**: a contiguous buffer produced by the compiler's `argsCreator`.
/// - **Type-erased**: a `std::vector<void *>` where each entry points to an
///   individual argument value.
///
/// The arguments are stored as references or pointers. It is up to the caller
/// to ensure the lifetime of the arguments.
///
/// `KernelArgs` can hold either representation, or both at once — the hybrid
/// path (`cudaq::hybridLaunchKernel`) supplies both so the receiving platform
/// can pick the shape it prefers.
class KernelArgs {
public:
  struct PackedArgs {
    std::span<std::byte> data;
    std::uint64_t resultOffset = 0;
    PackedArgs(void *data, std::uint64_t size, std::uint64_t resultOffset)
        : data(std::span<std::byte>(reinterpret_cast<std::byte *>(data), size)),
          resultOffset(resultOffset) {}
  };
  using TypeErasedArgs = std::span<void *const>;

  /// Default: the kernel takes no arguments.
  KernelArgs() = default;

  /// Wrap a packed buffer. If `packed.data == nullptr`, stores nothing.
  KernelArgs(PackedArgs packed);

  /// Wrap a type-erased pointer vector. If empty, stores nothing.
  KernelArgs(TypeErasedArgs rawArgs);

  /// Wrap both representations (hybrid). Either or both may be dropped
  /// if empty per the single-representation constructors.
  KernelArgs(PackedArgs packed, TypeErasedArgs rawArgs);

  /// Returns the packed buffer description if present, else `nullptr`.
  std::optional<PackedArgs> getPacked() const;

  /// Returns the type-erased pointer vector if present, else `nullptr`.
  std::optional<TypeErasedArgs> getTypeErased() const;

  bool hasPacked() const { return getPacked().has_value(); }
  bool hasTypeErased() const { return getTypeErased().has_value(); }

  /// True iff no representation is stored (kernel takes no arguments).
  bool empty() const { return !hasPacked() && !hasTypeErased(); }

private:
  detail::NamedVariantStore<PackedArgs, TypeErasedArgs> store;
};

} // namespace cudaq
