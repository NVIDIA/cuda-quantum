/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace cudaq {

/// A kernel may return results dynamically if the size of the result is not a
/// constant at compile-time.
struct KernelThunkResultType {
  void *data_buffer;  ///< Pointer to the first element of an array.
  std::uint64_t size; ///< The size of the buffer in bytes.
};

/// The universal signature of a kernel thunk.
using KernelThunkType = KernelThunkResultType (*)(void *, bool);

/// The degenerate form of a kernel call. In some launch cases, it may be
/// predetermined that the kernel can be called without a thunk.
using KernelDegenerateType = void (*)(void *);

/// In some cases, the launcher will bypass the thunk function and call a
/// degenerate stub. That means that the extra `bool` argument will be ignored
/// by the called kernel and the kernel will not return a dynamic result.
///
/// This is a terrible idea, generally speaking. However, if the launcher
/// neither looks for nor attempts to use the second `bool` argument at all, and
/// the launcher will drop any results returned from the kernel (regardless of
/// type) on the floor anyway, then one may be able to get away with using a
/// degenerate kernel type.
inline KernelDegenerateType
make_degenerate_kernel_type(KernelThunkType func_type) {
  return reinterpret_cast<KernelDegenerateType>(
      reinterpret_cast<void *>(func_type));
}

} // namespace cudaq
