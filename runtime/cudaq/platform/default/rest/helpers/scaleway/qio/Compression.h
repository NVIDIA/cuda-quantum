/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include <cstddef>
#include <string>

namespace cudaq::qio {
/// Default upper bound on the number of bytes `gzipDecompress` will produce.
inline constexpr std::size_t defaultMaxDecompressedSize = 512UL * 1024 * 1024;

std::string gzipCompress(const std::string &input);

/// Decompress a `zlib` stream, which produces less than `maxOutputSize` bytes.
std::string
gzipDecompress(const std::string &input,
               std::size_t maxOutputSize = defaultMaxDecompressedSize);
} // namespace cudaq::qio
