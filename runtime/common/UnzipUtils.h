/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <filesystem>
namespace cudaq {
namespace utils {
// Helper to unzip a zip file to the specified output directory.
void unzip(const std::filesystem::path &zipFile,
           const std::filesystem::path &outputDir);
} // namespace utils
} // namespace cudaq
