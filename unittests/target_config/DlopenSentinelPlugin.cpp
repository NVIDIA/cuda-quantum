/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cstdlib>
#include <fstream>

namespace {
// This shared library is intentionally tiny: the B3 plugin-libraries test lists
// it in target YAML and expects LinkedLibraryHolder::setTarget() to dlopen it.
// The constructor writes a sentinel file so the test can observe that load.
[[gnu::constructor]] void writeDlopenSentinel() {
  const char *sentinelPath = std::getenv("CUDAQ_DLOPEN_SENTINEL_PATH");
  if (!sentinelPath || !*sentinelPath)
    return;

  std::ofstream(sentinelPath) << "loaded\n";
}
} // namespace
