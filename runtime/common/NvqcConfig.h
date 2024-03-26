/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <stdlib.h>
#include <string>
namespace cudaq {
inline std::string getNvqcNcaId() {
  // Default NVQC NVIDIA Cloud Account (NCA) Id
  static constexpr const char *NVQC_NCA_ID =
      "audj0Ow_82RT0BbiewKaIryIdZWiSrOqiiDSaA8w7a8";
  // Allows runtime override by environment variable.
  if (auto ncaIdVar = std::getenv("NVQC_NCA_ID"))
    return std::string(ncaIdVar);
  return NVQC_NCA_ID;
}
} // namespace cudaq
