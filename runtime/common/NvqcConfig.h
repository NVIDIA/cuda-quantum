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

static constexpr const char *NVQC_NCA_ID_ENV_VAR = "NVQC_NCA_ID";
static constexpr const char *DEV_NVQC_NCA_ID =
    "mZraB3k06kOd8aPhD6MVXJwBVZ67aXDLsfmDo4MYXDs";
static constexpr const char *PROD_NVQC_NCA_ID =
    "audj0Ow_82RT0BbiewKaIryIdZWiSrOqiiDSaA8w7a8";

inline bool isNvqcNcaIdOverridden() {
  return std::getenv(NVQC_NCA_ID_ENV_VAR) != nullptr;
}

inline std::string getNvqcNcaId() {
  // Allows runtime override by environment variable.
  if (auto ncaIdVar = std::getenv(NVQC_NCA_ID_ENV_VAR))
    return std::string(ncaIdVar);
  return PROD_NVQC_NCA_ID;
}
} // namespace cudaq
