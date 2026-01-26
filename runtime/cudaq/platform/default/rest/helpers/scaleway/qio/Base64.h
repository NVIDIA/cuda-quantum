/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace cudaq::qio {

std::string encodeBase64(std::string_view input);
std::string decodeBase64(std::string_view input);

} // namespace cudaq::qio
