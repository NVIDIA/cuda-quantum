/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include <string>

namespace cudaq::qio::compression {
    std::string gzipCompress(const std::string &input);
    std::string gzipDecompress(const std::string &input);

    std::string base64Encode(const std::string &input);
    std::string base64Decode(const std::string &input);

    bool isBase64(unsigned char c);
}
