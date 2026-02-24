/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "Base64.h"
#include "llvm/Support/Base64.h"
#include <vector>

namespace cudaq::qio {

std::string encodeBase64(std::string_view input) {
  return llvm::encodeBase64(input);
}

std::string decodeBase64(std::string_view input) {
  std::vector<char> output;
  if (auto err = llvm::decodeBase64(input, output))
    throw std::runtime_error("Base64 decode failed");
  return std::string(output.begin(), output.end());
}

} // namespace cudaq::qio