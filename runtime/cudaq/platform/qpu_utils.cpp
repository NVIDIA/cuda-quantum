/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu_utils.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include "llvm/Support/Base64.h"

using namespace cudaq;

void detail::parseTargetConfigYml(const std::string &yamlContent,
                                  config::TargetConfig &targetConfig) {
  llvm::yaml::Input Input(yamlContent.c_str());
  Input >> targetConfig;
}

std::string detail::decodeBase64(const std::string &encoded) {
  std::vector<char> decoded_vec;
  if (auto err = llvm::decodeBase64(encoded, decoded_vec))
    throw std::runtime_error("DecodeBase64 error");
  return std::string(decoded_vec.data(), decoded_vec.size());
}

bool detail::isAnalogHamiltonianKernel(const std::string &kernelName) {
  return kernelName.find(cudaq::runtime::cudaqAHKPrefixName) == 0;
}
