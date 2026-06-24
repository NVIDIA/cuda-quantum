/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CustomOp.h"

namespace cudaq {
customOpRegistry &customOpRegistry::getInstance() {
  static customOpRegistry instance;
  return instance;
}

void customOpRegistry::clearRegisteredOperations() {
  std::unique_lock<std::shared_mutex> lock(mtx);
  registeredOperations.clear();
}

bool customOpRegistry::isOperationRegistered(const std::string &name) {
  std::shared_lock<std::shared_mutex> lock(mtx);
  return registeredOperations.find(name) != registeredOperations.end();
}

const unitary_operation &
customOpRegistry::getOperation(const std::string &name) {
  std::shared_lock<std::shared_mutex> lock(mtx);
  auto iter = registeredOperations.find(name);
  if (iter == registeredOperations.end()) {
    throw std::runtime_error("Operation not registered: " + name);
  }
  return *iter->second;
}
} // namespace cudaq
