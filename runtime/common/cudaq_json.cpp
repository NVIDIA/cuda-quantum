/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_json.h"
#include "nlohmann/json.hpp"

namespace cudaq {

cudaq_json::cudaq_json(const cudaq_json &other)
    : impl(std::make_unique<nlohmann::json>(*other.impl)) {}

cudaq_json::cudaq_json(cudaq_json &&) noexcept = default;

cudaq_json &cudaq_json::operator=(const cudaq_json &other) {
  return *this = cudaq_json(other);
}

cudaq_json &cudaq_json::operator=(cudaq_json &&) noexcept = default;

cudaq_json::~cudaq_json() = default;

cudaq_json::cudaq_json(const nlohmann::json &j)
    : impl(std::make_unique<nlohmann::json>(j)) {}

cudaq_json::cudaq_json(nlohmann::json &&j)
    : impl(std::make_unique<nlohmann::json>(std::move(j))) {}

nlohmann::json &cudaq_json::get() { return *impl; }
const nlohmann::json &cudaq_json::get() const { return *impl; }

nlohmann::json &cudaq_json::operator*() { return *impl; }
const nlohmann::json &cudaq_json::operator*() const { return *impl; }

nlohmann::json *cudaq_json::operator->() { return impl.get(); }
const nlohmann::json *cudaq_json::operator->() const { return impl.get(); }

} // namespace cudaq
