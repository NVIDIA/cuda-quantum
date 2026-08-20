/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "result.h"
#include <utility>

struct cudaq::detail::EstimateResultImpl {
  EstimateResultImpl() = default;
  EstimateResultImpl(Resources counts, cudaq_json annotations)
      : counts(std::move(counts)), annotations(std::move(annotations)) {}

  Resources counts;
  cudaq_json annotations;
};

cudaq::estimate_result::estimate_result(Resources counts,
                                        cudaq_json annotations) {
  impl = std::make_unique<detail::EstimateResultImpl>(std::move(counts),
                                                      std::move(annotations));
}

cudaq::estimate_result::estimate_result() {
  impl = std::make_unique<detail::EstimateResultImpl>();
}

cudaq::estimate_result::estimate_result(estimate_result &&other) noexcept {
  impl = std::move(other.impl);
}

cudaq::estimate_result &
cudaq::estimate_result::operator=(estimate_result &&other) {
  if (this != &other) {
    impl = std::move(other.impl);
  }
  return *this;
}

cudaq::estimate_result::estimate_result(const estimate_result &other) {
  impl = std::make_unique<detail::EstimateResultImpl>(*other.impl);
}

cudaq::estimate_result &
cudaq::estimate_result::operator=(const estimate_result &other) {
  if (this != &other) {
    auto temp = estimate_result(other);
    std::swap(impl, temp.impl);
  }
  return *this;
}

cudaq::estimate_result::~estimate_result() = default;

const cudaq::Resources &cudaq::estimate_result::get_resources() const {
  return impl->counts;
}

const cudaq::cudaq_json &cudaq::estimate_result::get_annotations() const {
  return impl->annotations;
}

cudaq::cudaq_json &cudaq::estimate_result::get_annotations() {
  return impl->annotations;
}
