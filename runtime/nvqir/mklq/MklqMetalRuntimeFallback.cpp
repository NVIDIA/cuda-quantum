/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "MklqMetalRuntime.h"

namespace nvqir::mklq {

MetalDeviceInfo queryMetalDevice() { return {}; }

struct MetalStateVectorExecutor::Impl {};

MetalStateVectorExecutor::MetalStateVectorExecutor()
    : impl(std::make_unique<Impl>()) {}

MetalStateVectorExecutor::~MetalStateVectorExecutor() = default;

MetalStateVectorExecutor::MetalStateVectorExecutor(
    MetalStateVectorExecutor &&) noexcept = default;

MetalStateVectorExecutor &MetalStateVectorExecutor::operator=(
    MetalStateVectorExecutor &&) noexcept = default;

bool MetalStateVectorExecutor::available() const { return false; }

MetalDeviceInfo MetalStateVectorExecutor::deviceInfo() const { return {}; }

std::string MetalStateVectorExecutor::lastError() const {
  return "Metal runtime is unavailable on this platform.";
}

bool MetalStateVectorExecutor::applySingleQubitGate(
    std::complex<double> *, std::size_t, const std::complex<double> *,
    const std::size_t *, std::size_t, std::size_t) {
  return false;
}

bool MetalStateVectorExecutor::applyTwoQubitGate(
    std::complex<double> *, std::size_t, const std::complex<double> *,
    const std::size_t *, std::size_t, const std::size_t *) {
  return false;
}

bool MetalStateVectorExecutor::fillFullRegisterProbabilities(
    const std::complex<double> *, std::size_t, double *, std::size_t) {
  return false;
}

bool MetalStateVectorExecutor::uploadState(const std::complex<double> *,
                                           std::size_t) {
  return false;
}

bool MetalStateVectorExecutor::downloadState(std::complex<double> *,
                                             std::size_t) {
  return false;
}

void MetalStateVectorExecutor::releaseResidentState() {}

bool MetalStateVectorExecutor::hasResidentState(std::size_t) const {
  return false;
}

bool MetalStateVectorExecutor::applyResidentSingleQubitGate(
    const std::complex<double> *, const std::size_t *, std::size_t,
    std::size_t) {
  return false;
}

bool MetalStateVectorExecutor::applyResidentTwoQubitGate(
    const std::complex<double> *, const std::size_t *, std::size_t,
    const std::size_t *) {
  return false;
}

bool MetalStateVectorExecutor::applyResidentThreeQubitGate(
    const std::complex<double> *, const std::size_t *, std::size_t,
    const std::size_t *) {
  return false;
}

bool MetalStateVectorExecutor::fillResidentFullRegisterProbabilities(
    double *, std::size_t) {
  return false;
}

bool MetalStateVectorExecutor::fillResidentMarginalProbabilities(
    const std::size_t *, std::size_t, double *, std::size_t) {
  return false;
}

bool MetalStateVectorExecutor::computeResidentQubitProbability(std::size_t,
                                                               double *) {
  return false;
}

bool MetalStateVectorExecutor::collapseResidentQubit(std::size_t, bool,
                                                     double) {
  return false;
}

std::size_t MetalStateVectorExecutor::singleQubitGateApplications() const {
  return 0;
}

std::size_t MetalStateVectorExecutor::twoQubitGateApplications() const {
  return 0;
}

std::size_t MetalStateVectorExecutor::threeQubitGateApplications() const {
  return 0;
}

std::size_t MetalStateVectorExecutor::probabilityFillApplications() const {
  return 0;
}

std::size_t MetalStateVectorExecutor::marginalProbabilityApplications() const {
  return 0;
}

std::size_t MetalStateVectorExecutor::measurementProbabilityApplications()
    const {
  return 0;
}

std::size_t
MetalStateVectorExecutor::measurementProbabilityReductionApplications() const {
  return 0;
}

std::size_t MetalStateVectorExecutor::measurementCollapseApplications() const {
  return 0;
}

std::size_t MetalStateVectorExecutor::residentStateUploads() const { return 0; }

std::size_t MetalStateVectorExecutor::residentStateDownloads() const {
  return 0;
}

} // namespace nvqir::mklq
