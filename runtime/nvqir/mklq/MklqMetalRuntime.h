/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <string>

namespace nvqir::mklq {

inline constexpr std::size_t marginalProbabilityThreadsPerThreadgroup = 256;

struct MetalDeviceInfo {
  bool available = false;
  std::string name;
  bool lowPower = false;
  bool headless = false;
  bool removable = false;
};

MetalDeviceInfo queryMetalDevice();

class MetalStateVectorExecutor {
public:
  MetalStateVectorExecutor();
  ~MetalStateVectorExecutor();

  MetalStateVectorExecutor(const MetalStateVectorExecutor &) = delete;
  MetalStateVectorExecutor &
  operator=(const MetalStateVectorExecutor &) = delete;
  MetalStateVectorExecutor(MetalStateVectorExecutor &&) noexcept;
  MetalStateVectorExecutor &operator=(MetalStateVectorExecutor &&) noexcept;

  bool available() const;
  MetalDeviceInfo deviceInfo() const;
  std::string lastError() const;

  bool applySingleQubitGate(std::complex<double> *state, std::size_t stateSize,
                            const std::complex<double> *matrix,
                            const std::size_t *controlQubits,
                            std::size_t controlCount, std::size_t targetQubit);
  bool applyTwoQubitGate(std::complex<double> *state, std::size_t stateSize,
                         const std::complex<double> *matrix,
                         const std::size_t *controlQubits,
                         std::size_t controlCount,
                         const std::size_t *targetQubits);
  bool fillFullRegisterProbabilities(const std::complex<double> *state,
                                     std::size_t stateSize,
                                     double *probabilities,
                                     std::size_t probabilityCount);
  bool uploadState(const std::complex<double> *state, std::size_t stateSize);
  bool downloadState(std::complex<double> *state, std::size_t stateSize);
  void releaseResidentState();
  bool hasResidentState(std::size_t stateSize) const;
  bool applyResidentSingleQubitGate(const std::complex<double> *matrix,
                                    const std::size_t *controlQubits,
                                    std::size_t controlCount,
                                    std::size_t targetQubit);
  bool applyResidentTwoQubitGate(const std::complex<double> *matrix,
                                 const std::size_t *controlQubits,
                                 std::size_t controlCount,
                                 const std::size_t *targetQubits);
  bool fillResidentFullRegisterProbabilities(double *probabilities,
                                             std::size_t probabilityCount);
  bool fillResidentMarginalProbabilities(const std::size_t *qubits,
                                         std::size_t qubitCount,
                                         double *probabilities,
                                         std::size_t probabilityCount);
  bool computeResidentQubitProbability(std::size_t qubit,
                                       double *probabilityOne);
  bool collapseResidentQubit(std::size_t qubit, bool result,
                             double branchProbability);

  std::size_t singleQubitGateApplications() const;
  std::size_t twoQubitGateApplications() const;
  std::size_t probabilityFillApplications() const;
  std::size_t marginalProbabilityApplications() const;
  std::size_t measurementProbabilityApplications() const;
  std::size_t measurementProbabilityReductionApplications() const;
  std::size_t measurementCollapseApplications() const;
  std::size_t residentStateUploads() const;
  std::size_t residentStateDownloads() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace nvqir::mklq
