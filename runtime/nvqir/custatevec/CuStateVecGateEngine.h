/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuStateVecConfig.h"
#include "CuStateVecState.h"
#include "CuStateVecTasks.h"

#include <cstddef>
#include <memory>
#include <span>

namespace cudaq::cusv {

/// @brief Abstract gate-application strategy for a `cuStateVecEx` state.
///
/// Implementations either apply each task immediately or enqueue tasks into a
/// fused state-vector updater.
template <typename Scalar>
struct GateEngine {
  virtual ~GateEngine() = default;
  virtual void apply(CuStateVecState<Scalar> &state,
                     const SimulationTask<Scalar> &task,
                     std::span<const double> randomNumbers = {}) = 0;
  virtual void flush(CuStateVecState<Scalar> &state,
                     std::span<const double> randomNumbers = {}) = 0;
};

/// @brief Applies every simulation task immediately.
///
/// This strategy provides the gate-by-gate execution path with no fusion.
template <typename Scalar>
struct DirectGateEngine : public GateEngine<Scalar> {
  void apply(CuStateVecState<Scalar> &state, const SimulationTask<Scalar> &task,
             std::span<const double> randomNumbers = {}) override;
  void flush(CuStateVecState<Scalar> &, std::span<const double> = {}) override {
  }
};

/// @brief Queues simulation tasks in a `cuStateVecEx` updater.
///
/// The engine owns the updater configuration and flushes fused tasks with the
/// random numbers required by any queued noise channels.
template <typename Scalar>
class FusedGateEngine : public GateEngine<Scalar> {
public:
  explicit FusedGateEngine(const CuStateVecConfig &config);
  ~FusedGateEngine() override;
  FusedGateEngine(const FusedGateEngine &) = delete;
  FusedGateEngine &operator=(const FusedGateEngine &) = delete;

  void apply(CuStateVecState<Scalar> &state, const SimulationTask<Scalar> &task,
             std::span<const double> randomNumbers = {}) override;
  void flush(CuStateVecState<Scalar> &state,
             std::span<const double> randomNumbers = {}) override;

private:
  custatevecExDictionaryDescriptor_t m_configuration = nullptr;
  custatevecExSVUpdaterDescriptor_t m_updater = nullptr;
  std::size_t m_pendingTaskCount = 0;
};

template <typename Scalar>
std::unique_ptr<GateEngine<Scalar>>
createGateEngine(const CuStateVecConfig &config);

extern template struct DirectGateEngine<float>;
extern template struct DirectGateEngine<double>;
extern template class FusedGateEngine<float>;
extern template class FusedGateEngine<double>;
extern template std::unique_ptr<GateEngine<float>>
createGateEngine(const CuStateVecConfig &);
extern template std::unique_ptr<GateEngine<double>>
createGateEngine(const CuStateVecConfig &);

} // namespace cudaq::cusv
