/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 *reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/SampleResult.h"
#include "cudaq/runtime/logger/logger.h"

#include <mutex>

/// @brief The default explanatory detail appended to the named-measurement
/// deprecation warning. Describes the standard sampling-mode behavior.
static constexpr std::string_view defaultNamedMeasurementsWarningDetail =
    "invoked in sampling mode. Support for sub-registers in `sample_result` is "
    "deprecated and will be removed in a future release. Use `run` to retrieve "
    "individual measurement results.";

void cudaq::emitNamedMeasurementsWarning(const std::string &kernelName,
                                         std::string_view detail) {
  if (detail.empty())
    detail = defaultNamedMeasurementsWarningDetail;

  static std::once_flag warned;
  std::call_once(warned, [&] {
    CUDAQ_WARN("Kernel \"{}\" uses named measurement results but is {}",
               kernelName, detail);
  });
}
