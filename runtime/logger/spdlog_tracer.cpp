/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/runtime/logger/spdlog_tracer.h"
#include "cudaq/runtime/logger/cudaq_fmt.h"
#include "cudaq/runtime/logger/logger.h"

#include <string>

namespace cudaq {

SpdlogTraceBackend::SpdlogTraceBackend() = default;
SpdlogTraceBackend::~SpdlogTraceBackend() = default;

void SpdlogTraceBackend::onBegin(const TraceEvent &) {}

void SpdlogTraceBackend::onEnd(const TraceEvent &e, uint64_t durUs) {
  const bool tagFound = (e.tag != 0) && cudaq::isTimingTagEnabled(e.tag);
  if (!tagFound && !details::should_log(details::LogLevel::trace))
    return;

  double duration = static_cast<double>(durUs) / 1000.0;
  std::string tagStr = tagFound ? cudaq_fmt::format("[tag={}] ", e.tag) : "";
  std::string sourceInfo =
      e.ctx.fileName
          ? cudaq_fmt::format("[{}:{}] ",
                              details::pathToFileName(e.ctx.fileName),
                              e.ctx.lineNo)
          : "";
  auto str = cudaq_fmt::format(
      "{}{}{}{} executed in {} ms.{}",
      e.depth > 0 ? std::string(e.depth, '-') + " " : "", tagStr, sourceInfo,
      std::string(e.name), duration, std::string(e.args));
  if (tagFound)
    cudaq::log(str);
  else
    details::trace(str);
}

} // namespace cudaq
