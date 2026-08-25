/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/runtime/logger/tracer.h"

namespace cudaq {

class SpdlogTraceBackend : public TraceBackend {
public:
  SpdlogTraceBackend();
  ~SpdlogTraceBackend() override;

  void onBegin(const TraceEvent &) override;
  void onEnd(const TraceEvent &e, uint64_t durUs) override;
};

} // namespace cudaq
