/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#include "common/ServerHelper.h"

#include <optional>
#include <qdmi/client.h>
#include <utility>
#include <vector>

namespace cudaq {

class QDMIServerHelper : public ServerHelper {
public:
  ~QDMIServerHelper() override;

  const std::string name() const override { return "qdmi"; }

  void initialize(BackendConfig config) override;
  RestHeaders getHeaders() override { return {}; }

  ServerJobPayload createJob(std::vector<KernelExecution> &) override;
  std::string extractJobId(ServerMessage &) override;
  std::string constructGetJobPath(ServerMessage &) override;
  std::string constructGetJobPath(std::string &) override;
  bool jobIsDone(ServerMessage &) override;
  sample_result processResults(ServerMessage &, std::string &) override;

  [[nodiscard]] QDMI_Device getDevice() const { return device; }
  [[nodiscard]] QDMI_Program_Format getProgramFormat() const {
    return programFormat;
  }
  [[nodiscard]] std::size_t getQubitCount() const { return qubitCount; }
  [[nodiscard]] const std::optional<
      std::vector<std::pair<std::size_t, std::size_t>>> &
  getConnectivity() const {
    return connectivity;
  }

private:
  QDMI_Device device = nullptr;
  QDMI_Program_Format programFormat = QDMI_PROGRAM_FORMAT_QASM2;
  std::size_t qubitCount = 0;
  std::optional<std::vector<std::pair<std::size_t, std::size_t>>> connectivity;
};

} // namespace cudaq
