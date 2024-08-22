
/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/FmtCore.h"

#include "common/Logger.h"
#include "cudaq.h"
#include "nvqpp_config.h"

#include "OrcaExecutor.h"
#include "OrcaRemoteRESTQPU.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include "orca_qpu.h"

#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <regex>
#include <sys/socket.h>
#include <sys/types.h>

namespace cudaq::orca {
cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles,
                            std::vector<double> &ps_angles, int n_samples) {
  TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
                           n_samples};
  cudaq::ExecutionContext context("sample", n_samples);
  auto &platform = get_platform();
  platform.set_exec_ctx(&context, 0);
  cudaq::altLaunchKernel("orca_launch", nullptr, &parameters,
                         sizeof(TBIParameters), 0);

  return context.result;
}
cudaq::sample_result sample(std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            std::vector<double> &bs_angles, int n_samples) {
  std::vector<double> ps_angles = {};
  TBIParameters parameters{input_state, loop_lengths, bs_angles, ps_angles,
                           n_samples};
  cudaq::ExecutionContext context("sample", n_samples);
  auto &platform = get_platform();
  platform.set_exec_ctx(&context, 0);
  cudaq::altLaunchKernel("orca_launch", nullptr, &parameters,
                         sizeof(TBIParameters), 0);

  return context.result;
}
} // namespace cudaq::orca
