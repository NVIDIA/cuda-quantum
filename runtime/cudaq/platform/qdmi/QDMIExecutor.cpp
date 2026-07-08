/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "QDMIServerHelper.h"

#include "common/Executor.h"
#include "common/Future.h"
#include "fomac/FoMaC.hpp"
#include "cudaq/runtime/logger/logger.h"

#include <future>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
cudaq::CountsDictionary
toCountsDictionary(const std::map<std::string, std::size_t> &counts) {
  cudaq::CountsDictionary result;
  result.reserve(counts.size());
  for (const auto &[bits, count] : counts)
    result[bits] = count;
  return result;
}
} // namespace

namespace cudaq {

class QDMIExecutor : public Executor {
public:
  detail::future execute(std::vector<KernelExecution> &codesToExecute,
                         detail::ExecutionContextType execType,
                         std::vector<char> *) override {
    if (execType == detail::ExecutionContextType::run)
      throw std::runtime_error("QDMI backend does not support cudaq::run.");

    serverHelper->setShots(shots);
    auto *helper = dynamic_cast<QDMIServerHelper *>(serverHelper);
    if (!helper)
      throw std::runtime_error("QDMI executor requires QDMIServerHelper.");

    const auto device = helper->getDevice();
    const auto format = helper->getProgramFormat();
    const auto shotCount = shots;
    auto codes = codesToExecute;

    return std::async(std::launch::async, [codes = std::move(codes), device,
                                           format, shotCount, execType]() {
      sample_result result;
      const auto qdmiDevice = fomac::Session::Device::fromQDMIDevice(device);

      for (const auto &code : codes) {
        auto job = qdmiDevice.submitJob(code.code, format, shotCount);
        if (!job.wait())
          throw std::runtime_error("QDMI job timed out.");

        const auto status = job.check();
        if (status == QDMI_JOB_STATUS_FAILED)
          throw std::runtime_error("QDMI job failed.");
        if (status == QDMI_JOB_STATUS_CANCELED)
          throw std::runtime_error("QDMI job was canceled.");
        if (status != QDMI_JOB_STATUS_DONE)
          throw std::runtime_error("QDMI job did not complete.");

        const bool observe = execType == detail::ExecutionContextType::observe;
        const auto registerName = observe ? code.name : GlobalRegisterName;
        ExecutionResult executionResult(toCountsDictionary(job.getCounts()),
                                        registerName);
        try {
          executionResult.sequentialData = job.getShots();
        } catch (const std::exception &e) {
          CUDAQ_DBG("QDMI shot data is unavailable: {}", e.what());
        }

        sample_result jobResult(std::move(executionResult));
        if (!code.mapping_reorder_idx.empty())
          jobResult.reorder(code.mapping_reorder_idx, registerName);
        result += jobResult;
      }

      return result;
    });
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::Executor, cudaq::QDMIExecutor, qdmi)
