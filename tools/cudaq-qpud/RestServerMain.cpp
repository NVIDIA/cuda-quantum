/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Registry.h"
#include "common/RemoteKernelExecutor.h"
#include "llvm/Support/CommandLine.h"

// Declare CUDAQ MPI API that we need since we cannot compile with cudaq.h
// without RTTI (needed to link this tool against LLVMSupport).
namespace cudaq {
namespace mpi {
void initialize();
void finalize();
bool available();
} // namespace mpi
} // namespace cudaq

//===----------------------------------------------------------------------===//
// Command line options.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<int>
    port("port", llvm::cl::desc("TCP/IP port that the server will listen to."),
         llvm::cl::init(3030));
constexpr const char *DEFAULT_SERVER_IMPL = "rest";
static llvm::cl::opt<std::string> serverSubType(
    "type", llvm::cl::desc("HTTP server subtype handling incoming requests."),
    llvm::cl::init(DEFAULT_SERVER_IMPL));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "CUDA Quantum REST server\n");
  if (cudaq::mpi::available())
    cudaq::mpi::initialize();
  // Check the server type arg is valid.
  if (!cudaq::registry::isRegistered<cudaq::RemoteRuntimeServer>(serverSubType))
    throw std::runtime_error(
        std::string("[cudaq-qpud] Unknown server sub-type requested: ") +
        std::string(serverSubType));
  // Only log if this is not the default locally-hosted Rest server
  // implementation.
  if (serverSubType != std::string(DEFAULT_SERVER_IMPL))
    printf("[cudaq-qpud] Using server subtype: %s\n", serverSubType.c_str());
  auto restServer =
      cudaq::registry::get<cudaq::RemoteRuntimeServer>(serverSubType);
  restServer->init({{"port", std::to_string(port)}});
  restServer->start();
  if (cudaq::mpi::available())
    cudaq::mpi::finalize();
}
