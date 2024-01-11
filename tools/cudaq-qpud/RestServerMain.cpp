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

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "CUDA Quantum REST server\n");
  if (cudaq::mpi::available())
    cudaq::mpi::initialize();
  auto restServer = cudaq::registry::get<cudaq::RemoteRuntimeServer>("rest");
  restServer->init({{"port", std::to_string(port)}});
  restServer->start();
  if (cudaq::mpi::available())
    cudaq::mpi::finalize();
}
