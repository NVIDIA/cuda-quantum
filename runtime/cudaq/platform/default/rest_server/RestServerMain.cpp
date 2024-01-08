/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "RemoteKernelExecutor.h"
#include "cudaq.h"

int main(int argc, char **argv) {
  constexpr int DEFAULT_PORT = 3030;
  const int port = [&]() {
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "-p" || std::string(argv[i]) == "-port" ||
          std::string(argv[i]) == "--port")
        return atoi(argv[i + 1]);

    return DEFAULT_PORT;
  }();

  cudaq::mpi::initialize();
  auto restServer = cudaq::registry::get<cudaq::RemoteRuntimeServer>("rest");
  restServer->init({{"port", std::to_string(port)}});
  restServer->start();
  cudaq::mpi::finalize();
}
