/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/registry/rpc_builder.h"
#include "cudaq/nvqlink/daemon/daemon.h"

using namespace cudaq::nvqlink;

void RPCBuilder::register_all(Daemon &daemon) const {
  for (const auto &metadata : entries_) {
    daemon.register_function(metadata);
  }
}
