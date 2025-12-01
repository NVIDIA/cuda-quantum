/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/nvqlink/device.h"

#include <cstring>

namespace cudaq::nvqlink {

class nv_simulation_device
    : public device_mixin<qcs_trait> {
  // Create the expected thunk args return structure
  struct thunk_ret {
    void *ptr;
    std::size_t i;
  };

protected:
  void *thunkArgs = nullptr;
  std::size_t (*argsCreator)(void **, void **);
  thunk_ret (*thunk)(void *, bool);

public:
  nv_simulation_device() : device_mixin() {}
  ~nv_simulation_device() {
    if (thunkArgs)
      std::free(thunkArgs);
  }

  void upload_program(const std::vector<std::byte>& program) override {
    // program here is a function pointer. but to what? Need to figure out
    // the entrypoint and how to call it with arguments.
    std::memcpy(&argsCreator, program.data(), sizeof(void *));
    std::memcpy(&thunk, program.data() + sizeof(void *), sizeof(void *));
  }

  void trigger(device_ptr &result, const std::vector<device_ptr> &args) override {
    std::vector<void *> concrete_args;
    concrete_args.resize(args.size());

    for (std::size_t i = 0; i < args.size(); i++)
      concrete_args[i] = reinterpret_cast<void *>(args[i].handle);

    auto thunkSize = argsCreator(concrete_args.data(), &thunkArgs);
    thunk(thunkArgs, false);
    if (result.is_nullptr())
      return;

    std::memcpy(reinterpret_cast<void *>(result.handle),
                reinterpret_cast<char *>(thunkArgs) + (thunkSize - result.size),
                result.size);
  }
};

} // namespace cudaq::nvqlink
