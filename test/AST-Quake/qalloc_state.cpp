/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct Eins {
  std::vector<bool> operator()(cudaq::state *state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

struct Zwei {
  std::vector<bool> operator()(const cudaq::state *state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

struct Drei {
  std::vector<bool> operator()(cudaq::state &state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

struct Vier {
  std::vector<bool> operator()(const cudaq::state &state) __qpu__ {
    cudaq::qvector v(state);
    h(v);
    return mz(v);
  }
};

struct Fuenf {
  std::vector<bool> operator()(cudaq::state &&state) __qpu__ {
    cudaq::qvector v(std::move(state));
    h(v);
    return mz(v);
  }
};
