/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 ******************************************************************************/

// clang-format off
// REQUIRES: iqm-frequencies-mock
// RUN: nvq++ --target iqm %s -o %t
// RUN: python3 %S/Inputs/frequencies_mock.py %cudaq_src_dir %t
// clang-format on

#include <cassert>
#include <cmath>
#include <cudaq.h>
#include <iostream>

struct prepare {
  void operator()(cudaq::qarray<2> &q, double theta) __qpu__ {
    x(q[0]);
    ry(theta, q[1]);
  }
};

struct bell {
  void operator()(cudaq::qarray<2> &q) __qpu__ {
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  }
};

struct legacy {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
  }
};

struct direct {
  void operator()(prepare preparation, std::size_t shots,
                  double theta) __qpu__ {
    cudaq::emit(cudaq::frequencies(preparation, shots, theta));
  }
};

struct extra_output {
  void operator()(prepare preparation) __qpu__ {
    cudaq::emit(std::int64_t{42});
    cudaq::emit(cudaq::frequencies(preparation, 11, 0.0));
  }
};

int main() {
  auto ordered = cudaq::sample_vnext(37, prepare{}, 0.0);
  assert(ordered.count("10") == 37 && ordered.get_total_shots() == 37);
  auto correlated = cudaq::sample_vnext(200, bell{});
  assert(correlated.count("00") == 100 && correlated.count("11") == 100);
  assert(correlated.get_total_shots() == 200);
  auto records =
      cudaq::launch(direct{}, prepare{}, std::size_t{9}, std::acos(-1.0));
  assert(records.size() == 1);
  assert(std::get<cudaq::sample_result>(records[0]).count("11") == 9);
  assert(cudaq::sample_vnext(0, prepare{}, 0.0).get_total_shots() == 0);
  bool rejected = false;
  try {
    cudaq::launch(extra_output{}, prepare{});
  } catch (const std::exception &error) {
    rejected =
        std::string(error.what()).find("exactly one top-level frequencies") !=
        std::string::npos;
  }
  assert(rejected);
  // Other kernels in this translation unit retain their existing sample path.
  assert(cudaq::sample(17, legacy{}).count("1") == 17);
  std::cout << "ServerHelper frequencies prototype passed\n";
}
