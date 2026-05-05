/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

// expected-note@* 0+ {{}}

struct BoundaryDirectParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle h) __qpu__ { (void)h; }
};

struct BoundaryDirectReturn {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  cudaq::measure_handle operator()() __qpu__ {
    cudaq::qubit q;
    return mz(q);
  }
};

struct BoundaryVectorParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::vector<cudaq::measure_handle> h) __qpu__ { (void)h; }
};

struct BoundaryTupleParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::tuple<int, cudaq::measure_handle> h) __qpu__ {
    (void)h;
  }
};

struct BoundaryPairParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::pair<bool, cudaq::measure_handle> h) __qpu__ {
    (void)h;
  }
};

struct BoundaryPointerParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle *h) __qpu__ { (void)h; }
};

struct BoundaryReferenceParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::measure_handle &h) __qpu__ { (void)h; }
};

struct MeasureHandleHolder {
  cudaq::measure_handle h;
};

struct BoundaryAggregateParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(MeasureHandleHolder s) __qpu__ { (void)s; }
};

struct BoundaryPairOfVectorParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::pair<int, std::vector<cudaq::measure_handle>> p) __qpu__ {
    (void)p;
  }
};

struct BoundaryFunctionTypeParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(std::function<void(cudaq::measure_handle)> f) __qpu__ {
    (void)f;
  }
};

struct BoundaryQkernelParam {
  // expected-error@+1{{measurement handle cannot cross the host-device boundary; entry-point kernels must discriminate first}}
  void operator()(cudaq::qkernel<void(cudaq::measure_handle)> k) __qpu__ {
    (void)k;
  }
};

void sink(std::int64_t);

struct ToIntegerDirectRejected {
  void operator()() __qpu__ {
    cudaq::qvector q(8);
    // expected-error@+1{{`cudaq::to_integer` accepts `std::vector<bool>`; wrap measurement results with `cudaq::to_bools(...)` first}}
    sink(cudaq::to_integer(mz(q)));
  }
};

struct DirectUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle h;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h;
    (void)b;
  }
};

struct CopyConstructedUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle h;
    cudaq::measure_handle h2 = h;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h2;
    (void)b;
  }
};

struct CopyAssignedUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle h;
    cudaq::measure_handle h2;
    h2 = h;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h2;
    (void)b;
  }
};

struct ChainedAssignedUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle h1;
    cudaq::measure_handle h2;
    cudaq::measure_handle h3;
    h3 = h2 = h1;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h3;
    (void)b;
  }
};

struct ArrayElementUnbound {
  void operator()() __qpu__ {
    cudaq::measure_handle hs[2];
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = hs[1];
    (void)b;
  }
};

struct Holder {
  cudaq::measure_handle h;
};

struct AggregateMemberUnbound {
  void operator()() __qpu__ {
    Holder holder;
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = holder.h;
    (void)b;
  }
};

struct ConditionalStoreUnbound {
  bool operator()(bool cond) __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h;
    if (cond)
      h = mz(q);
    // expected-error@+1{{discriminating an unbound measure_handle}}
    bool b = h;
    return b;
  }
};
