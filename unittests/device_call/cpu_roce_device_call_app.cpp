/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file cpu_roce_device_call_app.cpp
/// @brief Compiler-driven proof that cc.device_call lowers and runs over the
///        cpu_roce DeviceCallChannel.
///
/// This is the cpu_roce analogue of cudaq/test/NVQPP/device_call_realtime_*.cpp
/// (which prove the shared-memory / host-dispatch channels through the
/// compiler).  It compiles a real __qpu__ kernel with
///   nvq++ --target qpp-cpu -frealtime-lowering
/// so cudaq::device_call() is lowered to the realtime
/// acquire/marshal/dispatch/release ABI, then selects the cpu_roce channel at
/// runtime with --cudaq-device-call=cpu_roce.
///
/// Unlike the shared-memory / host-dispatch tests, NO service shim is
/// co-linked: cpu_roce is an external channel whose service lives in a separate
/// process (cpu_roce_test_daemon) reached over RDMA.  The device function
/// `addThem` is implemented by that daemon, keyed by fnv1a_hash("addThem").
///
/// The orchestration (network setup, daemon spawn, channel arguments) is done
/// by cpu_roce_device_call_test.sh --app; this file is only the caller app.
///
/// Expected output (checked by the script):
///   device_call int result = 42
///   device_call measured integer result = 42

#include <cudaq.h>

#include <cstdio>

// The real implementation lives in the remote service (cpu_roce_test_daemon's
// int32 addThem handler, keyed by fnv1a_hash("addThem")).  The device_call
// lowering routes the call there over RDMA; it never invokes this host symbol.
// But nvq++ still emits a link-time reference to the callee, so -- unlike the
// shared-memory / host-dispatch tests, which co-link a shim that *defines*
// addThem -- we must provide a local definition to satisfy the linker.  We make
// it a sentinel that returns a value the kernels can never sum to 42, so if
// dispatch ever fell back to this local stub instead of the remote daemon, the
// result check below would FAIL.  A passing run therefore proves the RDMA path.
extern "C" int addThem(int, int) { return -1000; }

__qpu__ int kernel(int a, int b) {
  return cudaq::device_call(0, addThem, a, b);
}

__qpu__ int measuredIntegerKernel() {
  cudaq::qvector q(3);
  x(q[0]);
  x(q[2]);
  auto measured = cudaq::to_integer(cudaq::to_bools(mz(q)));
  return cudaq::device_call(0, addThem, static_cast<int>(measured), 37);
}

int main(int argc, char **argv) {
  cudaq::realtime::initialize(argc, argv);

  auto results = cudaq::run(1, kernel, 19, 23);
  int value = results.front();
  std::printf("device_call int result = %d\n", value);

  auto measuredResults = cudaq::run(1, measuredIntegerKernel);
  int measuredValue = measuredResults.front();
  std::printf("device_call measured integer result = %d\n", measuredValue);

  cudaq::realtime::finalize();
  return value == 42 && measuredValue == 42 ? 0 : 1;
}
