/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file CpuRoceChannelTester.cpp
/// @brief GoogleTest fixture for the cpu_roce DeviceCallChannel, built as the
///        standalone test_cpu_roce_device_call binary (kept separate from
///        test_device_call_dispatch so it can be compiled on its own).
///
/// Exercises the full RDMA device_call round-trip:
///   CpuRoceChannel (caller, this process)  <-- RoCE -->  cpu_roce_test_daemon
/// over a NIC loopback cable.  SetUp() spawns the daemon as a subprocess, reads
/// its `CPU_ROCE_DAEMON_READY port=...` rendezvous line, and initializes the
/// device_call runtime with `--cudaq-device-call=cpu_roce` plus the channel's
/// connection arguments.  The whole fixture GTEST_SKIPs unless the test
/// topology is provided via environment variables, so the binary stays green on
/// CI hosts without an RDMA NIC / loopback cable:
///
///   CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE   IB device for the caller (e.g.
///   mlx5_0) CUDAQ_CPU_ROCE_TEST_CHANNEL_IP       RoCE IPv4 for the caller
///   (e.g. 10.0.0.1) CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE    IB device for the
///   daemon (e.g. mlx5_1) CUDAQ_CPU_ROCE_TEST_DAEMON_IP        RoCE IPv4 for
///   the daemon (e.g. 10.0.0.2) CUDAQ_CPU_ROCE_TEST_DAEMON_PATH      (optional)
///   path to cpu_roce_test_daemon
///
/// CPU_ROCE_DAEMON_PATH (compile definition) supplies the daemon path by
/// default; the env var overrides it.

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <gtest/gtest.h>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace cudaq_internal::device_call {
void initializeDeviceCallRuntime(int argc, char **argv);
void finalizeDeviceCallRuntime();
} // namespace cudaq_internal::device_call

extern "C" std::int32_t __cudaq_device_call_acquire_realtime_frame(
    std::uint32_t deviceId, std::uint32_t functionId,
    std::uint64_t requestBytes, std::uint64_t responseCapacity,
    void **frameHandle, void **requestPayload, void **responsePayload);
extern "C" std::int32_t
__cudaq_device_call_dispatch_realtime_frame(void *frameHandle,
                                            std::uint64_t *responseBytes);
extern "C" void
__cudaq_device_call_safely_release_realtime_frame(void *frameHandle);

namespace {

using cudaq::realtime::fnv1a_hash;

constexpr std::uint32_t AddThemFunctionId = fnv1a_hash("addThem");
constexpr std::uint32_t NoopFunctionId = fnv1a_hash("noop");
constexpr std::uint32_t AccumulateFunctionId = fnv1a_hash("accumulate");
constexpr std::uint32_t SumFunctionId = fnv1a_hash("sum");
constexpr std::uint32_t Slots = 8;
constexpr std::uint64_t SlotSize = 256;

const char *envOrNull(const char *name) {
  const char *v = std::getenv(name);
  return (v && *v) ? v : nullptr;
}

std::string daemonBinaryPath() {
  if (const char *v = envOrNull("CUDAQ_CPU_ROCE_TEST_DAEMON_PATH"))
    return v;
#ifdef CPU_ROCE_DAEMON_PATH
  return CPU_ROCE_DAEMON_PATH;
#else
  return "cpu_roce_test_daemon";
#endif
}

// Manages the cpu_roce_test_daemon subprocess + its stdout pipe.
class DaemonProcess {
public:
  // Spawn the daemon; block until it prints `CPU_ROCE_DAEMON_READY port=...`
  // (or the deadline elapses).  Returns the rendezvous port, or 0 on failure.
  std::uint16_t start(const std::string &device, const std::string &ip,
                      std::chrono::milliseconds timeout) {
    int fds[2];
    if (::pipe(fds) != 0)
      return 0;

    pid = ::fork();
    if (pid < 0) {
      ::close(fds[0]);
      ::close(fds[1]);
      return 0;
    }
    if (pid == 0) {
      // Child: stdout/stderr -> pipe, then exec the daemon.
      ::dup2(fds[1], STDOUT_FILENO);
      ::dup2(fds[1], STDERR_FILENO);
      ::close(fds[0]);
      ::close(fds[1]);
      const std::string path = daemonBinaryPath();
      const std::string devArg = "--device=" + device;
      const std::string ipArg = "--local-ip=" + ip;
      const std::string portArg = "--rendezvous-port=0";
      const std::string pagesArg = "--num-pages=" + std::to_string(Slots);
      const std::string pageSzArg = "--page-size=" + std::to_string(SlotSize);
      const std::string timeoutArg = "--timeout=120";
      char *argv[] = {const_cast<char *>(path.c_str()),
                      const_cast<char *>(devArg.c_str()),
                      const_cast<char *>(ipArg.c_str()),
                      const_cast<char *>(portArg.c_str()),
                      const_cast<char *>(pagesArg.c_str()),
                      const_cast<char *>(pageSzArg.c_str()),
                      const_cast<char *>(timeoutArg.c_str()),
                      nullptr};
      ::execv(path.c_str(), argv);
      ::_exit(127); // exec failed
    }

    // Parent: read the pipe until the READY token appears.
    ::close(fds[1]);
    readFd = fds[0];
    const std::uint16_t port = waitForReady(timeout);
    return port;
  }

  void stop() {
    if (pid > 0) {
      ::kill(pid, SIGTERM);
      int status = 0;
      // Give it a moment, then force-kill if needed.
      for (int i = 0; i < 50 && ::waitpid(pid, &status, WNOHANG) == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (::waitpid(pid, &status, WNOHANG) == 0) {
        ::kill(pid, SIGKILL);
        ::waitpid(pid, &status, 0);
      }
      pid = -1;
    }
    if (readFd >= 0) {
      ::close(readFd);
      readFd = -1;
    }
  }

  ~DaemonProcess() { stop(); }
  pid_t processId() const { return pid; }

private:
  std::uint16_t waitForReady(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string acc;
    char buf[512];
    while (std::chrono::steady_clock::now() < deadline) {
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              deadline - std::chrono::steady_clock::now())
              .count();
      pollfd pfd{readFd, POLLIN, 0};
      const int pr = ::poll(&pfd, 1, static_cast<int>(remaining));
      if (pr <= 0)
        continue;
      const ssize_t n = ::read(readFd, buf, sizeof(buf));
      if (n <= 0)
        break;
      acc.append(buf, static_cast<std::size_t>(n));
      const auto pos = acc.find("CPU_ROCE_DAEMON_READY");
      if (pos != std::string::npos) {
        const auto portPos = acc.find("port=", pos);
        if (portPos != std::string::npos)
          return static_cast<std::uint16_t>(
              std::strtoul(acc.c_str() + portPos + 5, nullptr, 10));
      }
    }
    return 0;
  }

  pid_t pid = -1;
  int readFd = -1;
};

class CpuRoceDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *chanDev = envOrNull("CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE");
    const char *chanIp = envOrNull("CUDAQ_CPU_ROCE_TEST_CHANNEL_IP");
    const char *daemonDev = envOrNull("CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE");
    const char *daemonIp = envOrNull("CUDAQ_CPU_ROCE_TEST_DAEMON_IP");
    if (!chanDev || !chanIp || !daemonDev || !daemonIp)
      GTEST_SKIP() << "cpu_roce test topology not configured (set "
                      "CUDAQ_CPU_ROCE_TEST_{CHANNEL,DAEMON}_{DEVICE,IP})";

    daemon = std::make_unique<DaemonProcess>();
    rendezvousPort =
        daemon->start(daemonDev, daemonIp, std::chrono::milliseconds(15000));
    ASSERT_NE(0, rendezvousPort) << "cpu_roce_test_daemon did not become ready";

    const std::string slots =
        "--cudaq-device-call-slots=" + std::to_string(Slots);
    const std::string slotSize =
        "--cudaq-device-call-slot-size=" + std::to_string(SlotSize);
    const std::string timeout = "--cudaq-device-call-timeout-ms=10000";
    const std::string ibArg = std::string("ib-device=") + chanDev;
    const std::string ipArg = std::string("local-ip=") + chanIp;
    const std::string rhost = "rendezvous-host=127.0.0.1";
    const std::string rport =
        "rendezvous-port=" + std::to_string(rendezvousPort);

    std::vector<std::string> args = {"test_device_call_dispatch",
                                     "--cudaq-device-call=cpu_roce",
                                     slots,
                                     slotSize,
                                     timeout,
                                     ibArg,
                                     ipArg,
                                     rhost,
                                     rport};
    std::vector<char *> argv;
    for (auto &a : args)
      argv.push_back(const_cast<char *>(a.c_str()));
    ASSERT_NO_THROW(cudaq_internal::device_call::initializeDeviceCallRuntime(
        static_cast<int>(argv.size()), argv.data()));
    runtimeUp = true;
  }

  void TearDown() override {
    if (runtimeUp) {
      cudaq_internal::device_call::finalizeDeviceCallRuntime();
      runtimeUp = false;
    }
    if (daemon)
      daemon->stop();
  }

  // Single synchronous addThem(a, b) -> a + b round-trip.  int32 args/result
  // match the CUDA-Q compiler ABI for `int addThem(int, int)`.
  std::int32_t addThem(std::int32_t a, std::int32_t b) {
    void *frame = nullptr, *req = nullptr, *resp = nullptr;
    EXPECT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                     0, AddThemFunctionId, 2 * sizeof(std::int32_t),
                     sizeof(std::int32_t), &frame, &req, &resp));
    EXPECT_NE(nullptr, frame);
    auto *args = static_cast<std::int32_t *>(req);
    args[0] = a;
    args[1] = b;
    std::uint64_t responseLen = 0;
    const std::int32_t status =
        __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen);
    std::int32_t out = 0;
    if (status == 0) {
      EXPECT_EQ(sizeof(std::int32_t), responseLen);
      out = *static_cast<std::int32_t *>(resp);
    }
    __cudaq_device_call_safely_release_realtime_frame(frame);
    EXPECT_EQ(0, status);
    return out;
  }

  std::unique_ptr<DaemonProcess> daemon;
  std::uint16_t rendezvousPort = 0;
  bool runtimeUp = false;
};

TEST_F(CpuRoceDispatchTest, DispatchesAddThemSynchronously) {
  for (int i = 0; i < 1000; ++i)
    EXPECT_EQ(i + 7, addThem(i, 7));
}

TEST_F(CpuRoceDispatchTest, DispatchesNoopFireAndForget) {
  for (int i = 0; i < 1000; ++i) {
    void *frame = nullptr, *req = nullptr, *resp = nullptr;
    ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                     0, NoopFunctionId, 0, 0, &frame, &req, &resp));
    std::uint64_t responseLen = 123;
    ASSERT_EQ(0,
              __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
    EXPECT_EQ(0u, responseLen);
    __cudaq_device_call_safely_release_realtime_frame(frame);
  }
}

TEST_F(CpuRoceDispatchTest, FireAndForgetDoesNotPoisonLaterResponses) {
  // Regression test for the late-RX-response poisoning bug: the service Writes
  // a (zero-length) response for every fire-and-forget request, and that late
  // write must be drained before its slot is reused -- otherwise a subsequent
  // response-bearing call on the same slot reads the stale fire-and-forget
  // response (wrong request_id) and fails validation.  Fire many
  // fire-and-forget accumulate(i) with distinct payloads (forcing slot reuse
  // far past the ring depth), then query the running sum: any poisoned/dropped
  // accumulate makes the sum wrong or the sum() dispatch fail.
  constexpr int kN = 2000;
  std::int64_t expected = 0;
  for (int i = 0; i < kN; ++i) {
    void *frame = nullptr, *req = nullptr, *resp = nullptr;
    ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                     0, AccumulateFunctionId, sizeof(std::int64_t), 0, &frame,
                     &req, &resp));
    *static_cast<std::int64_t *>(req) = i;
    expected += i;
    std::uint64_t responseLen = 0;
    ASSERT_EQ(0,
              __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
    __cudaq_device_call_safely_release_realtime_frame(frame);
  }

  // Response-bearing sum() is dispatched after all the fire-and-forget
  // accumulates (the daemon processes its receive ring in order), so it
  // observes the full total.
  void *frame = nullptr, *req = nullptr, *resp = nullptr;
  ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(0, SumFunctionId, 0,
                                                          sizeof(std::int64_t),
                                                          &frame, &req, &resp));
  std::uint64_t responseLen = 0;
  ASSERT_EQ(0,
            __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
  EXPECT_EQ(sizeof(std::int64_t), responseLen);
  EXPECT_EQ(expected, *static_cast<std::int64_t *>(resp));
  __cudaq_device_call_safely_release_realtime_frame(frame);
}

TEST_F(CpuRoceDispatchTest, DispatchesAddThemConcurrentlyAcrossThreads) {
  // v1 serializes dispatch internally; this verifies correctness (and no slot
  // double-allocation / response cross-talk) under concurrent callers.
  constexpr int kThreads = 4;
  constexpr int kPerThread = 250;
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t)
    threads.emplace_back([&, t] {
      for (int i = 0; i < kPerThread; ++i) {
        const std::int32_t a = t * 1000 + i, b = 5;
        if (addThem(a, b) != a + b)
          failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  for (auto &th : threads)
    th.join();
  EXPECT_EQ(0, failures.load());
}

TEST_F(CpuRoceDispatchTest, MeasuresDispatchLatency) {
  constexpr int kIters = 2000;
  // Warm-up (exclude first round-trips).
  for (int i = 0; i < 50; ++i)
    (void)addThem(i, 1);
  std::vector<double> us;
  us.reserve(kIters);
  for (int i = 0; i < kIters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(i + 3, addThem(i, 3));
    const auto t1 = std::chrono::steady_clock::now();
    us.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() /
        1000.0);
  }
  std::sort(us.begin(), us.end());
  const double p50 = us[us.size() / 2];
  const double p99 = us[(us.size() * 99) / 100];
  std::fprintf(stderr, "[cpu_roce] dispatch latency p50=%.2f us p99=%.2f us\n",
               p50, p99);
  SUCCEED();
}

TEST_F(CpuRoceDispatchTest, RecoversCleanlyWhenServiceRestarts) {
  // A few good round-trips, then kill the daemon and confirm the next
  // response-bearing dispatch fails (timeout / error) rather than hanging.
  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(i, addThem(i, 0));

  daemon->stop();

  void *frame = nullptr, *req = nullptr, *resp = nullptr;
  ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                   0, AddThemFunctionId, 2 * sizeof(std::int32_t),
                   sizeof(std::int32_t), &frame, &req, &resp));
  auto *args = static_cast<std::int32_t *>(req);
  args[0] = 1;
  args[1] = 2;
  std::uint64_t responseLen = 0;
  const std::int32_t status =
      __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen);
  __cudaq_device_call_safely_release_realtime_frame(frame);
  EXPECT_NE(0, status) << "dispatch should fail once the service is gone";
}

} // namespace
