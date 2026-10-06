/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file roce_probe.h
/// @brief Decide whether an RDMA test can run by probing the RoCE transport
///        itself, rather than by whether the environment names one.
///
/// A test names the two endpoints it needs (`roce_test_topology()`) and asks
/// `roce_test_gate()` whether to run, skip or fail.  That choice is what lets
/// one test serve both kinds of CI host: a runner with no RoCE NIC skips (the
/// default), while a host that is supposed to have one -- built with
/// CUDAQ_REALTIME_REQUIRE_ROCE_TESTS=ON, or run with
/// CUDAQ_REALTIME_REQUIRE_ROCE=1 -- turns the same skip into a failure.
///
/// The probe goes through `libibverbs` rather than `sysfs` on purpose: inside a
/// container without `/dev/infiniband`, `/sys/class/infiniband` still lists
/// the host's devices, but `ibv_get_device_list` sees none -- which is the
/// failure a test would actually hit.

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef CUDAQ_REALTIME_REQUIRE_ROCE_TESTS
#define CUDAQ_REALTIME_REQUIRE_ROCE_TESTS 0
#endif

namespace cudaq::realtime::testing {

/// One end of a RoCE link: an RDMA device and the IPv4 address its RoCEv2
/// GID must carry.
struct RoceEndpoint {
  std::string device;
  std::string ip;
};

/// The two endpoints of a caller/service RDMA test.
struct RoceTopology {
  RoceEndpoint caller;
  RoceEndpoint service;
};

namespace detail {
inline std::string env_or(const char *name, const char *fallback) {
  const char *v = std::getenv(name);
  return (v && *v) ? v : fallback;
}

inline bool has_ipv4_rocev2_gid(ibv_context *ctx, std::uint32_t port,
                                int gid_tbl_len, const in_addr &want) {
  // Scan the whole table: slots are recycled as addresses come and go, so an
  // empty slot is not the end of it.
  for (int i = 0; i < gid_tbl_len; ++i) {
    ibv_gid_entry entry{};
    if (ibv_query_gid_ex(ctx, port, static_cast<std::uint32_t>(i), &entry, 0) !=
        0)
      continue;
    if (entry.gid_type != IBV_GID_TYPE_ROCE_V2)
      continue;
    const std::uint8_t *raw = entry.gid.raw;
    static constexpr std::uint8_t kV4MappedPrefix[12] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    if (std::memcmp(raw, kV4MappedPrefix, sizeof(kV4MappedPrefix)) == 0 &&
        std::memcmp(raw + 12, &want.s_addr, 4) == 0)
      return true;
  }
  return false;
}
} // namespace detail

/// The topology the RDMA tests run on: the DGX Spark single-port loopback
/// (both ends on rocep1s0f0; the HCA loops the frames back internally).  Each
/// field can be overridden through CUDAQ_CPU_ROCE_TEST_{CHANNEL,DAEMON}_
/// {DEVICE,IP}, the same names test_cpu_roce_device_call uses.
inline RoceTopology roce_test_topology() {
  return RoceTopology{
      {detail::env_or("CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE", "rocep1s0f0"),
       detail::env_or("CUDAQ_CPU_ROCE_TEST_CHANNEL_IP", "10.0.0.1")},
      {detail::env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE", "rocep1s0f0"),
       detail::env_or("CUDAQ_CPU_ROCE_TEST_DAEMON_IP", "10.0.0.2")}};
}

/// Whether `endpoint` is usable on port 1: the device is visible and opens,
/// the port is an ACTIVE Ethernet (RoCE) port, and it has an IPv4-mapped
/// RoCEv2 GID for the endpoint's IP.  Returns an empty string when usable,
/// otherwise a one-line reason naming the first check that failed.
inline std::string probe_roce_endpoint(const RoceEndpoint &endpoint) {
  const std::string where = "'" + endpoint.device + "' (" + endpoint.ip + ")";
  in_addr want{};
  if (inet_pton(AF_INET, endpoint.ip.c_str(), &want) != 1)
    return where + ": not an IPv4 address";

  int num_devices = 0;
  ibv_device **devices = ibv_get_device_list(&num_devices);
  if (!devices || num_devices == 0) {
    if (devices)
      ibv_free_device_list(devices);
    return "no RDMA devices visible (in a container, pass "
           "--device=/dev/infiniband)";
  }
  ibv_device *picked = nullptr;
  for (int i = 0; i < num_devices; ++i)
    if (endpoint.device == ibv_get_device_name(devices[i]))
      picked = devices[i];
  ibv_context *ctx = picked ? ibv_open_device(picked) : nullptr;
  const int open_errno = errno;
  ibv_free_device_list(devices);
  if (!picked)
    return where + ": RDMA device not found among " +
           std::to_string(num_devices);
  if (!ctx)
    return where + ": cannot open RDMA device (" + std::strerror(open_errno) +
           ")";

  constexpr std::uint32_t kPort = 1;
  std::string why;
  ibv_port_attr port{};
  if (ibv_query_port(ctx, kPort, &port) != 0)
    why = where + ": cannot query port 1";
  else if (port.state != IBV_PORT_ACTIVE)
    why = where + ": port 1 is " + ibv_port_state_str(port.state) +
          ", not ACTIVE";
  else if (port.link_layer != IBV_LINK_LAYER_ETHERNET)
    why = where + ": port 1 is not a RoCE (Ethernet) port";
  else if (!detail::has_ipv4_rocev2_gid(ctx, kPort, port.gid_tbl_len, want))
    why = where + ": no RoCEv2 GID for this address on port 1 (is it "
                  "configured on the device's netdev?)";
  ibv_close_device(ctx);
  return why;
}

/// Both endpoints of `topology`, caller first; empty when both are usable.
inline std::string probe_roce_topology(const RoceTopology &topology) {
  std::string why = probe_roce_endpoint(topology.caller);
  if (why.empty())
    why = probe_roce_endpoint(topology.service);
  return why;
}

/// What an RDMA test should do before touching the transport.
struct RoceGate {
  enum Action { Run, Skip, Fail };
  Action action = Run;
  std::string reason; ///< Empty for Run.
};

/// Decide Run / Skip / Fail for `topology`.  A build configured with
/// CUDAQ_REALTIME_REQUIRE_ROCE_TESTS=ON requires RoCE, and nothing at run
/// time can lower that: CUDAQ_REALTIME_REQUIRE_ROCE=0 against such a build is
/// itself a failure, whether or not the transport is usable.  Otherwise
/// CUDAQ_REALTIME_REQUIRE_ROCE=1 raises the requirement for one run.  When
/// required, any probe failure fails; when not, it skips.
inline RoceGate roce_test_gate(const RoceTopology &topology) {
  const bool built_required = CUDAQ_REALTIME_REQUIRE_ROCE_TESTS != 0;
  const char *env = std::getenv("CUDAQ_REALTIME_REQUIRE_ROCE");
  bool required = built_required;
  if (env && *env) {
    if (std::strcmp(env, "0") != 0 && std::strcmp(env, "1") != 0)
      return {RoceGate::Fail, std::string("CUDAQ_REALTIME_REQUIRE_ROCE='") +
                                  env + "' must be 0 or 1"};
    if (built_required && env[0] == '0')
      return {RoceGate::Fail,
              "CUDAQ_REALTIME_REQUIRE_ROCE=0 contradicts this build, which "
              "was configured with CUDAQ_REALTIME_REQUIRE_ROCE_TESTS=ON"};
    required = env[0] == '1' || built_required;
  }
  std::string why = probe_roce_topology(topology);
  if (why.empty())
    return {};
  if (required)
    return {RoceGate::Fail, "RoCE transport required but unavailable: " + why};
  return {RoceGate::Skip, "RoCE transport unavailable: " + why};
}

} // namespace cudaq::realtime::testing
