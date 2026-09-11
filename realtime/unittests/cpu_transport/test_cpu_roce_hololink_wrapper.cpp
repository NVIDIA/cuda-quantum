/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

namespace {

std::size_t send_full_slot(void *, const void *, void *,
                           std::size_t slot_size) {
  return slot_size;
}

cpu_roce_transceiver_t create_valid(int forward = 0, int rx_only = 0,
                                    int tx_only = 0, int unified = 0) {
  return cpu_roce_create_transceiver("not-opened-by-create", 1, 2, 64, 256, 8,
                                     "192.0.2.1", forward, rx_only, tx_only,
                                     unified, CPU_ROCE_TX_MODE_RDMA_SEND, 0, 0);
}

TEST(CpuRoceHololinkWrapper, CreateDoesNotOpenDevice) {
  auto handle = create_valid();
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(cpu_roce_get_page_size(handle), 256);
  EXPECT_EQ(cpu_roce_get_num_pages(handle), 8);
  EXPECT_EQ(cpu_roce_get_qp_number(handle), 0);
  EXPECT_EQ(cpu_roce_get_rkey(handle), 0);
  EXPECT_EQ(cpu_roce_get_buffer_addr(handle), 0);
  EXPECT_EQ(cpu_roce_get_rx_ring_data_addr(handle), nullptr);
  EXPECT_EQ(cpu_roce_get_rx_ring_flag_addr(handle), nullptr);
  EXPECT_EQ(cpu_roce_get_tx_ring_data_addr(handle), nullptr);
  EXPECT_EQ(cpu_roce_get_tx_ring_flag_addr(handle), nullptr);
  cpu_roce_destroy_transceiver(handle);
}

TEST(CpuRoceHololinkWrapper, RejectsConflictingModesAtCreate) {
  EXPECT_EQ(create_valid(/*forward=*/1, /*rx_only=*/1), nullptr);
}

TEST(CpuRoceHololinkWrapper, AcceptsPreSetupLocalIpOverride) {
  auto handle = create_valid();
  ASSERT_NE(handle, nullptr);
  testing::internal::CaptureStderr();
  cpu_roce_set_local_ip(handle, "192.0.2.2");
  cpu_roce_set_local_ip(handle, nullptr);
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
  cpu_roce_destroy_transceiver(handle);
}

TEST(CpuRoceHololinkWrapper, ReportsInvalidLocalIpWithoutBlockingSetup) {
  auto handle = create_valid();
  ASSERT_NE(handle, nullptr);
  testing::internal::CaptureStderr();
  cpu_roce_set_local_ip(handle, "not-an-ipv4-address");
  EXPECT_NE(testing::internal::GetCapturedStderr().find(
                "local_ip must be a valid IPv4 address"),
            std::string::npos);
  // The rejected value is not stored, so setup() fails only because the
  // placeholder device cannot be opened, matching legacy GID fallback.
  testing::internal::CaptureStderr();
  EXPECT_EQ(cpu_roce_setup(handle), 0);
  EXPECT_EQ(testing::internal::GetCapturedStderr().find("local_ip"),
            std::string::npos);
  cpu_roce_destroy_transceiver(handle);
}

TEST(CpuRoceHololinkWrapper, InstallsLegacyUnifiedCallbackBeforeMonitor) {
  // Invocation needs a live QP, but installation itself must remain valid
  // when frame_size is smaller than the slot size the C callback receives.
  auto handle = create_valid(/*forward=*/0, /*rx_only=*/0, /*tx_only=*/0,
                             /*unified=*/1);
  ASSERT_NE(handle, nullptr);
  cpu_roce_set_unified_dispatch(handle, &send_full_slot, nullptr);
  cpu_roce_destroy_transceiver(handle);
}

TEST(CpuRoceHololinkWrapper, RejectsInvalidRingGeometryAtCreate) {
  EXPECT_EQ(cpu_roce_create_transceiver("not-opened-by-create", 1, 2, 64, 32, 8,
                                        "192.0.2.1", 0, 0, 0, 0,
                                        CPU_ROCE_TX_MODE_RDMA_SEND, 0, 0),
            nullptr);
  EXPECT_EQ(cpu_roce_create_transceiver("not-opened-by-create", 1, 2, 64, 256,
                                        3, "192.0.2.1", 0, 0, 0, 0,
                                        CPU_ROCE_TX_MODE_RDMA_SEND, 0, 0),
            nullptr);
}

TEST(CpuRoceHololinkWrapper, NullHandleAccessorsPreserveCContract) {
  EXPECT_EQ(cpu_roce_start(nullptr), 0);
  EXPECT_EQ(cpu_roce_setup(nullptr), 0);
  EXPECT_EQ(cpu_roce_connect(nullptr, 1, "192.0.2.1", 1), 0);
  EXPECT_EQ(cpu_roce_get_qp_number(nullptr), 0);
  EXPECT_EQ(cpu_roce_get_rkey(nullptr), 0);
  EXPECT_EQ(cpu_roce_get_buffer_addr(nullptr), 0);
  EXPECT_EQ(cpu_roce_get_rx_ring_data_addr(nullptr), nullptr);
  EXPECT_EQ(cpu_roce_get_rx_ring_flag_addr(nullptr), nullptr);
  EXPECT_EQ(cpu_roce_get_tx_ring_data_addr(nullptr), nullptr);
  EXPECT_EQ(cpu_roce_get_tx_ring_flag_addr(nullptr), nullptr);
  EXPECT_EQ(cpu_roce_get_page_size(nullptr), 0);
  EXPECT_EQ(cpu_roce_get_num_pages(nullptr), 0);
  cpu_roce_close(nullptr);
  cpu_roce_blocking_monitor(nullptr);
  cpu_roce_set_local_ip(nullptr, "192.0.2.1");
  cpu_roce_set_unified_dispatch(nullptr, &send_full_slot, nullptr);
  cpu_roce_destroy_transceiver(nullptr);
}

} // namespace
