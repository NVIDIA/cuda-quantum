/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/roce/roce_channel.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"

#include <cstring>
#include <stdexcept>
#include <sys/mman.h>

namespace cudaq::nvqlink {

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

RoCEChannel::RoCEChannel(const std::string &device_name,
                         std::uint16_t listen_port,
                         std::shared_ptr<FlowSwitch> flow_switch)
    : listen_port_(listen_port), flow_switch_(std::move(flow_switch)) {
  init_independent(device_name);
  buffer_pool_ = std::make_unique<RoCEBufferPool>(RECV_RING_SIZE);
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEChannel: Created (independent mode, device: {}, port: "
                   "{}, UC mode with memory polling)",
                   device_name, listen_port_);
}

RoCEChannel::RoCEChannel(std::shared_ptr<VerbsContext> shared_ctx,
                         std::uint16_t listen_port,
                         std::shared_ptr<FlowSwitch> flow_switch)
    : listen_port_(listen_port), flow_switch_(std::move(flow_switch)) {
  init_shared(std::move(shared_ctx));
  buffer_pool_ = std::make_unique<RoCEBufferPool>(RECV_RING_SIZE);
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEChannel: Created (shared mode, device: {}, port: {}, "
                   "UC mode with memory polling)",
                   shared_ctx_->get_device_name(), listen_port_);
}

RoCEChannel::~RoCEChannel() {
  cleanup();

  // Deregister and free RX ring buffer memory
  if (rx_ring_buffer_mr_ != nullptr) {
    ibv_dereg_mr(rx_ring_buffer_mr_);
    rx_ring_buffer_mr_ = nullptr;
  }
  if (rx_ring_buffer_base_ != nullptr) {
    munmap(rx_ring_buffer_base_, rx_ring_buffer_config_.total_size());
    rx_ring_buffer_base_ = nullptr;
  }

  // Deregister and free TX ring buffer memory
  if (tx_ring_buffer_mr_ != nullptr) {
    ibv_dereg_mr(tx_ring_buffer_mr_);
    tx_ring_buffer_mr_ = nullptr;
  }
  if (tx_ring_buffer_base_ != nullptr) {
    munmap(tx_ring_buffer_base_, tx_num_slots_ * tx_slot_size_);
    tx_ring_buffer_base_ = nullptr;
  }

  // Cleanup independent mode resources
  if (owns_context_) {
    if (owned_pd_) {
      ibv_dealloc_pd(owned_pd_);
      owned_pd_ = nullptr;
    }
    if (owned_context_) {
      ibv_close_device(owned_context_);
      owned_context_ = nullptr;
    }
  }
}

//===----------------------------------------------------------------------===//
// Initialization helpers
//===----------------------------------------------------------------------===//

void RoCEChannel::init_independent(const std::string &device_name) {
  // Open device and create protection domain ourselves
  int num_devices;
  struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list || num_devices == 0)
    throw std::runtime_error("No InfiniBand devices found");

  // Find device by name
  struct ibv_device *device = nullptr;
  for (int i = 0; i < num_devices; i++) {
    const char *dev_name = ibv_get_device_name(dev_list[i]);
    if (device_name == dev_name) {
      device = dev_list[i];
      break;
    }
  }

  // If not found by name, try using index
  if (!device) {
    try {
      int dev_idx = std::stoi(device_name);
      if (dev_idx >= 0 && dev_idx < num_devices)
        device = dev_list[dev_idx];
    } catch (const std::invalid_argument &) {
      // Not a number
    }
  }

  if (!device) {
    ibv_free_device_list(dev_list);
    throw std::runtime_error("Device '" + device_name + "' not found");
  }

  // Open device context
  owned_context_ = ibv_open_device(device);
  if (!owned_context_) {
    ibv_free_device_list(dev_list);
    throw std::runtime_error("Failed to open device '" + device_name + "'");
  }

  // Free device list immediately after opening device - no longer needed
  ibv_free_device_list(dev_list);
  dev_list = nullptr;

  // Create protection domain
  owned_pd_ = ibv_alloc_pd(owned_context_);
  if (!owned_pd_) {
    ibv_close_device(owned_context_);
    owned_context_ = nullptr;
    throw std::runtime_error("Failed to allocate Protection Domain");
  }

  owns_context_ = true;

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEChannel: Initialized independent context for device {}",
                   device_name);
}

void RoCEChannel::init_shared(std::shared_ptr<VerbsContext> shared_ctx) {
  if (!shared_ctx)
    throw std::runtime_error("Shared VerbsContext cannot be null");

  shared_ctx_ = std::move(shared_ctx);
  owns_context_ = false;

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEChannel: Using shared context for device {}",
                   shared_ctx_->get_device_name());
}

//===----------------------------------------------------------------------===//
// Resource access helpers
//===----------------------------------------------------------------------===//

struct ibv_context *RoCEChannel::get_context() const {
  if (shared_ctx_)
    return shared_ctx_->get_context();
  return owned_context_;
}

struct ibv_pd *RoCEChannel::get_pd() const {
  if (shared_ctx_)
    return shared_ctx_->get_protection_domain();
  return owned_pd_;
}

std::uint8_t RoCEChannel::get_port_num() const {
  if (shared_ctx_)
    return shared_ctx_->get_port_num();
  return port_num_;
}

//===----------------------------------------------------------------------===//
// Channel interface implementation
//===----------------------------------------------------------------------===//

void RoCEChannel::initialize() {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "RoCEChannel::initialize");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "RoCEChannel: Initializing");

  // Create completion queues
  create_completion_queues();

  // Preallocate receive buffers
  preallocate_recv_buffers(RECV_RING_SIZE);

  // Create Queue Pair
  create_queue_pair();

  // Initialize ring buffers (pre-allocated, NO allocation on hot path)
  initialize_rx_ring_buffer(); // For receiving RDMA WRITEs
  initialize_tx_ring_buffer(); // For sending responses

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "RoCEChannel: Initialization complete");
}

void RoCEChannel::cleanup() {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "RoCEChannel::cleanup");
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "RoCEChannel: Cleaning up");

  // Unregister from flow switch
  if (flow_switch_ && qp_) {
    flow_switch_->remove_steering_rule(listen_port_);
  }

  // Destroy Queue Pair
  if (qp_) {
    ibv_destroy_qp(qp_);
    qp_ = nullptr;
  }

  // Destroy Completion Queues
  if (recv_cq_) {
    ibv_destroy_cq(recv_cq_);
    recv_cq_ = nullptr;
  }

  if (send_cq_) {
    ibv_destroy_cq(send_cq_);
    send_cq_ = nullptr;
  }

  // Free pre-allocated receive buffers and deregister MRs
  for (RecvBuffer &rb : recv_buffers_) {
    if (rb.mr)
      ibv_dereg_mr(rb.mr);
    if (rb.addr)
      free(rb.addr);
  }
  recv_buffers_.clear();
  buffer_id_map_.clear();
}

Buffer *RoCEChannel::acquire_buffer() {
  // Acquire from pre-allocated TX ring buffer
  if (!buffer_pool_ || !tx_ring_buffer_base_)
    return nullptr;

  // Find a free slot using round-robin search (O(1) amortized)
  for (std::uint32_t i = 0; i < tx_num_slots_; ++i) {
    std::size_t slot_idx = (tx_next_slot_ + i) % tx_num_slots_;
    if (tx_slot_free_[slot_idx]) {
      tx_slot_free_[slot_idx] = false; // Mark as allocated
      tx_next_slot_ = (slot_idx + 1) % tx_num_slots_;

      // Get a pre-allocated buffer wrapper from pool
      Buffer *buffer = buffer_pool_->get_free_buffer();
      if (!buffer) {
        tx_slot_free_[slot_idx] = true; // Release slot
        return nullptr;
      }

      // Point buffer wrapper to TX slot (zero-copy, just pointer assignment)
      char *slot_addr = static_cast<char *>(tx_ring_buffer_base_) +
                        (slot_idx * tx_slot_size_);
      buffer->reset(slot_addr, slot_addr, tx_slot_size_, 0, 0, tx_slot_size_);

      // Store slot_idx for deallocation
      buffer_pool_->set_wr_id(buffer, slot_idx);

      return buffer;
    }
  }

  return nullptr; // No free slots
}

void RoCEChannel::release_buffer(Buffer *buffer) {
  if (!buffer || !buffer_pool_)
    return;

  // Check if this buffer is from TX ring buffer
  void *data = buffer->get_base_address();
  uintptr_t addr = reinterpret_cast<uintptr_t>(data);
  uintptr_t tx_start = reinterpret_cast<uintptr_t>(tx_ring_buffer_base_);
  uintptr_t tx_end = tx_start + (tx_num_slots_ * tx_slot_size_);

  if (tx_ring_buffer_base_ && addr >= tx_start && addr < tx_end) {
    // TX buffer - try to free the slot (may already be freed if double-dealloc)
    try {
      std::uint64_t slot_idx = buffer_pool_->get_wr_id(buffer);
      if (slot_idx < tx_num_slots_) {
        tx_slot_free_[slot_idx] = true;
      }
    } catch (const std::exception &) {
      // Buffer already deallocated or RX buffer in TX range (shouldn't happen)
      // Just ignore - the slot may already be freed
    }
  }
  // RX buffers are managed by ring buffer poller, not freed here

  // Return buffer wrapper to pool (idempotent - safe to call multiple times)
  buffer_pool_->return_buffer(buffer);
}

void RoCEChannel::create_completion_queues() {
  auto *context = get_context();

  // Create send completion queue
  send_cq_ = ibv_create_cq(context, CQ_SIZE, nullptr, nullptr, 0);
  if (!send_cq_)
    throw std::runtime_error("Failed to create send CQ");

  // Create receive completion queue
  recv_cq_ = ibv_create_cq(context, CQ_SIZE, nullptr, nullptr, 0);
  if (!recv_cq_) {
    ibv_destroy_cq(send_cq_);
    throw std::runtime_error("Failed to create receive CQ");
  }

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEChannel: Created completion queues (size={})", CQ_SIZE);
}

void RoCEChannel::configure_queues(const std::vector<std::uint32_t> &) {
  // No-op: Queue configuration is now handled via FlowSwitch
  // This method is kept for Channel interface compatibility
}

void RoCEChannel::create_queue_pair() {
  struct ibv_pd *pd = get_pd();

  // Create Queue Pair (UC mode only)
  struct ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = send_cq_;
  qp_attr.recv_cq = recv_cq_;
  qp_attr.qp_type = IBV_QPT_UC;
  qp_attr.cap.max_send_wr = SEND_RING_SIZE;
  qp_attr.cap.max_recv_wr = RECV_RING_SIZE;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  qp_attr.cap.max_inline_data = MAX_INLINE_DATA;

  qp_ = ibv_create_qp(pd, &qp_attr);
  if (!qp_)
    throw std::runtime_error("Failed to create QP");

  // UC mode: Wait for remote QP info before transitioning
  // Will be done in set_remote_qp()
  NVQLINK_LOG_INFO(
      DOMAIN_CHANNEL,
      "RoCEChannel: Created UC QP (qpn={}) - waiting for remote QP info",
      qp_->qp_num);

  // Register with flow switch for traffic steering
  if (flow_switch_) {
    flow_switch_->add_steering_rule(listen_port_, static_cast<void *>(qp_));
  }
}

void RoCEChannel::preallocate_recv_buffers(std::uint32_t count) {
  struct ibv_pd *pd = get_pd();

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEChannel: Pre-allocating {} receive buffers...", count);

  recv_buffers_.reserve(count);
  for (std::uint32_t i = 0; i < count; i++) {
    RecvBuffer rb;

    rb.addr = aligned_alloc(64, RECV_BUFFER_SIZE);
    if (!rb.addr)
      throw std::runtime_error("Failed to allocate receive buffer");

    rb.mr = ibv_reg_mr(pd, rb.addr, RECV_BUFFER_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!rb.mr) {
      free(rb.addr);
      throw std::runtime_error("Failed to register receive buffer MR");
    }

    // Assign unique buffer ID
    rb.buffer_id = i;

    recv_buffers_.push_back(rb);
    buffer_id_map_[rb.buffer_id] = &recv_buffers_[i];
  }
}

void RoCEChannel::initial_post_recv_buffers(struct ibv_qp *qp) {
  // Post all pre-allocated buffers to the QP
  for (RecvBuffer &rb : recv_buffers_)
    repost_recv_buffer(qp, &rb);
}

void RoCEChannel::repost_recv_buffer(struct ibv_qp *qp, RecvBuffer *rb) {
  // Just post WR for existing buffer
  struct ibv_sge sge = {};
  sge.addr = reinterpret_cast<std::uint64_t>(rb->addr);
  sge.length = RECV_BUFFER_SIZE;
  sge.lkey = rb->mr->lkey;

  struct ibv_recv_wr wr = {};
  wr.wr_id = rb->buffer_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr *bad_wr = nullptr;
  int ret = ibv_post_recv(qp, &wr, &bad_wr);
  if (ret != 0)
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "RoCEChannel: Failed to repost receive buffer: {}",
                      strerror(errno));
}

void *RoCEChannel::allocate_and_register_ring_buffer(std::size_t size,
                                                     int access_flags,
                                                     struct ibv_mr **out_mr,
                                                     const char *buffer_name) {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL,
                     "RoCEChannel::allocate_and_register_ring_buffer");

  // Step 1: Try to allocate with hugepages
  void *buffer_base = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

  if (buffer_base == MAP_FAILED) {
    NVQLINK_LOG_WARNING(DOMAIN_CHANNEL,
                        "Warning: Failed to allocate hugepages for {} ring "
                        "buffer, using regular pages",
                        buffer_name);

    // Step 2: Fallback to regular pages
    buffer_base = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (buffer_base == MAP_FAILED) {
      throw std::runtime_error(std::string("Failed to allocate memory for ") +
                               buffer_name + " ring buffer");
    }
  }

  // Step 3: Register memory with RDMA
  *out_mr = ibv_reg_mr(get_pd(), buffer_base, size, access_flags);

  if (!*out_mr) {
    munmap(buffer_base, size);
    throw std::runtime_error(
        std::string("Failed to register ") + buffer_name +
        " ring buffer memory region: " + std::string(strerror(errno)));
  }

  return buffer_base;
}

void RoCEChannel::initialize_rx_ring_buffer() {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "RoCEChannel::initialize_rx_ring_buffer");

  if (rx_ring_buffer_poller_)
    return; // Already initialized

  // Configure RX ring buffer
  rx_ring_buffer_config_.num_slots = 1024;
  rx_ring_buffer_config_.slot_size = 2048;
  std::size_t ring_buffer_size = rx_ring_buffer_config_.total_size();

  // Allocate and register RX ring buffer
  // Needs REMOTE_WRITE for client to write to us via RDMA WRITE
  rx_ring_buffer_base_ = allocate_and_register_ring_buffer(
      ring_buffer_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
      &rx_ring_buffer_mr_, "RX");

  // Initialize RX ring buffer poller
  rx_ring_buffer_poller_ = std::make_unique<RingBufferPoller>(
      rx_ring_buffer_base_, rx_ring_buffer_config_);

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "✓ Initialized RX ring buffer:");
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "  Base: {}, Slots: {}, Size: {} MB",
                   rx_ring_buffer_base_, rx_ring_buffer_config_.num_slots,
                   ring_buffer_size / (1024 * 1024));
}

void RoCEChannel::initialize_tx_ring_buffer() {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "RoCEChannel::initialize_tx_ring_buffer");

  if (tx_ring_buffer_base_)
    return; // Already initialized

  // Pre-allocate TX ring buffer (NO allocation on hot path!)
  std::size_t tx_buffer_size = tx_num_slots_ * tx_slot_size_;

  // Allocate and register TX ring buffer
  // Only LOCAL_WRITE needed for TX (we write, then send via RDMA SEND)
  tx_ring_buffer_base_ = allocate_and_register_ring_buffer(
      tx_buffer_size, IBV_ACCESS_LOCAL_WRITE, &tx_ring_buffer_mr_, "TX");

  // Pre-allocate slot tracking (all slots start free)
  tx_slot_free_.resize(tx_num_slots_, true);
  tx_next_slot_ = 0;

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "✓ Initialized TX ring buffer:");
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "  Base: {}, Slots: {}, Size: {} MB",
                   tx_ring_buffer_base_, tx_num_slots_,
                   tx_buffer_size / (1024 * 1024));
}

void RoCEChannel::register_memory(void *addr, std::size_t size) {
  // Register memory with our protection domain
  struct ibv_pd *pd = get_pd();

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEChannel: Registering memory: {} size {} MB", addr,
                   size / (1024 * 1024));

  struct ibv_mr *mr =
      ibv_reg_mr(pd, addr, size,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                     IBV_ACCESS_REMOTE_READ);

  if (!mr)
    throw std::runtime_error("Failed to register memory region: " +
                             std::string(strerror(errno)));

  NVQLINK_LOG_INFO(
      DOMAIN_CHANNEL,
      "✓ Successfully registered {} MB memory with RoCE (lkey={}, rkey={})",
      size / (1024 * 1024), mr->lkey, mr->rkey);
}

std::uint32_t RoCEChannel::receive_burst(Buffer **buffers,
                                         std::uint32_t max_count) {
  NVQLINK_TRACE_HOTPATH(DOMAIN_CHANNEL, "receive_burst");

  std::uint32_t count = 0;
  while (count < max_count) {
    char *data = nullptr;
    std::uint32_t len = 0;

    // Poll for next packet from RX ring buffer (non-blocking)
    if (!rx_ring_buffer_poller_->poll_next(&data, &len))
      break; // No more packets available

    // Wrap packet data into Buffer (zero-copy, just pointer)
    // Use wrap_ring_buffer_data() because RDMA WRITE delivers raw payload
    try {
      buffers[count] = buffer_pool_->wrap_ring_buffer_data(data, len, count);
      count++;
    } catch (const std::exception &e) {
      NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                        "RoCEChannel: Failed to wrap buffer: {}", e.what());
      break;
    }
  }

  return count;
}

std::uint32_t RoCEChannel::send_burst(Buffer **buffers, std::uint32_t count) {
  NVQLINK_TRACE_HOTPATH(DOMAIN_CHANNEL, "send_burst");

  if (!qp_) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "RoCEChannel: No QP configured");
    return 0;
  }

  if (!remote_qp_set_) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "RoCEChannel: Cannot send - remote QP not configured");
    return 0;
  }

  if (!tx_ring_buffer_mr_ && !rx_ring_buffer_mr_) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                      "RoCEChannel: No ring buffers registered");
    return 0;
  }

  std::uint32_t sent = 0;

  for (std::uint32_t i = 0; i < count; i++) {
    void *response_data = buffers[i]->get_data();
    std::size_t response_len = buffers[i]->get_data_length();

    // Determine which ring buffer this data is in and use correct lkey
    // TX ring buffer: for buffers allocated via allocate_buffer()
    // RX ring buffer: for buffers received and reused (Daemon pattern)
    uintptr_t addr = reinterpret_cast<uintptr_t>(response_data);
    uintptr_t tx_start = reinterpret_cast<uintptr_t>(tx_ring_buffer_base_);
    uintptr_t tx_end = tx_start + (tx_num_slots_ * tx_slot_size_);
    uintptr_t rx_start = reinterpret_cast<uintptr_t>(rx_ring_buffer_base_);
    uintptr_t rx_end = rx_start + rx_ring_buffer_config_.total_size();

    uint32_t lkey;
    if (tx_ring_buffer_base_ && addr >= tx_start && addr < tx_end) {
      lkey = tx_ring_buffer_mr_->lkey; // TX buffer (Channel/Stream mode)
    } else if (rx_ring_buffer_base_ && addr >= rx_start && addr < rx_end) {
      lkey = rx_ring_buffer_mr_->lkey; // RX buffer reuse (Daemon mode)
    } else {
      NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                        "RoCEChannel: Buffer not in any ring buffer");
      continue;
    }

    // Post RDMA SEND with correct lkey
    struct ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(response_data);
    sge.length = response_len;
    sge.lkey = lkey;

    struct ibv_send_wr wr = {};
    wr.wr_id = i; // Simple ID for tracking
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED; // Request completion notification

    struct ibv_send_wr *bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    if (ret != 0) {
      NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                        "RoCEChannel: Failed to post RDMA SEND: {}",
                        strerror(errno));
      // Ownership semantics: send_burst takes ownership even on failure
      release_buffer(buffers[i]);
      continue;
    }

    NVQLINK_LOG_DEBUG(DOMAIN_CHANNEL,
                      "[RoCE TX] Sent RDMA SEND: {} bytes (qpn={})",
                      response_len, qp_->qp_num);
    sent++;

    // Poll for send completion immediately (blocking)
    struct ibv_wc wc;
    int n = 0;
    while (n == 0)
      n = ibv_poll_cq(send_cq_, 1, &wc);

    if (n < 0 || wc.status != IBV_WC_SUCCESS) {
      NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                        "RoCEChannel: Send completion failed: {}",
                        (n < 0 ? "poll error" : ibv_wc_status_str(wc.status)));
    }

    // Ownership semantics: send_burst owns buffer, releases after TX completes
    release_buffer(buffers[i]);
  }

  return sent;
}

void RoCEChannel::transition_qp_to_rts(struct ibv_qp *qp) {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "RoCEChannel::transition_qp_to_rts");

  if (!remote_qp_set_)
    throw std::runtime_error(
        "Cannot transition UC QP to RTS: remote QP not set");

  auto port_num = get_port_num();

  // Step 1: RESET → INIT
  struct ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = port_num;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;

  int mask =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  int ret = ibv_modify_qp(qp, &attr, mask);
  if (ret != 0)
    throw std::runtime_error("Failed to transition UC QP to INIT");

  // Step 2: INIT → RTR (requires remote QP info for UC!)
  std::memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_1024; // Or query from port
  attr.dest_qp_num = remote_qpn_;
  attr.rq_psn = 0; // Expected receive PSN

  // Address vector for RoCEv2
  attr.ah_attr.dlid = 0; // Not used in RoCEv2
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = port_num;
  attr.ah_attr.is_global = 1; // Must be 1 for RoCEv2
  attr.ah_attr.grh.dgid = remote_gid_;
  attr.ah_attr.grh.sgid_index = 1; // Use GID index 1 for Soft-RoCE
  attr.ah_attr.grh.hop_limit = 64;

  mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
         IBV_QP_RQ_PSN;
  ret = ibv_modify_qp(qp, &attr, mask);
  if (ret != 0)
    throw std::runtime_error("Failed to transition UC QP to RTR");

  // Step 3: RTR → RTS
  std::memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0; // Starting send PSN

  mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  ret = ibv_modify_qp(qp, &attr, mask);
  if (ret != 0)
    throw std::runtime_error("Failed to transition UC QP to RTS");

  NVQLINK_LOG_INFO(
      DOMAIN_CHANNEL,
      "RoCEChannel: UC QP transitioned to RTS (qpn={}, remote_qpn={})",
      qp->qp_num, remote_qpn_);
}

RoCEChannel::ConnectionParams RoCEChannel::get_connection_params() const {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "RoCEChannel::get_connection_params");

  ConnectionParams params = {};

  // QPN from queue pair
  if (qp_)
    params.qpn = qp_->qp_num;

  params.psn = 0; // Starting PSN
  params.lid = 0; // Not used in RoCEv2

  // Get local GID (use index 1 for Soft-RoCE, 0 often doesn't work)
  int ret = ibv_query_gid(get_context(), get_port_num(),
                          1, // GID index (use 1 for Soft-RoCE, not 0)
                          &params.gid);
  if (ret != 0) {
    NVQLINK_LOG_WARNING(
        DOMAIN_CHANNEL,
        "Warning: Failed to query GID at index 1, trying index 0");
    // Fallback to index 0 if index 1 fails
    ret = ibv_query_gid(get_context(), get_port_num(), 0, &params.gid);
    if (ret != 0)
      NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Error: Failed to query GID");
  }

  // Memory region info (for RDMA WRITE target - client writes to RX ring
  // buffer)
  if (rx_ring_buffer_base_ != nullptr && rx_ring_buffer_mr_ != nullptr) {
    params.vaddr = reinterpret_cast<std::uint64_t>(rx_ring_buffer_base_);
    params.rkey = rx_ring_buffer_mr_->rkey;
    params.num_slots = rx_ring_buffer_config_.num_slots;
    params.slot_size = rx_ring_buffer_config_.slot_size;
  } else {
    throw std::runtime_error(
        "RX ring buffer not initialized - cannot export connection params");
  }

  return params;
}

void RoCEChannel::set_remote_qp(std::uint32_t remote_qpn,
                                const union ibv_gid &remote_gid) {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "RoCEChannel::set_remote_qp");

  remote_qpn_ = remote_qpn;
  remote_gid_ = remote_gid;
  remote_qp_set_ = true;

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "RoCEChannel: Set remote QP (qpn={})",
                   remote_qpn_);

  // Transition QP to RTS
  if (qp_)
    transition_qp_to_rts(qp_);

  // Now that QP is in RTS, post all pre-allocated receive buffers
  if (qp_)
    initial_post_recv_buffers(qp_);

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEChannel: Posted {} receive buffers (after UC "
                   "connection established)",
                   RECV_RING_SIZE);
}

} // namespace cudaq::nvqlink
