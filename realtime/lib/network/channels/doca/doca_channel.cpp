/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/channels/doca/doca_channel.h"
#include "cudaq/nvqlink/network/channels/doca/doca_buffer_pool.h"
#include "cudaq/nvqlink/network/channels/doca/doca_completion_queue.h"
#include "cudaq/nvqlink/network/channels/doca/doca_queue_pair.h"
#include "cudaq/nvqlink/network/channels/doca/doca_rpc_kernel.h"
#include "cudaq/nvqlink/utils/instrumentation/domains.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <doca_log.h>
#include <doca_rdma_bridge.h>
#include <doca_verbs_bridge.h>
#include <infiniband/verbs.h>

#include <cerrno>
#include <cstring>
#include <unistd.h>

namespace cudaq::nvqlink {

// Utility functions
static std::size_t get_page_size() {
  long ret = sysconf(_SC_PAGESIZE);
  if (ret == -1)
    return 4096;
  return static_cast<std::size_t>(ret);
}

static doca_verbs_context *open_ib_device(const std::string &dev_name) {
  int nb_ibdevs = 0;
  ibv_device **ibdev_list = ibv_get_device_list(&nb_ibdevs);
  doca_verbs_context *context = nullptr;

  if (ibdev_list == nullptr || nb_ibdevs == 0) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to get RDMA devices list");
    return nullptr;
  }

  for (int i = 0; i < nb_ibdevs; i++) {
    if (strncmp(ibv_get_device_name(ibdev_list[i]), dev_name.c_str(),
                dev_name.length()) == 0) {
      ibv_device *dev_handle = ibdev_list[i];
      ibv_free_device_list(ibdev_list);

      // Pass ibv_device* (NOT ibv_context*) per DOCA 3.x API
      if (doca_verbs_bridge_verbs_context_create(
              dev_handle, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, &context) !=
          DOCA_SUCCESS) {
        NVQLINK_LOG_ERROR(DOMAIN_CHANNEL,
                          "Failed to create DOCA Verbs context");
        return nullptr;
      }

      return context;
    }
  }

  ibv_free_device_list(ibdev_list);
  NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "IB device not found: {}", dev_name);
  return nullptr;
}

DOCAChannel::DOCAChannel(const DOCAChannelConfig &config) : config_(config) {
  if (!config_.is_valid()) {
    throw std::invalid_argument("Invalid DOCA channel configuration");
  }
}

DOCAChannel::~DOCAChannel() { cleanup(); }

void DOCAChannel::initialize() {
  try {
    NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Initializing DOCA channel");

    doca_log_backend *sdk_backend = nullptr;
    doca_error_t result =
        doca_log_backend_create_with_file_sdk(stderr, &sdk_backend);
    if (result != DOCA_SUCCESS) {
      NVQLINK_LOG_WARNING(DOMAIN_CHANNEL,
                          "Failed to create DOCA SDK log backend");
    } else {
      // Enable verbose DOCA logging for debugging
      doca_log_level_set_global_sdk_limit(DOCA_LOG_LEVEL_DEBUG);
    }

    init_doca_device();
    init_gpu_device();
    init_protection_domain();
    find_roce_gid();
    init_uar();
    init_completion_queues();
    init_buffer_pool();
    init_queue_pair();
    init_exit_flag();

    NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "DOCA channel initialized successfully");

  } catch (const std::exception &e) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "DOCA channel initialization failed: {}",
                      e.what());
    cleanup();
    throw;
  }
}

void DOCAChannel::init_doca_device() {
  NVQLINK_LOG_DEBUG(DOMAIN_CHANNEL, "Opening DOCA device: {}",
                    config_.nic_device);
  verbs_ctx_ = open_ib_device(config_.nic_device);
  if (!verbs_ctx_) {
    throw std::runtime_error("Failed to open DOCA device: " +
                             config_.nic_device);
  }
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "DOCA device opened successfully");
}

void DOCAChannel::init_gpu_device() {
  cudaError_t result_cuda;
  int gpu_id = config_.gpu_device_id;

  result_cuda = cudaSetDevice(gpu_id);
  if (result_cuda != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice failed");
  }

  char gpu_bus_id[256];
  result_cuda = cudaDeviceGetPCIBusId(gpu_bus_id, sizeof(gpu_bus_id), gpu_id);
  if (result_cuda != cudaSuccess) {
    throw std::runtime_error("cudaDeviceGetPCIBusId failed");
  }

  doca_error_t result = doca_gpu_create(gpu_bus_id, &doca_gpu_device_);
  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to create DOCA GPU device: " +
                             std::string(doca_error_get_descr(result)));
  }

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Created GPU device successfully");
}

void DOCAChannel::init_protection_domain() {
  doca_error_t result = doca_verbs_pd_create(verbs_ctx_, &verbs_pd_);
  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to create DOCA Verbs PD: " +
                             std::string(doca_error_get_descr(result)));
  }

  ibv_pd_ = doca_verbs_bridge_verbs_pd_get_ibv_pd(verbs_pd_);
  if (!ibv_pd_) {
    throw std::runtime_error("Failed to get ibv_pd");
  }

  result = doca_rdma_bridge_open_dev_from_pd(ibv_pd_, &doca_device_);
  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to open DOCA device from PD: " +
                             std::string(doca_error_get_descr(result)));
  }
}

void DOCAChannel::find_roce_gid() {
  ibv_gid_entry ib_gid_entry;
  bool gid_found = false;

  for (gid_index_ = 0; gid_index_ < 256; gid_index_++) {
    std::uint32_t flags = 0;
    int ret = ibv_query_gid_ex(ibv_pd_->context, config_.nic_port, gid_index_,
                               &ib_gid_entry, flags);
    if (ret != 0 && errno != ENODATA) {
      break;
    }

    NVQLINK_LOG_DEBUG(DOMAIN_CHANNEL,
                      "gid_index={} port_num={} gid_type={} subnet_prefix={} "
                      "interface_id={:#x}",
                      gid_index_, ib_gid_entry.port_num, ib_gid_entry.gid_type,
                      ib_gid_entry.gid.global.subnet_prefix,
                      ib_gid_entry.gid.global.interface_id);

    // Check for RoCE v2 GID characteristics
    if (ib_gid_entry.gid_type == IBV_GID_TYPE_ROCE_V2 &&
        ib_gid_entry.gid.global.subnet_prefix == 0 &&
        (ib_gid_entry.gid.global.interface_id & 0xFFFFFFFF) == 0xFFFF0000) {
      gid_found = true;
      NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Found RoCE v2 GID at index {}",
                       gid_index_);
      break;
    }
  }

  if (!gid_found) {
    throw std::runtime_error("Cannot find GID for RoCE v2");
  }
}

void DOCAChannel::init_uar() {
  doca_error_t result =
      doca_uar_create(doca_device_, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &uar_);
  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to create UAR: " +
                             std::string(doca_error_get_descr(result)));
  }
}

void DOCAChannel::init_completion_queues() {
  doca_error_t result;

  cq_rq_ = std::make_unique<DOCACompletionQueue>(
      config_.wqe_num, doca_gpu_device_, doca_device_, uar_, verbs_ctx_);
  result = cq_rq_->create();
  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to create RQ CQ: " +
                             std::string(doca_error_get_descr(result)));
  }

  cq_sq_ = std::make_unique<DOCACompletionQueue>(
      config_.wqe_num, doca_gpu_device_, doca_device_, uar_, verbs_ctx_);
  result = cq_sq_->create();
  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to create SQ CQ: " +
                             std::string(doca_error_get_descr(result)));
  }

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Created CQs successfully");
}

void DOCAChannel::init_buffer_pool() {
  buffer_pool_ = std::make_unique<DOCABufferPool>(
      doca_gpu_device_, doca_device_, ibv_pd_, config_.buffer_size,
      config_.page_size, config_.num_pages);

  if (!buffer_pool_->initialize()) {
    throw std::runtime_error("Failed to initialize buffer pool");
  }

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Buffer pool initialized, rkey={:#x}",
                   buffer_pool_->get_rkey());
}

void DOCAChannel::init_queue_pair() {
  doca_error_t result;

  qp_ = std::make_unique<DOCAQueuePair>(
      config_.wqe_num, doca_gpu_device_, doca_device_, uar_, verbs_ctx_,
      verbs_pd_, cq_rq_->get(), cq_sq_->get());

  result = qp_->create();
  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to create QP: " +
                             std::string(doca_error_get_descr(result)));
  }

  std::uint32_t qp_number = doca_verbs_qp_get_qpn(qp_->get());
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Created QP with number: {:#x}", qp_number);

  // If remote QP is already configured, connect now
  if (!config_.peer_ip.empty() && config_.remote_qpn != 0) {
    // Convert peer IP to GID
    unsigned long client_ip = 0;
    if (inet_pton(AF_INET, config_.peer_ip.c_str(), &client_ip) != 1) {
      throw std::runtime_error("Unable to convert peer IP: " + config_.peer_ip);
    }

    std::uint64_t client_interface_id = client_ip;
    client_interface_id <<= 32;
    client_interface_id |= 0xFFFF0000;

    doca_verbs_gid doca_rgid;
    std::memset(&doca_rgid, 0, sizeof(doca_rgid));
    std::memcpy(&doca_rgid.raw[8], &client_interface_id, 8);

    result = qp_->connect(doca_rgid, gid_index_, config_.remote_qpn);
    if (result != DOCA_SUCCESS) {
      throw std::runtime_error("Failed to connect QP: " +
                               std::string(doca_error_get_descr(result)));
    }

    connected_ = true;
    NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "QP connected to remote peer");
  }
}

void DOCAChannel::init_exit_flag() {
  doca_error_t result;

  // Create CUDA stream for kernel launch
  cudaError_t result_cuda =
      cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
  if (result_cuda != cudaSuccess) {
    throw std::runtime_error("Failed to create CUDA stream");
  }

  // Allocate exit flag (lines 905-914)
  result = doca_gpu_mem_alloc(doca_gpu_device_, sizeof(std::uint32_t),
                              get_page_size(), DOCA_GPU_MEM_TYPE_GPU_CPU,
                              reinterpret_cast<void **>(&gpu_exit_flag_),
                              reinterpret_cast<void **>(&cpu_exit_flag_));

  if (result != DOCA_SUCCESS || !gpu_exit_flag_ || !cpu_exit_flag_) {
    throw std::runtime_error("Failed to allocate exit flag: " +
                             std::string(doca_error_get_descr(result)));
  }

  *cpu_exit_flag_ = 0;
}

void DOCAChannel::cleanup() {
  // DOCA resource cleanup requires careful ordering:
  // 1. Signal exit and wait for GPU kernel to stop
  // 2. Free GPU exit flag
  // 3. Destroy QP (releases references to CQs and UMEMs)
  // 4. Destroy CQs (releases references to UMEMs)
  // 5. Destroy buffer pool
  // 6. Destroy GPU device, UAR, PD, verbs context, DOCA device

  signal_exit();

  if (cuda_stream_) {
    cudaStreamSynchronize(cuda_stream_);
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }

  // Ensure all GPU operations are complete before destroying DOCA resources
  // cudaDeviceSynchronize();

  if (gpu_exit_flag_) {
    doca_gpu_mem_free(doca_gpu_device_, gpu_exit_flag_);
    gpu_exit_flag_ = nullptr;
    cpu_exit_flag_ = nullptr;
  }

  // QP must be destroyed before CQs.
  // QP holds references to the CQs, so destroying CQs first causes errors.
  qp_.reset();
  cq_rq_.reset();
  cq_sq_.reset();
  buffer_pool_.reset();

  if (doca_gpu_device_) {
    doca_gpu_destroy(doca_gpu_device_);
    doca_gpu_device_ = nullptr;
  }

  if (uar_) {
    doca_uar_destroy(uar_);
    uar_ = nullptr;
  }

  if (verbs_pd_) {
    doca_verbs_pd_destroy(verbs_pd_);
    verbs_pd_ = nullptr;
  }

  if (verbs_ctx_) {
    doca_verbs_context_destroy(verbs_ctx_);
    verbs_ctx_ = nullptr;
  }

  if (doca_device_) {
    doca_dev_close(doca_device_);
    doca_device_ = nullptr;
  }
}

DOCAConnectionParams DOCAChannel::get_connection_params() const {
  if (!qp_) {
    throw std::runtime_error("QP not initialized");
  }

  DOCAConnectionParams params;
  params.qpn = doca_verbs_qp_get_qpn(qp_->get());
  params.buffer_addr = buffer_pool_->get_external_address();
  params.rkey = buffer_pool_->get_rkey();
  params.num_slots = buffer_pool_->get_num_pages();
  params.slot_size = buffer_pool_->get_page_size();

  ibv_gid_entry ib_gid_entry;
  int ret = ibv_query_gid_ex(ibv_pd_->context, config_.nic_port, gid_index_,
                             &ib_gid_entry, 0);
  if (ret == 0) {
    std::memcpy(params.gid.data(), ib_gid_entry.gid.raw, 16);
  }

  return params;
}

void DOCAChannel::set_remote_qp(
    std::uint32_t remote_qpn, const std::array<std::uint8_t, 16> &remote_gid) {
  doca_verbs_gid doca_rgid;
  std::memcpy(doca_rgid.raw, remote_gid.data(), 16);

  doca_error_t result = qp_->connect(doca_rgid, gid_index_, remote_qpn);
  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to connect to remote QP: " +
                             std::string(doca_error_get_descr(result)));
  }

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Connected to remote QP {:#x}", remote_qpn);

  // CRITICAL: Prepare WQEs AFTER QP is connected
  // This must happen after connect() because the NIC needs the QP in RTR/RTS
  // state to process receive WQEs
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "Preparing receive WQEs after QP connection...");
  result = doca_prepare_wqes(nullptr, // Use default stream
                             qp_->get_gpu_dev(), config_.max_rpc_size,
                             buffer_pool_->get_mkey_be(), config_.wqe_num);

  if (result != DOCA_SUCCESS) {
    throw std::runtime_error("Failed to prepare WQEs after connection: " +
                             std::string(doca_error_get_descr(result)));
  }
  cudaStreamSynchronize(nullptr);
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Receive WQEs prepared successfully");

  connected_ = true;
}

void DOCAChannel::signal_exit() {
  if (cpu_exit_flag_) {
    *cpu_exit_flag_ = 1;
  }
}

Channel::GPUMemoryHandles DOCAChannel::get_gpu_memory_handles() {
  if (!qp_ || !buffer_pool_) {
    throw std::runtime_error("Channel not initialized");
  }

  Channel::GPUMemoryHandles handles{};
  handles.rx_queue_addr = qp_->get_gpu_dev();
  handles.tx_queue_addr = qp_->get_gpu_dev(); // Same QP for bidirectional
  handles.buffer_pool_addr = buffer_pool_->get_gpu_buffer();
  handles.buffer_pool_size = config_.buffer_size;

  // DOCA-specific fields
  // NOTE: CQ is obtained from QP in kernel via
  // doca_gpu_dev_verbs_qp_get_cq_rq()
  handles.cq_rq_addr = qp_->get_gpu_dev(); // Kernel extracts CQ from QP
  handles.buffer_mkey = buffer_pool_->get_mkey_be();
  // For DOCA_GPU_MEM_TYPE_GPU_CPU memory, we have two pointers:
  // - cpu_exit_flag_: for host writes
  // - gpu_exit_flag_: for kernel reads
  handles.exit_flag = cpu_exit_flag_;
  handles.gpu_exit_flag = gpu_exit_flag_;
  handles.page_size = config_.page_size;
  handles.num_pages = config_.num_pages;

  return handles;
}

// Interface methods (minimal implementation for GPU datapath)
std::uint32_t DOCAChannel::receive_burst(Buffer **buffers, std::uint32_t max) {
  // Not used in GPU datapath - GPU kernel handles receive directly
  return 0;
}

std::uint32_t DOCAChannel::send_burst(Buffer **buffers, std::uint32_t count) {
  // Not used in GPU datapath - GPU kernel handles send directly
  return 0;
}

Buffer *DOCAChannel::acquire_buffer() {
  return buffer_pool_ ? buffer_pool_->acquire() : nullptr;
}

void DOCAChannel::release_buffer(Buffer *buffer) {
  if (buffer_pool_)
    buffer_pool_->release(buffer);
}

void DOCAChannel::register_memory(void *addr, std::size_t size) {
  // Memory registration handled in init_buffer_pool()
}

void DOCAChannel::configure_queues(
    const std::vector<std::uint32_t> &queue_ids) {
  // Queue configuration handled in initialization
}

} // namespace cudaq::nvqlink
