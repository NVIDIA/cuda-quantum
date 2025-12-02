/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <arpa/inet.h>
#include <cstring>
#include <fstream>
#include <ifaddrs.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <nlohmann/json.hpp>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

// Ring buffer layout (must match server)
constexpr size_t RING_HEADER_SIZE = 128;
constexpr size_t SLOT_HEADER_SIZE = 16;

// RPC Function IDs
constexpr uint32_t FUNCTION_ECHO = 1;
constexpr uint32_t FUNCTION_ADD = 2;
constexpr uint32_t FUNCTION_MULTIPLY = 3;

/**
 * Base class for RoCE clients with common IB Verbs setup
 */
class RoCEClientBase {
public:
  RoCEClientBase(const std::string &device_name) : device_name_(device_name) {}

  virtual ~RoCEClientBase() { cleanup(); }

  bool initialize() {
    std::cout << "\n[INIT] Initializing InfiniBand Verbs..." << std::endl;

    // Get device list
    int num_devices;
    dev_list_ = ibv_get_device_list(&num_devices);
    if (!dev_list_ || num_devices == 0) {
      std::cerr << "No InfiniBand devices found" << std::endl;
      return false;
    }

    // Find the specified device
    ibv_device *device = nullptr;
    for (int i = 0; i < num_devices; i++) {
      if (device_name_ == ibv_get_device_name(dev_list_[i])) {
        device = dev_list_[i];
        break;
      }
    }

    if (!device) {
      std::cerr << "Device " << device_name_ << " not found" << std::endl;
      return false;
    }

    // Open device
    context_ = ibv_open_device(device);
    if (!context_) {
      std::cerr << "Failed to open device" << std::endl;
      return false;
    }

    // Query port attributes
    if (ibv_query_port(context_, port_num_, &port_attr_) != 0) {
      std::cerr << "Failed to query port" << std::endl;
      return false;
    }

    // Allocate protection domain
    pd_ = ibv_alloc_pd(context_);
    if (!pd_) {
      std::cerr << "Failed to allocate protection domain" << std::endl;
      return false;
    }

    // Create completion queues
    send_cq_ = ibv_create_cq(context_, 16, nullptr, nullptr, 0);
    recv_cq_ = ibv_create_cq(context_, 16, nullptr, nullptr, 0);
    if (!send_cq_ || !recv_cq_) {
      std::cerr << "Failed to create completion queues" << std::endl;
      return false;
    }

    // Get local GID
    if (ibv_query_gid(context_, port_num_, gid_index_, &local_gid_) != 0) {
      std::cerr << "Failed to query GID" << std::endl;
      return false;
    }

    // Allocate receive buffers
    recv_buffer_size_ = 4096;
    recv_buffer_ = malloc(recv_buffer_size_);
    if (!recv_buffer_) {
      std::cerr << "Failed to allocate receive buffer" << std::endl;
      return false;
    }

    recv_mr_ = ibv_reg_mr(pd_, recv_buffer_, recv_buffer_size_,
                          IBV_ACCESS_LOCAL_WRITE);
    if (!recv_mr_) {
      std::cerr << "Failed to register receive buffer" << std::endl;
      return false;
    }

    std::cout << "✓ InfiniBand Verbs initialized" << std::endl;
    std::cout << "  Device: " << device_name_ << std::endl;
    std::cout << "  Local GID: " << gid_to_string(local_gid_) << std::endl;

    return true;
  }

protected:
  void cleanup() {
    if (recv_mr_)
      ibv_dereg_mr(recv_mr_);
    if (recv_buffer_)
      free(recv_buffer_);
    if (send_cq_)
      ibv_destroy_cq(send_cq_);
    if (recv_cq_)
      ibv_destroy_cq(recv_cq_);
    if (pd_)
      ibv_dealloc_pd(pd_);
    if (context_)
      ibv_close_device(context_);
    if (dev_list_)
      ibv_free_device_list(dev_list_);
  }

  bool transition_qp_to_rts(struct ibv_qp *qp, uint32_t remote_qpn,
                            const union ibv_gid &remote_gid) {
    // RESET -> INIT
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = port_num_;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;

    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(qp, &attr, mask) != 0) {
      std::cerr << "Failed to transition QP to INIT" << std::endl;
      return false;
    }

    // INIT -> RTR
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;

    // Address vector
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = port_num_;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote_gid;
    attr.ah_attr.grh.sgid_index = gid_index_;
    attr.ah_attr.grh.hop_limit = 64;

    mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
           IBV_QP_RQ_PSN;
    if (ibv_modify_qp(qp, &attr, mask) != 0) {
      std::cerr << "Failed to transition QP to RTR" << std::endl;
      return false;
    }

    // RTR -> RTS
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;

    mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
    if (ibv_modify_qp(qp, &attr, mask) != 0) {
      std::cerr << "Failed to transition QP to RTS" << std::endl;
      return false;
    }

    return true;
  }

  bool post_recv_buffer(struct ibv_qp *qp) {
    struct ibv_sge sge = {};
    sge.addr = (uint64_t)recv_buffer_;
    sge.length = recv_buffer_size_;
    sge.lkey = recv_mr_->lkey;

    struct ibv_recv_wr wr = {};
    wr.wr_id = recv_wr_id_++;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr *bad_wr = nullptr;
    if (ibv_post_recv(qp, &wr, &bad_wr) != 0) {
      std::cerr << "Failed to post receive buffer" << std::endl;
      return false;
    }
    return true;
  }

  std::string gid_to_string(const union ibv_gid &gid) {
    char buf[64];
    inet_ntop(AF_INET6, gid.raw, buf, sizeof(buf));
    return buf;
  }

  union ibv_gid string_to_gid(const std::string &gid_str) {
    union ibv_gid gid;
    inet_pton(AF_INET6, gid_str.c_str(), gid.raw);
    return gid;
  }

  std::string device_name_;
  uint8_t port_num_ = 1;
  uint8_t gid_index_ = 1; // Use GID index 1 for Soft-RoCE

  // IB Verbs resources
  struct ibv_device **dev_list_ = nullptr;
  struct ibv_context *context_ = nullptr;
  struct ibv_pd *pd_ = nullptr;
  struct ibv_cq *send_cq_ = nullptr;
  struct ibv_cq *recv_cq_ = nullptr;
  struct ibv_port_attr port_attr_ = {};

  union ibv_gid local_gid_ = {};

  // Receive buffers
  void *recv_buffer_ = nullptr;
  size_t recv_buffer_size_ = 0;
  struct ibv_mr *recv_mr_ = nullptr;

  // Work request IDs
  uint64_t send_wr_id_ = 0;
  uint64_t recv_wr_id_ = 0;
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

inline std::string get_interface_name(const std::string &roce_device) {
  // Map RoCE device to network interface by reading sysfs parent file
  std::string sysfs_path = "/sys/class/infiniband/" + roce_device + "/parent";

  std::ifstream parent_file(sysfs_path);
  if (parent_file.is_open()) {
    std::string iface_name;
    std::getline(parent_file, iface_name);
    parent_file.close();

    // Trim whitespace
    iface_name.erase(0, iface_name.find_first_not_of(" \t\r\n"));
    iface_name.erase(iface_name.find_last_not_of(" \t\r\n") + 1);

    if (!iface_name.empty()) {
      return iface_name;
    }
  }

  return "unknown";
}

inline std::string get_interface_ip(const std::string &iface_name) {
  struct ifaddrs *ifaddr, *ifa;
  std::string ip;

  if (getifaddrs(&ifaddr) == -1) {
    return ip;
  }

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr)
      continue;

    if (ifa->ifa_addr->sa_family == AF_INET &&
        std::string(ifa->ifa_name) == iface_name) {
      char addr_buf[INET_ADDRSTRLEN];
      struct sockaddr_in *addr = (struct sockaddr_in *)ifa->ifa_addr;
      inet_ntop(AF_INET, &addr->sin_addr, addr_buf, INET_ADDRSTRLEN);
      ip = addr_buf;
      break;
    }
  }

  freeifaddrs(ifaddr);
  return ip;
}

inline std::string get_primary_ip() {
  // Find the first non-loopback, non-docker IPv4 address
  struct ifaddrs *ifaddr, *ifa;
  std::string ip;

  if (getifaddrs(&ifaddr) == -1) {
    return ip;
  }

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr)
      continue;

    if (ifa->ifa_addr->sa_family == AF_INET) {
      std::string iface_name = ifa->ifa_name;

      // Skip loopback and docker interfaces
      if (iface_name == "lo" || iface_name.find("docker") == 0) {
        continue;
      }

      char addr_buf[INET_ADDRSTRLEN];
      struct sockaddr_in *addr = (struct sockaddr_in *)ifa->ifa_addr;
      inet_ntop(AF_INET, &addr->sin_addr, addr_buf, INET_ADDRSTRLEN);
      ip = addr_buf;
      break; // Use the first valid IP found
    }
  }

  freeifaddrs(ifaddr);
  return ip;
}

inline std::string get_interface_mac(const std::string &iface_name) {
  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) {
    return "unknown";
  }

  struct ifreq ifr;
  strncpy(ifr.ifr_name, iface_name.c_str(), IFNAMSIZ - 1);

  if (ioctl(sock, SIOCGIFHWADDR, &ifr) < 0) {
    close(sock);
    return "unknown";
  }

  close(sock);

  unsigned char *mac = (unsigned char *)ifr.ifr_hwaddr.sa_data;
  char mac_str[18];
  snprintf(mac_str, sizeof(mac_str), "%02x:%02x:%02x:%02x:%02x:%02x", mac[0],
           mac[1], mac[2], mac[3], mac[4], mac[5]);

  return mac_str;
}
