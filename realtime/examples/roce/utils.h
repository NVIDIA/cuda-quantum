/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <arpa/inet.h>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

inline std::string get_interface_name(const std::string &roce_device) {
  // Try multiple methods to map RoCE device to network interface
  
  // Method 1: Read parent file (works for soft-RoCE)
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

  // Method 2: Check device/net/ directory (works for hardware NICs like mlx5)
  std::string net_dir = "/sys/class/infiniband/" + roce_device + "/device/net/";
  DIR *dir = opendir(net_dir.c_str());
  if (dir) {
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string name = entry->d_name;
      if (name != "." && name != "..") {
        closedir(dir);
        return name; // Return first interface found
      }
    }
    closedir(dir);
  }

  // Method 3: Check ports/1/gid_attrs/ndevs/0 (another common location)
  std::string port_path = "/sys/class/infiniband/" + roce_device + 
                          "/ports/1/gid_attrs/ndevs/0";
  std::ifstream port_file(port_path);
  if (port_file.is_open()) {
    std::string iface_name;
    std::getline(port_file, iface_name);
    port_file.close();
    
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
