/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cudaq {

#if __cplusplus >= 202002L
template <typename A>
using nvqlink_remove_cvref_t = std::remove_cvref_t<A>;
#else
template <typename A>
using nvqlink_remove_cvref_t = std::remove_cv_t<std::remove_reference_t<A>>;
#endif

namespace nvqlink {

using handle = std::size_t;

static inline std::size_t device_counter = 0;

/// @brief Represents a pointer to memory allocated on a quantum processing
/// unit (controller or classical device)
/// @details Encapsulates device memory management details including location
/// and size. Supports both device-allocated memory and host memory references.
struct device_ptr {

  /// @brief Opaque handle to device memory block
  /// @details Used as an identifier for the memory allocation. Set to max value
  /// when invalid.
  std::size_t handle = std::numeric_limits<std::size_t>::max();

  /// @brief Size of allocated memory in bytes
  /// @details Zero indicates no allocation or null pointer state.
  std::size_t size = 0;

  /// @brief Physical device identifier
  /// @details Identifies which quantum processing unit or classical device owns
  /// this memory. Set to max value when unassigned.
  std::size_t deviceId = std::numeric_limits<std::size_t>::max();

  /// @brief Pointer to host memory when referencing local memory
  /// @details Non-null when this device_ptr wraps a host memory address rather
  /// than device-allocated memory. Null for true device allocations.
  void *host_value = nullptr;

  /// @brief Default constructor
  /// @details Creates an invalid device_ptr with default sentinel values.
  device_ptr() = default;

  /// @brief Copy constructor (const version)
  /// @param other The device_ptr to copy from
  device_ptr(const device_ptr &) = default;

  /// @brief Copy constructor (non-const version)
  /// @param other The device_ptr to copy from
  device_ptr(device_ptr &) = default;

  /// @brief Constructor with handle and size
  /// @param hdl Opaque handle to the device memory block
  /// @param s Size of the allocated memory in bytes
  device_ptr(std::size_t hdl, std::size_t s) : handle(hdl), size(s) {}

  /// @brief Constructor with handle, size, and device ID
  /// @param hdl Opaque handle to the device memory block
  /// @param s Size of the allocated memory in bytes
  /// @param dev Physical device identifier
  device_ptr(std::size_t hdl, std::size_t s, std::size_t dev)
      : handle(hdl), size(s), deviceId(dev) {}

  /// @brief Template constructor for host memory references
  /// @tparam T Type of the host object being referenced
  /// @param t Pointer to host memory object
  /// @details Creates a device_ptr that wraps a host memory address.
  /// Sets handle to the reinterpreted pointer value and size to sizeof(T).
  template <typename T>
  device_ptr(T *t) : host_value(t) {
    handle = reinterpret_cast<uintptr_t>(host_value);
    size = sizeof(T);
  }

  /// @brief Template assignment operator
  /// @tparam T Type of the value being assigned
  /// @param t Value to assign (forwarded)
  /// @return New device_ptr constructed from the assigned value
  /// @details Creates and returns a new device_ptr from the assigned value.
  template <typename T>
  device_ptr operator=(T &&t) {
    return device_ptr(t);
  }

  /// @brief Check if this device_ptr references host memory
  /// @return True if this wraps a host memory address, false if device memory
  bool is_host_value() const { return host_value != nullptr; }

  /// @brief Check if this device_ptr is in a null/empty state
  /// @return True if both size is zero and host_value is null, false otherwise
  bool is_nullptr() const { return size == 0 && host_value == nullptr; }

  /// @brief Assignment operator (non-const reference)
  /// @param o The device_ptr to copy from
  /// @return Reference to this device_ptr after assignment
  /// @details Copies all member values from the source device_ptr.
  device_ptr &operator=(device_ptr &o) {
    handle = o.handle;
    size = o.size;
    deviceId = o.deviceId;
    host_value = o.host_value;
    return *this;
  }

  /// @brief Assignment operator (const reference)
  /// @param o The device_ptr to copy from
  /// @return Reference to this device_ptr after assignment
  /// @details Copies all member values from the source device_ptr.
  device_ptr &operator=(const device_ptr &o) {
    handle = o.handle;
    size = o.size;
    deviceId = o.deviceId;
    host_value = o.host_value;
    return *this;
  }
};

class device {
public:
  std::size_t device_id = 0;

  // A device function has a callback name,
  // and the user can optionally provide a function
  // that takes the device function pointer and
  // the result and arg device ptrs for its invocation.
  struct device_function {
    const std::string name;
    const std::optional<std::function<void(void *, device_ptr &,
                                           const std::vector<device_ptr> &)>>
        unmarshaller;
  };

  // Map device library locations -> {devFuncName1, devFuncName2, ...}
  std::unordered_map<std::string, std::vector<device_function>>
      device_callbacks;

  device() {
    device_id = device_counter;
    device_counter++;
  }

  device(const std::unordered_map<std::string, std::vector<device_function>>
             &devCallbacks)
      : device_callbacks(devCallbacks) {
    device_id = device_counter;
    device_counter++;
  }

  virtual void connect() = 0;
  virtual void disconnect() = 0;
};

template <typename... Traits>
class device_mixin : public device, public Traits... {
public:
  using device::device;

  virtual ~device_mixin() = default;

  void connect() override {
    // by default do nothing
  }
  void disconnect() override {}

  // Provide non-const and const accessors.
  std::size_t get_id() const { return device_id; }
};

} // namespace nvqlink

/// @brief Alias for quantum device memory pointer handle
template <typename T>
struct device_ptr : public nvqlink::device_ptr {
  operator T *();
};
} // namespace cudaq

namespace cudaq::nvqlink {

// DEVICE TRAITS HERE ...

// RTH mediated data marshaling
template <typename Derived>
class explicit_data_marshalling_trait {
public:
  void *resolve_pointer(device_ptr &devPtr) {
    return static_cast<Derived *>(this)->resolve_pointer(devPtr);
  }

  device_ptr malloc(std::size_t size) const {
    return static_cast<Derived *>(this)->malloc(size);
  }

  template <typename... Sizes,
            std::enable_if_t<(std::conjunction_v<std::is_integral<Sizes>...>),
                             int> = 0>
  auto malloc(Sizes... szs) {
    return std::make_tuple(static_cast<Derived *>(this)->malloc(szs)...);
  }

  void free(device_ptr &d) { static_cast<Derived *>(this)->free(d); }

  template <typename... Ptrs,
            typename = std::enable_if_t<
                (std::conjunction_v<std::is_same<
                     std::remove_cv_t<std::remove_reference_t<Ptrs>>,
                     device_ptr>...>)>>
  void free(Ptrs &&...d) {
    (free(d), ...);
  }

  void send(device_ptr &dest, const void *src) {
    static_cast<Derived *>(this)->send(dest, src);
  }

  void recv(void *dest, const device_ptr &src) {
    static_cast<Derived *>(this)->recv(dest, src);
  }
};

// RTH mediated device function callback invocation
template <typename Derived>
class device_callback_trait {
public:
  void launch_callback(const std::string &callbackName,
                       const std::vector<device_ptr> &args) {
    device_ptr null;
    static_cast<Derived *>(this)->launch_callback(callbackName, null, args);
  }
  void launch_callback(const std::string &callbackName, device_ptr &result,
                       const std::vector<device_ptr> &args) {
    static_cast<Derived *>(this)->launch_callback(callbackName, result, args);
  }
};

// QCS Device API
template <typename Derived>
class qcs_trait {
public:
  void upload_program(const std::vector<std::byte> program_data) {
    static_cast<Derived *>(this)->upload_program(program_data);
  }
  void trigger(device_ptr &result, const std::vector<device_ptr> &args) {
    static_cast<Derived *>(this)->trigger(result, args);
  }
};

// fpga_gpu_rdma is a channel that is asynchronous,
// it starts an implicit event loop, it only needs connect / disconnect
template <typename Derived, typename RDMATypeT>
class rdma_trait {
public:
  RDMATypeT &get_rdma_connection_data() {
    return static_cast<const Derived *>(this)->get_rdma_connection_data();
  }
};

namespace detail {

// ===== explicit_data_marshalling_trait detection =====
template <typename T, typename = void>
struct has_data_marshalling_methods : std::false_type {};

template <typename T>
struct has_data_marshalling_methods<
    T, std::void_t<
           // Check for resolve_pointer method
           decltype(std::declval<T>().resolve_pointer(
               std::declval<device_ptr &>())),

           // Check for malloc method with size_t
           decltype(std::declval<T>().malloc(std::declval<std::size_t>())),

           // Check for free method
           decltype(std::declval<T>().free(std::declval<device_ptr &>())),

           // Check for send method
           decltype(std::declval<T>().send(std::declval<device_ptr &>(),
                                           std::declval<const void *>())),

           // Check for recv method
           decltype(std::declval<T>().recv(
               std::declval<void *>(), std::declval<const device_ptr &>()))>>
    : std::true_type {};

// ===== rdma_trait detection =====
template <typename T, typename = void>
struct has_rdma_methods : std::false_type {};

template <typename T>
struct has_rdma_methods<
    T, std::void_t<decltype(std::declval<T>().get_rdma_connection_data())>>
    : std::true_type {};

// ===== device_callback_trait detection =====
template <typename T, typename = void>
struct has_callback_methods : std::false_type {};

template <typename T>
struct has_callback_methods<
    T,
    std::void_t<
        // Check for launch_callback with 2 parameters (no result)
        decltype(std::declval<T>().launch_callback(
            std::declval<const std::string &>(),
            std::declval<const std::vector<device_ptr> &>())),

        // Check for launch_callback with 3 parameters (with result)
        decltype(std::declval<T>().launch_callback(
            std::declval<const std::string &>(), std::declval<device_ptr &>(),
            std::declval<const std::vector<device_ptr> &>()))>>
    : std::true_type {};

// ===== QCS trait detection (from your original code) =====

template <typename T, typename = void>
struct has_qcs_trait : std::false_type {};

template <typename T>
struct has_qcs_trait<
    T, std::void_t<decltype(std::declval<T>().upload_program(
                       std::declval<std::vector<std::byte>>())),
                   decltype(std::declval<T>().trigger(
                       std::declval<device_ptr &>(),
                       std::declval<const std::vector<device_ptr> &>()))>>
    : std::true_type {};
} // namespace detail

// Public interface - these are the ones you'll use
template <typename T>
constexpr bool has_data_marshalling_trait_v =
    detail::has_data_marshalling_methods<nvqlink_remove_cvref_t<T>>::value;

template <typename T>
constexpr bool has_rdma_trait_v =
    detail::has_rdma_methods<nvqlink_remove_cvref_t<T>>::value;

template <typename T>
constexpr bool has_callback_trait_v =
    detail::has_callback_methods<nvqlink_remove_cvref_t<T>>::value;

template <typename T>
constexpr bool has_qcs_trait_v =
    detail::has_qcs_trait<nvqlink_remove_cvref_t<T>>::value;
} // namespace cudaq::nvqlink
