/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "cudaq/platform.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
namespace cudaq {

/// @brief Wrapper for kernel address
///
/// This automatically handles kernels defined as struct/class or free
/// functions.
class KernelAddress {
public:
  KernelAddress() = default;
  explicit KernelAddress(uint64_t address) : m_addr(address) {}

  template <typename T,
            typename = std::enable_if_t<!std::is_class_v<std::decay_t<T>>>>
  static KernelAddress fromKernel(T *ptr) {
    return KernelAddress(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
  }

  template <typename T,
            typename = std::enable_if_t<std::is_class_v<std::decay_t<T>>>>
  static KernelAddress fromKernel(T &&ref) {
    return KernelAddress(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&ref)));
  }

  template <typename T>
  std::enable_if_t<std::is_pointer<T>::value, T> toPtr() const {
    return reinterpret_cast<T>(static_cast<uintptr_t>(m_addr));
  }

private:
  uint64_t m_addr = 0;
};

/// @brief Fixed size buffer to write to
class SerializeOutputBuffer {
public:
  SerializeOutputBuffer(char *buffer, std::size_t remaining)
      : m_buffer(buffer), m_remaining(remaining) {}
  bool write(const char *data, std::size_t size) {
    if (size > m_remaining)
      return false;
    std::memcpy(m_buffer, data, size);
    m_buffer += size;
    m_remaining -= size;
    return true;
  }

private:
  char *m_buffer = nullptr;
  std::size_t m_remaining = 0;
};

/// @brief Fixed size buffer to read from
class SerializeInputBuffer {
public:
  SerializeInputBuffer() = default;
  SerializeInputBuffer(const char *buffer, std::size_t remaining)
      : m_buffer(buffer), m_remaining(remaining) {}
  bool read(char *data, std::size_t size) {
    if (size > m_remaining)
      return false;
    std::memcpy(data, m_buffer, size);
    m_buffer += size;
    m_remaining -= size;
    return true;
  }

  const char *data() const { return m_buffer; }

private:
  const char *m_buffer = nullptr;
  std::size_t m_remaining = 0;
};

/// Specialize to describe how to serialize/deserialize to/from the given
/// concrete type.
template <typename T, typename _ = void>
class SerializeArgImpl;

/// A utility class for serializing kernel `args` as a flat buffer
template <typename... ArgTs>
class SerializeArgs;

// Empty list specialization
template <>
class SerializeArgs<> {
public:
  static std::size_t size() { return 0; }
  static bool serialize(SerializeOutputBuffer &) { return true; }
  static bool deserialize(SerializeInputBuffer &) { return true; }
};

// Non-empty list specialization for SerializeArgs.
template <typename ArgT, typename... ArgTs>
class SerializeArgs<ArgT, ArgTs...> {
public:
  static std::size_t size(const ArgT &arg, const ArgTs &...args) {
    return SerializeArgImpl<ArgT>::size(arg) +
           SerializeArgs<ArgTs...>::size(args...);
  }

  static bool serialize(SerializeOutputBuffer &buf, const ArgT &arg,
                        const ArgTs &...args) {
    return SerializeArgImpl<ArgT>::serialize(buf, arg) &&
           SerializeArgs<ArgTs...>::serialize(buf, args...);
  }

  static bool deserialize(SerializeInputBuffer &buf, ArgT &arg,
                          ArgTs &...args) {
    return SerializeArgImpl<ArgT>::deserialize(buf, arg) &&
           SerializeArgs<ArgTs...>::deserialize(buf, args...);
  }
};

/// Serialization for 'trivial' types (POD)
template <typename T>
class SerializeArgImpl<T, std::enable_if_t<std::is_trivial<T>::value>> {
public:
  static std::size_t size(const T &x) { return sizeof(T); }

  static bool serialize(SerializeOutputBuffer &buf, const T &x) {
    T tempVal = x;
    return buf.write(reinterpret_cast<const char *>(&tempVal), sizeof(tempVal));
  }

  static bool deserialize(SerializeInputBuffer &buf, T &x) {
    T tempVal;
    if (!buf.read(reinterpret_cast<char *>(&tempVal), sizeof(tempVal)))
      return false;
    x = tempVal;
    return true;
  }
};

// Serialization for a vector of 'trivial' types (POD), e.g.,
// std::vector<double>
// Format: vector size followed by data
template <class T>
class SerializeArgImpl<std::vector<T>,
                       std::enable_if_t<std::is_trivial<T>::value>> {
public:
  static std::size_t size(const std::vector<T> &vec) {
    const std::size_t size =
        SerializeArgs<uint64_t>::size(static_cast<uint64_t>(vec.size())) +
        (vec.empty() ? 0 : vec.size() * SerializeArgs<T>::size(vec.front()));
    return size;
  }

  static bool serialize(SerializeOutputBuffer &buf, const std::vector<T> &vec) {
    // Size followed by data
    if (!SerializeArgs<uint64_t>::serialize(buf,
                                            static_cast<uint64_t>(vec.size())))
      return false;
    for (const auto &elem : vec)
      if (!SerializeArgs<T>::serialize(buf, elem))
        return false;
    return true;
  }

  static bool deserialize(SerializeInputBuffer &buf, std::vector<T> &vec) {
    uint64_t size;
    vec.clear();
    if (!SerializeArgs<uint64_t>::deserialize(buf, size))
      return false;
    for (std::size_t i = 0; i < size; ++i) {
      T elem;
      if (!SerializeArgs<T>::deserialize(buf, elem))
        return false;
      vec.emplace_back(elem);
    }
    return true;
  }
};

template <typename... Args>
std::vector<char> serializeArgs(const Args &...args) {
  using Serializer = SerializeArgs<Args...>;
  std::vector<char> serializedArgs;
  serializedArgs.resize(Serializer::size(args...));
  SerializeOutputBuffer buf(serializedArgs.data(), serializedArgs.size());
  if (!Serializer::serialize(buf, args...))
    throw std::runtime_error("Unable to serialize kernel arguments!");
  return serializedArgs;
}

// Helper to invoke a callable with a deserialized args tuple
class WrapperFunctionHandlerCaller {
public:
  template <typename CallableT, typename ArgTupleT, std::size_t... I>
  static void call(CallableT &&func, ArgTupleT &args,
                   std::index_sequence<I...>) {
    std::forward<CallableT>(func)(std::get<I>(args)...);
  }
};

template <typename WrapperFunctionImplT, typename... ArgTs>
class WrapperFunctionHandlerHelper
    : public WrapperFunctionHandlerHelper<
          decltype(&std::remove_reference_t<WrapperFunctionImplT>::operator()),
          ArgTs...> {};

template <typename... ArgTs>
class WrapperFunctionHandlerHelper<void(ArgTs...), ArgTs...> {
public:
  using ArgTuple = std::tuple<std::decay_t<ArgTs>...>;
  using ArgIndices = std::make_index_sequence<std::tuple_size<ArgTuple>::value>;

  template <typename CallableT>
  static void invoke(CallableT &&func, const char *argData,
                     std::size_t argSize) {
    ArgTuple argsTuple;
    // Deserialize buffer to args tuple
    if (!deserialize(argData, argSize, argsTuple, ArgIndices{}))
      throw std::runtime_error(
          "Failed to deserialize arguments for wrapper function call");
    // Call the wrapped function with args tuple
    WrapperFunctionHandlerCaller::call(std::forward<CallableT>(func), argsTuple,
                                       ArgIndices{});
  }

private:
  // Helper to deserialize a flat args buffer into typed args tuple.
  template <std::size_t... I>
  static bool deserialize(const char *argData, std::size_t argSize,
                          ArgTuple &argsTuple, std::index_sequence<I...>) {
    SerializeInputBuffer buf(argData, argSize);
    return SerializeArgs<ArgTs...>::deserialize(buf, std::get<I>(argsTuple)...);
  }
};

/// Specialization for class member function
template <typename ClassT, typename... ArgTs>
class WrapperFunctionHandlerHelper<void (ClassT::*)(ArgTs...), ArgTs...>
    : public WrapperFunctionHandlerHelper<void(ArgTs...), ArgTs...> {};

/// Invoke a typed callable (functions) with serialized `args`.
template <typename CallableT, typename... ArgTs>
void invokeCallableWithSerializedArgs(const char *argData, std::size_t argSize,
                                      CallableT &&func) {
  WrapperFunctionHandlerHelper<std::remove_reference_t<CallableT>,
                               ArgTs...>::invoke(std::forward<CallableT>(func),
                                                 argData, argSize);
}

template <typename QuantumKernel, typename... Args>
std::invoke_result_t<QuantumKernel, Args...> invokeKernel(QuantumKernel &&fn,
                                                          Args &&...args) {
#if defined(CUDAQ_REMOTE_SIM) && defined(CUDAQ_LIBRARY_MODE)
  // In library mode, to use the remote simulator platform, we need to pack the
  // argument and delegate to the platform's launchKernel rather than invoking
  // the kernel function directly.
  auto serializedArgsBuffer = serializeArgs(std::forward<Args>(args)...);
  // Note: we explicitly instantiate this wrapper so that the symbol is present
  // in the IR.
  auto *wrappedKernel = reinterpret_cast<void (*)(void *)>(
      invokeCallableWithSerializedArgs<QuantumKernel, std::decay_t<Args>...>);
  cudaq::get_platform().launchKernel(cudaq::getKernelName(fn), nullptr,
                                     (void *)serializedArgsBuffer.data(),
                                     serializedArgsBuffer.size(), 0);
#else
  return fn(std::forward<Args>(args)...);
#endif
}
} // namespace cudaq
