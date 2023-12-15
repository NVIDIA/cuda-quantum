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
  static size_t size() { return 0; }
  static bool serialize(SerializeOutputBuffer &) { return true; }
  static bool deserialize(SerializeInputBuffer &) { return true; }
};

// Non-empty list specialization for SerializeArgs.
template <typename ArgT, typename... ArgTs>
class SerializeArgs<ArgT, ArgTs...> {
public:
  static size_t size(const ArgT &arg, const ArgTs &...args) {
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
  static size_t size(const T &x) { return sizeof(T); }

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
template <class T>
class SerializeArgImpl<std::vector<T>,
                       std::enable_if_t<std::is_trivial<T>::value>> {
public:
  static size_t size(const std::vector<T> &S) {
    size_t Size =
        SerializeArgs<uint64_t>::size(static_cast<uint64_t>(S.size()));
    for (const auto &E : S)
      Size += SerializeArgs<T>::size(E);
    return Size;
  }

  static bool serialize(SerializeOutputBuffer &buf, const std::vector<T> &S) {
    // Size followed by data
    if (!SerializeArgs<uint64_t>::serialize(buf,
                                            static_cast<uint64_t>(S.size())))
      return false;
    for (const auto &E : S)
      if (!SerializeArgs<T>::serialize(buf, E))
        return false;
    return true;
  }

  static bool deserialize(SerializeInputBuffer &buf, std::vector<T> &S) {
    uint64_t Size;
    if (!SerializeArgs<uint64_t>::deserialize(buf, Size))
      return false;
    for (size_t I = 0; I != Size; ++I) {
      T E;
      if (!SerializeArgs<T>::deserialize(buf, E))
        return false;
      S.emplace_back(E);
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
  static void call(CallableT &&H, ArgTupleT &Args, std::index_sequence<I...>) {
    std::forward<CallableT>(H)(std::get<I>(Args)...);
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
  static void invoke(CallableT &&H, const char *ArgData, size_t ArgSize) {
    ArgTuple args;
    // Deserialize buffer to args tuple
    if (!deserialize(ArgData, ArgSize, args, ArgIndices{}))
      throw std::runtime_error(
          "Failed to deserialize arguments for wrapper function call");
    // Call the wrapped function with args tuple
    WrapperFunctionHandlerCaller::call(std::forward<CallableT>(H), args,
                                       ArgIndices{});
  }

private:
  // Helper to deserialize a flat args buffer into typed args tuple.
  template <std::size_t... I>
  static bool deserialize(const char *ArgData, size_t ArgSize, ArgTuple &Args,
                          std::index_sequence<I...>) {
    SerializeInputBuffer buf(ArgData, ArgSize);
    return SerializeArgs<ArgTs...>::deserialize(buf, std::get<I>(Args)...);
  }
};

/// Specialization for class member function
template <typename ClassT, typename... ArgTs>
class WrapperFunctionHandlerHelper<void (ClassT::*)(ArgTs...), ArgTs...>
    : public WrapperFunctionHandlerHelper<void(ArgTs...), ArgTs...> {};

/// Invoke a typed callable (functions) with serialized args.
template <typename CallableT, typename... ArgTs>
void invokeCallableWithSerializedArgs(const char *ArgData, size_t ArgSize,
                                      CallableT &&Handler) {
  WrapperFunctionHandlerHelper<std::remove_reference_t<CallableT>, ArgTs...>::
      invoke(std::forward<CallableT>(Handler), ArgData, ArgSize);
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
      invokeCallableWithSerializedArgs<QuantumKernel, Args...>);
  cudaq::get_platform().launchKernel(cudaq::getKernelName(fn), nullptr,
                                     (void *)serializedArgsBuffer.data(),
                                     serializedArgsBuffer.size(), 0);
#else
  return fn(std::forward<Args>(args)...);
#endif
}
} // namespace cudaq
