/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform.h"
#include "cudaq/utils/registry.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

/*! \file KernelWrapper.h
    \brief Utility classes to support library-mode kernel launch.

  This header file defines classes supporting launching quantum kernels in
  library mode (without the bridge) when using the remote platform.
  Specifically, it provides a layer of indirection, `cudaq::invokeKernel`,
  similar to the `altLaunchKernel` function when using the bridge. In addition,
  argument packing (for serialization) is handled by C++ templates in lieu of
  bridge-generated functions. These utilities will only be used for the remote
  platform target in library-mode (guarded by pre-defined macros).
*/

namespace cudaq {
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

//===----------------------------------------------------------------------===//
//
// Utilities to package (serialize) kernel arguments as a flat byte buffer.
// This is equivalent to the bridge-generated args creator function for each
// kernel signature type.
//
//===----------------------------------------------------------------------===//

// Specialize to describe how to serialize/deserialize to/from the given
// concrete type.
template <typename T, typename _ = void>
class SerializeArgImpl;

// A utility class for serializing kernel `args` as a flat buffer
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

namespace __internal {
// Member detection idiom to detect if this is a class with `operator()`
// invocation with any signature.
// https://en.wikibooks.org/wiki/More_C++_Idioms/Member_Detector
template <typename T>
struct isCallableClassObj {
private:
  typedef char (&yes)[1];
  typedef char (&no)[2];

  struct Fallback {
    void operator()();
  };
  struct Derived : T, Fallback {};

  template <typename U, U>
  struct Check;

  template <typename>
  static yes test(...);

  template <typename C>
  static no test(Check<void (Fallback::*)(), &C::operator()> *);

public:
  static const bool value = sizeof(test<Derived>(0)) == sizeof(yes);
};

// Check if this a callable: free functions or invocable class objects (e.g.,
// lambdas, class with `operator()`)
template <typename T>
struct isCallable
    : std::conditional<
          std::is_pointer_v<T>, std::is_function<std::remove_pointer_t<T>>,
          typename std::conditional<std::is_class_v<T>, isCallableClassObj<T>,
                                    std::false_type>::type>::type {};

} // namespace __internal

// Non-empty list specialization for SerializeArgs.
template <typename ArgT, typename... ArgTs>
class SerializeArgs<ArgT, ArgTs...> {
public:
  static std::size_t size(const ArgT &arg, const ArgTs &...args) {
    static_assert(!__internal::isCallable<ArgT>::value,
                  "Callable entry-point kernel arguments are not supported for "
                  "the remote simulator platform in library mode. Please "
                  "rewrite the entry point kernel or use MLIR mode.");
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

// Serialization for 'trivial' types (POD)
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
// `std::vector<double>`.
// Format: vector size followed by data
// Note: vectors of non-POD types are not supported by this library-mode
// serializer.
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
    if (!SerializeArgs<uint64_t>::serialize(
            buf, static_cast<uint64_t>(vec.size() * sizeof(T))))
      return false;
    for (const auto &elem : vec)
      if (!SerializeArgs<T>::serialize(buf, elem))
        return false;
    return true;
  }

  static bool deserialize(SerializeInputBuffer &buf, std::vector<T> &vec) {
    uint64_t size = 0;
    vec.clear();
    if (!SerializeArgs<uint64_t>::deserialize(buf, size))
      return false;
    size /= sizeof(T);

    vec.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
      T elem;
      if (!SerializeArgs<T>::deserialize(buf, elem))
        return false;
      vec.emplace_back(elem);
    }
    return true;
  }
};

// Serialize a list of args into a flat buffer.
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

//===----------------------------------------------------------------------===//
//
// Utilities to wrap a kernel call `f(args...)` as a generic one that takes a
// single flat buffer argument. The buffer is deserialized into a typed
// `std::tuple` for invocation.
//
//===----------------------------------------------------------------------===//
/// @cond
template <typename WrapperFunctionImplT, typename... ArgTs>
class WrapperFunctionHandlerHelper
    : public WrapperFunctionHandlerHelper<
          decltype(&std::remove_reference_t<WrapperFunctionImplT>::operator()),
          ArgTs...> {};

template <typename... SignatureArgTs, typename... InvokeArgTs>
class WrapperFunctionHandlerHelper<void(SignatureArgTs...), InvokeArgTs...> {
public:
  using ArgTuple = std::tuple<std::decay_t<InvokeArgTs>...>;
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
    return SerializeArgs<InvokeArgTs...>::deserialize(
        buf, std::get<I>(argsTuple)...);
  }
};

template <typename... SignatureArgTs, typename... InvokeArgTs>
class WrapperFunctionHandlerHelper<void (*)(SignatureArgTs...), InvokeArgTs...>
    : public WrapperFunctionHandlerHelper<void(SignatureArgTs...),
                                          InvokeArgTs...> {};

// Specialization for class member function
// Non-const member function
template <typename ClassT, typename... SignatureArgTs, typename... InvokeArgTs>
class WrapperFunctionHandlerHelper<void (ClassT::*)(SignatureArgTs...),
                                   InvokeArgTs...>
    : public WrapperFunctionHandlerHelper<void(SignatureArgTs...),
                                          InvokeArgTs...> {};
// Const member function (e.g., lambda)
template <typename ClassT, typename... SignatureArgTs, typename... InvokeArgTs>
class WrapperFunctionHandlerHelper<void (ClassT::*)(SignatureArgTs...) const,
                                   InvokeArgTs...>
    : public WrapperFunctionHandlerHelper<void(SignatureArgTs...),
                                          InvokeArgTs...> {};
/// @endcond

// Invoke a typed callable (functions) with serialized `args`.
template <typename CallableT, typename... InvokeArgTs>
void invokeCallableWithSerializedArgs(const char *argData, std::size_t argSize,
                                      CallableT &&func) {
  WrapperFunctionHandlerHelper<
      std::remove_reference_t<CallableT>,
      InvokeArgTs...>::invoke(std::forward<CallableT>(func), argData, argSize);
}

// Wrapper for quantum kernel invocation, i.e., `kernel(args...)`.
// In library mode, if the remote platform is used, we redirect it to the
// platform's `launchKernel` instead of invoking it.
template <typename QuantumKernel, typename... Args>
std::invoke_result_t<QuantumKernel, Args...> invokeKernel(QuantumKernel &&fn,
                                                          Args &&...args) {
#if defined(CUDAQ_REMOTE_SIM) && defined(CUDAQ_LIBRARY_MODE)
  if constexpr (has_name<QuantumKernel>::value) {
    // kernel_builder kernel: it always has quake representation; hence, no need
    // to wrap the kernel (run as MLIR mode).
    // Just need to JIT code to get it registered.
    static_cast<cudaq::details::kernel_builder_base &>(fn).jitCode();
    auto serializedArgsBuffer = serializeArgs(std::forward<Args>(args)...);
    cudaq::get_platform().launchKernel(fn.name(), nullptr,
                                       (void *)serializedArgsBuffer.data(),
                                       serializedArgsBuffer.size(), 0);
  } else {
    // In library mode, to use the remote simulator platform, we need to pack
    // the argument and delegate to the platform's launchKernel rather than
    // invoking the kernel function directly.
    auto serializedArgsBuffer = serializeArgs(std::forward<Args>(args)...);
    // Note: we explicitly instantiate this wrapper so that the symbol is
    // present in the IR.
    auto *wrappedKernel = reinterpret_cast<void (*)(void *)>(
        invokeCallableWithSerializedArgs<QuantumKernel, std::decay_t<Args>...>);

    // In the remote execution mode, we don't need the function pointer.
    // For raw function pointers, i.e., kernels described as free functions, we
    // send on the function pointer to the platform to retrieve the symbol name
    // since the typeid of a function only contains signature info.
    if constexpr (std::is_class_v<std::decay_t<QuantumKernel>>)
      cudaq::get_platform().launchKernel(cudaq::getKernelName(fn), nullptr,
                                         (void *)serializedArgsBuffer.data(),
                                         serializedArgsBuffer.size(), 0);
    else
      cudaq::get_platform().launchKernel(
          cudaq::getKernelName(fn), reinterpret_cast<void (*)(void *)>(&fn),
          (void *)serializedArgsBuffer.data(), serializedArgsBuffer.size(), 0);
  }
#else
  return fn(std::forward<Args>(args)...);
#endif
}

// `invokeKernel` specialization for `std::function` wrapped quantum kernels.
// If a `__qpu__` kernel is wrapped inside a function, to make it worked with
// `CUDAQ_REMOTE_SIM` in **library** mode, the `__qpu__` kernel invocation
// should already have been wrapped with `invokeKernel`. Hence, we just invoke
// it. For example,
// ```
// __qpu__ void kernel() {}
// auto wrapped = [&](){invokeKernel(kernel);};
// cudaq::observe(H, wrapped);
// ```
// Since `cudaq::observe` again uses `invokeKernel` to execute the callable (a
// `std::function`), we want to pass it through.
template <typename Signature, typename... Args>
std::invoke_result_t<std::function<Signature>, Args...>
invokeKernel(std::function<Signature> &fn, Args &&...args) {
  return fn(std::forward<Args>(args)...);
}
} // namespace cudaq
