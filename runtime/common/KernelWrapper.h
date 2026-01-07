/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/DeviceCodeRegistry.h"
#include "cudaq/platform.h"
#include "cudaq/qis/pauli_word.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

/*! \file KernelWrapper.h
    \brief Utility classes to support kernel launch on remote platforms.

  This header file defines classes supporting launching quantum kernels when
  using the remote platform. It provides serialization utilities for kernel
  arguments. The argument packing (for serialization) is handled by C++
  templates in lieu of bridge-generated functions. These utilities will only be
  used for the remote platform target (guarded by pre-defined macros).
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

// Serialization for `pauli_word`
// Data is packed in the same way as a `vector<char>`.
template <>
class SerializeArgImpl<cudaq::pauli_word> {
public:
  static std::size_t size(const cudaq::pauli_word &x) {
    return SerializeArgImpl<std::vector<char>>::size(x.data());
  }

  static bool serialize(SerializeOutputBuffer &buf,
                        const cudaq::pauli_word &x) {
    return SerializeArgImpl<std::vector<char>>::serialize(buf, x.data());
  }

  static bool deserialize(SerializeInputBuffer &buf, cudaq::pauli_word &x) {
    std::vector<char> pauliStr;
    const bool isVectorCharDeserialized =
        SerializeArgImpl<std::vector<char>>::deserialize(buf, pauliStr);
    if (!isVectorCharDeserialized)
      return false;
    x = cudaq::pauli_word(std::string(pauliStr.begin(), pauliStr.end()));
    return true;
  }
};

// Serialization for a vector of vectors
// Note: we don't support recursively nested vectors (> 2 levels) at the moment.
template <class T>
class SerializeArgImpl<std::vector<std::vector<T>>,
                       std::enable_if_t<std::is_trivial<T>::value>> {
public:
  static std::size_t size(const std::vector<std::vector<T>> &vec) {
    std::size_t size =
        SerializeArgs<uint64_t>::size(static_cast<uint64_t>(vec.size()));
    for (const auto &el : vec) {
      size += SerializeArgs<std::vector<T>>::size(el);
    }
    return size;
  }

  static bool serialize(SerializeOutputBuffer &buf,
                        const std::vector<std::vector<T>> &vec) {
    // Top-level size followed by size data of all sub-vectors then their buffer
    // data.
    if (!SerializeArgs<uint64_t>::serialize(buf,
                                            static_cast<uint64_t>(vec.size())))
      return false;
    for (const auto &elem : vec) {
      if (!SerializeArgs<uint64_t>::serialize(
              buf, static_cast<uint64_t>(elem.size())))
        return false;
    }

    for (const auto &subVec : vec) {
      for (const auto &elem : subVec)
        if (!SerializeArgs<T>::serialize(buf, elem))
          return false;
    }
    return true;
  }

  static bool deserialize(SerializeInputBuffer &buf,
                          std::vector<std::vector<T>> &vec) {
    uint64_t size = 0;
    vec.clear();
    if (!SerializeArgs<uint64_t>::deserialize(buf, size))
      return false;

    vec.reserve(size);

    for (std::size_t i = 0; i < size; ++i) {
      uint64_t subVecSizeBytes = 0;
      if (!SerializeArgs<uint64_t>::deserialize(buf, subVecSizeBytes))
        return false;
      vec.emplace_back(std::vector<T>(subVecSizeBytes));
    }
    for (std::size_t i = 0; i < size; ++i) {
      auto &subVec = vec[i];
      for (auto &el : subVec) {
        if (!SerializeArgs<T>::deserialize(buf, el))
          return false;
      }
    }
    return true;
  }
};

// Serialization for `std::vector<pauli_word>`
template <>
class SerializeArgImpl<std::vector<cudaq::pauli_word>> {
  using ContainerEquivalentTy = std::vector<std::vector<char>>;
  static ContainerEquivalentTy
  paulisToStdEquivalent(const std::vector<cudaq::pauli_word> &x) {
    ContainerEquivalentTy equiv;
    std::transform(x.begin(), x.end(), std::back_inserter(equiv),
                   [](const cudaq::pauli_word &p) -> std::vector<char> {
                     return p.data();
                   });
    return equiv;
  }
  static std::vector<cudaq::pauli_word>
  fromStdEquivalentToPaulis(const ContainerEquivalentTy &x) {
    std::vector<cudaq::pauli_word> paulis;
    std::transform(x.begin(), x.end(), std::back_inserter(paulis),
                   [](const std::vector<char> &pauliStr) -> cudaq::pauli_word {
                     return cudaq::pauli_word(
                         std::string(pauliStr.begin(), pauliStr.end()));
                   });
    return paulis;
  }

public:
  static std::size_t size(const std::vector<cudaq::pauli_word> &x) {
    return SerializeArgImpl<ContainerEquivalentTy>::size(
        paulisToStdEquivalent(x));
  }

  static bool serialize(SerializeOutputBuffer &buf,
                        const std::vector<cudaq::pauli_word> &x) {
    return SerializeArgImpl<ContainerEquivalentTy>::serialize(
        buf, paulisToStdEquivalent(x));
  }

  static bool deserialize(SerializeInputBuffer &buf,
                          std::vector<cudaq::pauli_word> &x) {
    ContainerEquivalentTy pauliStrs;
    const bool isVectorOfPauliStrsDeserialized =
        SerializeArgImpl<ContainerEquivalentTy>::deserialize(buf, pauliStrs);
    if (!isVectorOfPauliStrsDeserialized)
      return false;
    x = fromStdEquivalentToPaulis(pauliStrs);
    return true;
  }
};

//===----------------------------------------------------------------------===//
//
// Utilities to check `SerializeArgImpl` exists
// We use this to customize the error message.
//
//===----------------------------------------------------------------------===//
namespace internal {
// We utilize the fact that an incomplete type doesn't support sizeof.
template <class T, std::size_t = sizeof(T)>
std::true_type __has_complete_impl(T *);
std::false_type __has_complete_impl(...);
template <typename ArgT>
// Check whether SerializeArgImpl is defined for this argument type.
using isSerializable =
    decltype(__has_complete_impl(std::declval<SerializeArgImpl<ArgT> *>()));
} // namespace internal

// Serialize a list of args into a flat buffer.
template <typename... Args>
std::vector<char> serializeArgs(const Args &...args) {
  static_assert(std::conjunction_v<internal::isSerializable<Args>...>,
                "Argument type can't be serialized.");

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
  using ArgIndicesPlus1 =
      std::make_index_sequence<1 + std::tuple_size<ArgTuple>::value>;

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

  // Specialization when the 1st std::vector<double> argument has been excluded
  // from the serialized args, but now you want to call it.
  template <typename CallableT>
  static void invoke(CallableT &&func, const std::vector<double> &vec_parms,
                     const char *argData, std::size_t argSize) {
    ArgTuple argsTuple;
    // Deserialize buffer to args tuple
    if (!deserialize(argData, argSize, argsTuple, ArgIndices{}))
      throw std::runtime_error(
          "Failed to deserialize arguments for wrapper function call");
    // Call the wrapped function with args tuple
    auto newArgsTuple = std::tuple_cat(std::make_tuple(vec_parms), argsTuple);
    WrapperFunctionHandlerCaller::call(std::forward<CallableT>(func),
                                       newArgsTuple, ArgIndicesPlus1{});
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

// Invoke a typed callable (functions) with a std::vec<double> + serialized
// `args`.
template <typename CallableT, typename... InvokeArgTs>
void invokeCallableWithSerializedArgs_vec(const std::vector<double> &vec_parms,
                                          const char *argData,
                                          std::size_t argSize,
                                          CallableT &&func) {
  WrapperFunctionHandlerHelper<
      std::remove_reference_t<CallableT>,
      InvokeArgTs...>::invoke(std::forward<CallableT>(func), vec_parms, argData,
                              argSize);
}

} // namespace cudaq
