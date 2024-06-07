/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach-o/dyld.h>
#else
#include <link.h>
#endif
namespace cudaq::__internal__ {

struct CUDAQLibraryData {
  std::string path;
};

#if defined(__APPLE__) && defined(__MACH__)
inline static void getCUDAQLibraryPath(CUDAQLibraryData *data) {
  auto nLibs = _dyld_image_count();
  for (uint32_t i = 0; i < nLibs; i++) {
    auto ptr = _dyld_get_image_name(i);
    std::string libName(ptr);
    if (libName.find("cudaq-common") != std::string::npos) {
      auto casted = static_cast<CUDAQLibraryData *>(data);
      casted->path = std::string(ptr);
    }
  }
}
#else
inline static int getCUDAQLibraryPath(struct dl_phdr_info *info, size_t size,
                                      void *data) {
  std::string libraryName(info->dlpi_name);
  if (libraryName.find("cudaq-common") != std::string::npos) {
    auto casted = static_cast<CUDAQLibraryData *>(data);
    casted->path = std::string(info->dlpi_name);
  }
  return 0;
}
#endif

extern std::string demangle_kernel(const char *);
} // namespace cudaq::__internal__

namespace cudaq {
class spin_op;
class sample_result;

inline static std::string getCUDAQLibraryPath() {
  __internal__::CUDAQLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
  getCUDAQLibraryPath(&data);
#else
  dl_iterate_phdr(__internal__::getCUDAQLibraryPath, &data);
#endif
  return data.path;
}

template <typename T, typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T &&iterable) {
  struct iterator {
    size_t i;
    TIter iter;
    bool operator!=(const iterator &other) const { return iter != other.iter; }
    void operator++() {
      ++i;
      ++iter;
    }
    auto operator*() const { return std::tie(i, *iter); }
  };
  struct iterable_wrapper {
    T iterable;
    auto begin() { return iterator{0, std::begin(iterable)}; }
    auto end() { return iterator{0, std::end(iterable)}; }
  };
  return iterable_wrapper{std::forward<T>(iterable)};
}

namespace detail {

template <class F>
auto make_copyable_function(F &&f) {
  using dF = std::decay_t<F>;
  auto spf = std::make_shared<dF>(std::forward<F>(f));
  return [spf](auto &&...args) -> decltype(auto) {
    return (*spf)(decltype(args)(args)...);
  };
}

template <std::size_t Ofst, class Tuple, std::size_t... I>
constexpr auto slice_impl(Tuple &&t, std::index_sequence<I...>) {
  return std::forward_as_tuple(std::get<I + Ofst>(std::forward<Tuple>(t))...);
}

/// @brief Return the index of a type in a variant
template <typename VariantType, typename T, std::size_t index = 0>
constexpr std::size_t variant_index() {
  static_assert(std::variant_size_v<VariantType> > index,
                "Type not found in variant");
  if constexpr (index == std::variant_size_v<VariantType>) {
    return index;
  } else if constexpr (std::is_same_v<
                           std::variant_alternative_t<index, VariantType>, T>) {
    return index;
  } else {
    return variant_index<VariantType, T, index + 1>();
  }
}
} // namespace detail

template <std::size_t I1, std::size_t I2, class Cont>
constexpr auto tuple_slice(Cont &&t) {
  static_assert(I2 >= I1, "invalid slice");
  static_assert(std::tuple_size<std::decay_t<Cont>>::value >= I2,
                "slice index out of bounds");

  return detail::slice_impl<I1>(std::forward<Cont>(t),
                                std::make_index_sequence<I2 - I1>{});
}

template <typename... Ts>
struct voider {
  using type = void;
};

template <typename T, class = void>
struct has_name : std::false_type {};

template <typename T>
struct has_name<T, typename voider<decltype(std::declval<T>().name())>::type>
    : std::true_type {};

template <typename QuantumKernel>
std::string getKernelName(QuantumKernel &kernel) {
  std::string kernel_name;
  if constexpr (has_name<QuantumKernel>::value) {
    kernel_name = kernel.name();
  } else {
    kernel_name = __internal__::demangle_kernel(typeid(kernel).name());
  }
  return kernel_name;
}

template <typename TupleType, typename FunctionType>
void tuple_for_each(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<
                    typename std::remove_reference<TupleType>::type>::value>) {}
// Utility function for looping over tuple elements
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
// Utility function for looping over tuple elements
void tuple_for_each(TupleType &&t, FunctionType f,
                    std::integral_constant<size_t, I>) {
  f(std::get<I>(t));
  tuple_for_each(std::forward<TupleType>(t), f,
                 std::integral_constant<size_t, I + 1>());
}
// Utility function for looping over tuple elements
template <typename TupleType, typename FunctionType>
void tuple_for_each(TupleType &&t, FunctionType f) {
  tuple_for_each(std::forward<TupleType>(t), f,
                 std::integral_constant<size_t, 0>());
}

template <typename TupleType, typename FunctionType>
void tuple_for_each_with_idx(
    TupleType &&, FunctionType,
    std::integral_constant<
        size_t, std::tuple_size<
                    typename std::remove_reference<TupleType>::type>::value>) {}
// Utility function for looping over tuple elements
template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
// Utility function for looping over tuple elements
void tuple_for_each_with_idx(TupleType &&t, FunctionType f,
                             std::integral_constant<size_t, I>) {
  f(std::get<I>(t), std::integral_constant<size_t, I>());
  tuple_for_each_with_idx(std::forward<TupleType>(t), f,
                          std::integral_constant<size_t, I + 1>());
}
// Utility function for looping over tuple elements
template <typename TupleType, typename FunctionType>
void tuple_for_each_with_idx(TupleType &&t, FunctionType f) {
  tuple_for_each_with_idx(std::forward<TupleType>(t), f,
                          std::integral_constant<size_t, 0>());
}

// Function check if file with given path+name exists
inline bool fileExists(const std::string &name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  }
  return false;
}

// Split a string on the given delimiter
template <class Op>
void split(const std::string_view s, char delim, Op op) {
  std::stringstream ss(s.data());
  for (std::string item; std::getline(ss, item, delim);) {
    *op++ = item;
  }
}

// Split a string on the given delimiter
inline std::vector<std::string> split(const std::string_view s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

// Trim a string on the left
inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
  ltrim(s);
  rtrim(s);
}

std::vector<double> linspace(const double a, const double b, std::size_t size);

// Override the seed if you want repeatably random numbers
std::vector<double> random_vector(const double l_range, const double r_range,
                                  const std::size_t size,
                                  const uint32_t seed = std::random_device{}());

/// @brief Return a vector of integers. The first element is the
/// user-specified `start` value. The remaining values are all values
/// incremented by `step` (defaults to 1) until the `stop` value is reached
/// (exclusive).
#if CUDAQ_USE_STD20
template <typename ElementType>
  requires(std::signed_integral<ElementType>)
#else
template <typename ElementType,
          typename = std::enable_if_t<std::is_integral_v<ElementType> &&
                                      std::is_signed_v<ElementType>>>
#endif
inline std::vector<ElementType> range(ElementType start, ElementType stop,
                                      ElementType step = 1) {
  std::vector<ElementType> vec;
  auto val = start;
  while ((step > 0) ? (val < stop) : (val > stop)) {
    vec.push_back(val);
    val += step;
  }
  return vec;
}

/// @brief Return a vector of integers. The first element is zero, and
/// the remaining elements are all values incremented by 1 to the total
/// size value provided (exclusive).
#if CUDAQ_USE_STD20
template <typename ElementType>
  requires(std::signed_integral<ElementType>)
#else
template <typename ElementType,
          typename = std::enable_if_t<std::is_integral_v<ElementType> &&
                                      std::is_signed_v<ElementType>>>
#endif
inline std::vector<ElementType> range(ElementType N) {
  return range(ElementType(0), N);
}

/// @brief Return a vector of unsigned integers. The first element is zero, and
/// the remaining elements are all values incremented by 1 to the total
/// size value provided (exclusive).
inline std::vector<std::size_t> range(std::size_t N) {
  if (N > std::numeric_limits<std::make_signed_t<std::size_t>>::max())
    throw std::runtime_error("invalid size value to cudaq::range()");
  std::vector<std::size_t> vec(N);
  std::iota(vec.begin(), vec.end(), 0);
  return vec;
}

inline std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  auto internal_split = [](const std::string &s, char delim, auto op) {
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, delim);) {
      *op++ = item;
    }
  };
  internal_split(s, delim, std::back_inserter(elems));
  return elems;
}

} // namespace cudaq
