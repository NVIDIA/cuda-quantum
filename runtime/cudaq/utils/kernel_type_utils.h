/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <array>
#include <string_view>
#include <type_traits>

namespace cudaq {
/// Helper utils to customize compiler error messages when concept requirements
/// are not satisfied.
///
/// Notes: There are ongoing work in the C++ standard to
/// address this, hence making it easier to implement what we are doing here.
/// (1) User-generated static_assert message (using std::format to construct
/// static_assert messages) It will be available in C++-26
/// (https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2741r3.pdf).
///
/// (2) Support for customizing concept-related error messages e.g.,
/// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1267r0.pdf

/// @brief  Join string_views at compile time via std::array
template <std::string_view const &...StrViews>
struct JoinStringView {
  // Concatenate string_view's into a single std::array of chars
  static constexpr auto join() noexcept {
    constexpr std::size_t len = (StrViews.size() + ... + 0);
    std::array<char, len + 1> arr{};
    auto append = [i = 0, &arr](auto const &s) mutable {
      for (auto c : s)
        arr[i++] = c;
    };
    (append(StrViews), ...);
    arr[len] = 0;
    return arr;
  }
  // Static storage of the joined array
  static inline constexpr auto arr = join();
  // View the array as a std::string_view
  static constexpr std::string_view value{arr.data(), arr.size() - 1};
};
template <std::string_view const &...Strs>
static constexpr auto JoinStringView_v = JoinStringView<Strs...>::value;

template <std::size_t... Idxs>
constexpr auto subStringviewToArray(std::string_view str,
                                    std::index_sequence<Idxs...>) {
  return std::array{str[Idxs]...};
}

/// Util class to query the function signature of kernel's operator() member
/// function.
template <typename T>
struct MemberFuncArgs;
template <typename RT, typename Owner, typename... Args>
struct MemberFuncArgs<RT (Owner::*)(Args...)> {
  static constexpr std::size_t argCount = sizeof...(Args);
  using ReturnType = RT;
  using InArgsTuple = std::tuple<Args...>;
};

/// Helper to retrieve argument list from typename at compile time
/// Adopted from LLVM Support's TypeName.h.
template <typename T>
struct StaticArgsList {
  static constexpr auto parse_args_types() {
#if defined(__clang__) || defined(__GNUC__)
    constexpr auto prefix = std::string_view{"T = "};
    constexpr auto suffix = std::string_view{"]"};
    constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#else
#error Unsupported compiler
#endif
    // First, parse the full type, e.g., "kernel_builder<double>"
    constexpr auto start = function.find(prefix) + prefix.size();
    constexpr auto end = function.rfind(suffix);
    static_assert(start < end);
    constexpr auto name = function.substr(start, (end - start));
    if constexpr (name.find("void (") != std::string_view::npos) {
      constexpr auto startArgsSig = name.find_first_of('(');
      constexpr auto endArgsSig = name.find_last_of(')');
      constexpr auto argsList =
          name.substr(startArgsSig + 1, (endArgsSig - startArgsSig - 1));
      return subStringviewToArray(argsList,
                                  std::make_index_sequence<argsList.size()>{});
    } else if constexpr (name.find("kernel_builder") !=
                             std::string_view::npos ||
                         name.find("std::tuple") != std::string_view::npos) {
      // Now, remove the type to get args, e.g.,
      // "kernel_builder<double>" -> double
      // "std::tuple<double>"" -> double
      constexpr auto startTemplatePack = name.find_first_of('<');
      constexpr auto endTemplatePack = name.find_last_of('>');
      constexpr auto paramPack = name.substr(
          startTemplatePack + 1, (endTemplatePack - startTemplatePack - 1));
      if constexpr (paramPack.size() == 0)
        return std::array<char, 0>{};
      else
        return subStringviewToArray(
            paramPack, std::make_index_sequence<paramPack.size()>{});
    } else {
      constexpr std::string_view notParsable = "<unknown>";
      return subStringviewToArray(
          notParsable, std::make_index_sequence<notParsable.size()>{});
    }
  }
  // Array storage
  static inline constexpr auto arr = parse_args_types();
  static constexpr std::string_view value{arr.data(), arr.size()};
};

template <typename T>
static constexpr auto staticArgsList_v = StaticArgsList<T>::value;

/// Compile-time static string (as a char array)
template <size_t N>
struct Msg {
  char chars[N + 1] = {};
  constexpr Msg(const char (&str)[N + 1]) { std::copy_n(str, N + 1, chars); }
};

template <size_t N>
Msg(const char (&str)[N]) -> Msg<N - 1>;

/// Assert with a custom statically-constructed error message.
template <auto>
constexpr bool InvalidArgs = false;
template <Msg ErrorMsg>
void compileTimeAssert() {
  static_assert(InvalidArgs<ErrorMsg>);
}

template <typename QuantumKernel, typename... Args>
  requires(!ValidArgumentsPassed<QuantumKernel, Args...>)
void generateInvalidKernelInvocationCompilerError() {
  // Convert args to a tuple => arg types will be parsed from the tuple
  // typename.
  using ArgsTuple = std::tuple<Args...>;

  // Error message format: InvalidArgs<...>{"requires <float>, got <double,
  // int>"}>
  static constexpr std::string_view messagePrefixSv = "requires <";
  static constexpr std::string_view gotSv = ">, got <";
  static constexpr std::string_view messagePostfixSv = ">";
  static constexpr std::string_view kernelTypeNameSv = []() {
    if constexpr (has_name<QuantumKernel>::value ||
                  !std::is_class_v<QuantumKernel>) {
      return staticArgsList_v<std::remove_cvref_t<QuantumKernel>>;
    } else {
      return staticArgsList_v<typename MemberFuncArgs<
          decltype(&QuantumKernel::operator())>::InArgsTuple>;
    }
  }();

  static constexpr std::string_view argsTypeNameSv =
      staticArgsList_v<ArgsTuple>;
  // Construct the string
  constexpr auto sv = JoinStringView_v<messagePrefixSv, kernelTypeNameSv, gotSv,
                                       argsTypeNameSv, messagePostfixSv>;
  constexpr size_t n = sv.size();
  constexpr auto indices = std::make_index_sequence<n>();

  [&]<std::size_t... I>(std::index_sequence<I...>) {
    static constexpr char text[]{(char)sv.at(I)..., 0};
    compileTimeAssert<text>();
  }(indices);
}
} // namespace cudaq
