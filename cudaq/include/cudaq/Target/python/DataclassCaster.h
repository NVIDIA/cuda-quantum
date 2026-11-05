/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Generic nanobind type casters between C++ structs and enums and Python
/// dataclasses and enums. Describe a struct with a `Dataclass` specialization
/// and an enum with an `Enum` specialization; values are deep-copied when
/// converted.
///
/// The casters take a `Module` class whose static `get()` returns the Python
/// module defining the types, with `isInstance`, `create`, `enumMemberName`
/// and `enumMember` methods.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <string_view>
#include <tuple>

namespace cudaq::python {

namespace nb = nanobind;

template <typename T, typename M>
struct Field {
  const char *name;
  M T::*member;
};

/// Maps the C++ struct `T` to a Python dataclass: specialize with the
/// dataclass `name` and its `fields`.
template <typename T>
struct Dataclass {};

template <typename T>
constexpr bool isDataclass = requires { Dataclass<T>::fields; };

template <typename E>
struct EnumMember {
  const char *name;
  E value;
};

/// Maps the C++ enum `E` to a Python enum, or to strings if `name` is "str":
/// specialize with `name` and `members`.
template <typename E>
struct Enum {};

/// Where and why the last conversion from Python failed, e.g.
/// ".stages[0].lowering: ...". Casters report failures here rather than
/// throwing, because nanobind casters for containers are `noexcept`.
inline std::string &conversionError() {
  thread_local std::string error;
  return error;
}

template <typename T>
constexpr bool isVector = false;
template <typename T>
constexpr bool isVector<std::vector<T>> = true;

template <typename T>
std::string expectedType() {
  if constexpr (std::is_same_v<T, bool>)
    return "bool";
  else if constexpr (std::is_floating_point_v<T>)
    return "float";
  else if constexpr (std::is_integral_v<T>)
    return "int";
  else if constexpr (std::is_same_v<T, std::string>)
    return "str";
  else if constexpr (isVector<T>)
    return "a list of " + expectedType<typename T::value_type>();
  else
    return nb::detail::make_caster<T>::Name.text;
}

template <typename T>
bool fail(nb::handle got) {
  conversionError() = ": expected " + expectedType<T>() + ", got " +
                      Py_TYPE(got.ptr())->tp_name;
  return false;
}

/// Converts `src` into `out`, recording why in `conversionError()` on failure.
template <typename T>
bool load(nb::handle src, T &out, nb::detail::cleanup_list *cleanup) {
  if constexpr (isVector<T>) {
    if (!PyList_Check(src.ptr()) && !PyTuple_Check(src.ptr()))
      return fail<T>(src);
    out.clear();
    for (nb::handle item : src) {
      if (!load(item, out.emplace_back(), cleanup)) {
        conversionError().insert(0, "[" + std::to_string(out.size() - 1) + "]");
        return false;
      }
    }
    return true;
  } else {
    conversionError().clear();
    nb::detail::make_caster<T> caster;
    if (caster.from_python(src, (uint8_t)nb::detail::cast_flags::convert,
                           cleanup) &&
        caster.template can_cast<T>()) {
      out = caster.operator nb::detail::cast_t<T>();
      return true;
    }
    return conversionError().empty() ? fail<T>(src) : false;
  }
}

/// Converts the fields of the dataclass instance `src` into `out`.
template <typename T>
bool loadFields(nb::handle src, T &out, nb::detail::cleanup_list *cleanup) {
  return std::apply(
      [&](const auto &...field) {
        return ([&] {
          if (load(nb::getattr(src, field.name, nb::none()), out.*field.member,
                   cleanup))
            return true;
          conversionError().insert(0, std::string(".") + field.name);
          return false;
        }() && ...);
      },
      Dataclass<T>::fields);
}

template <typename T>
nb::dict fieldsToPython(const T &value) {
  nb::dict kwargs;
  std::apply(
      [&](const auto &...field) {
        ((kwargs[field.name] = nb::cast(value.*field.member)), ...);
      },
      Dataclass<T>::fields);
  return kwargs;
}

/// Runs `f`, recording an exception in `conversionError()`.
template <typename F>
bool recordingErrors(F &&f) noexcept {
  try {
    return f();
  } catch (const std::exception &e) {
    conversionError() = std::string(": ") + e.what();
    return false;
  }
}

/// Runs `f`, which builds a Python object, and reports an exception as a
/// pending Python error and a null handle, as nanobind expects of `from_cpp`.
template <typename F>
nb::handle releaseOrNull(F &&f) noexcept {
  try {
    return f().release();
  } catch (nb::python_error &e) {
    e.restore();
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_TypeError, e.what());
  }
  return nb::handle();
}

} // namespace cudaq::python

namespace nanobind::detail {

template <typename T, typename Module>
struct dataclass_caster {
  using Dataclass = cudaq::python::Dataclass<T>;
  NB_TYPE_CASTER(T, Dataclass::name)

  bool from_python(handle src, uint8_t, cleanup_list *cleanup) noexcept {
    return cudaq::python::recordingErrors([&] {
      return Module::get().isInstance(src, Dataclass::name.text) &&
             cudaq::python::loadFields(src, value, cleanup);
    });
  }

  static handle from_cpp(const T &value, rv_policy, cleanup_list *) noexcept {
    return cudaq::python::releaseOrNull([&] {
      return Module::get().create(Dataclass::name.text,
                                  cudaq::python::fieldsToPython(value));
    });
  }
};

/// Accepts a member of the enum or its name.
template <typename E, typename Module>
struct enum_caster {
  using Enum = cudaq::python::Enum<E>;
  NB_TYPE_CASTER(E, Enum::name)

  static constexpr bool isStr = std::string_view(Enum::name.text) == "str";

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    return cudaq::python::recordingErrors([&] {
      std::string name;
      std::optional<std::string> member;
      if constexpr (!isStr)
        member = Module::get().enumMemberName(src, Enum::name.text);
      if (member)
        name = *member;
      else if (!try_cast(src, name))
        return false;
      std::string choices;
      for (auto [memberName, memberValue] : Enum::members) {
        if (name == memberName) {
          value = memberValue;
          return true;
        }
        choices +=
            (choices.empty() ? "'" : ", '") + std::string(memberName) + "'";
      }
      cudaq::python::conversionError() =
          ": expected one of " + choices + ", got '" + name + "'";
      return false;
    });
  }

  static handle from_cpp(E value, rv_policy, cleanup_list *) noexcept {
    return cudaq::python::releaseOrNull([&]() -> object {
      for (auto [memberName, memberValue] : Enum::members)
        if (memberValue == value) {
          if constexpr (isStr)
            return str(memberName);
          else
            return Module::get().enumMember(Enum::name.text, memberName);
        }
      throw std::invalid_argument("not a member of the enum");
    });
  }
};

} // namespace nanobind::detail
