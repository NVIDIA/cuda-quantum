/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompileTargetCaster.h"
#include "Reflection.h"
#include <cctype>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <utility>

namespace {

namespace nb = nanobind;
using cudaq::reflection::typeName;

/// The Python type called `name` in `cudaq.core._target`.
nb::object pyType(std::string_view name) {
  // Never released: the module outlives the interpreter's finalization.
  static auto *module =
      new nb::module_(nb::module_::import_("cudaq.core._target"));
  return module->attr(nb::str(name.data(), name.size()));
}

/// The Python name of the C++ field `name`: `eraseQEC` becomes `erase_qec`.
std::string pythonName(std::string_view name) {
  std::string snake;
  for (std::size_t i = 0; i < name.size(); ++i) {
    bool wordStart = i > 0 && std::isupper(name[i]) &&
                     (std::islower(name[i - 1]) ||
                      (i + 1 < name.size() && std::islower(name[i + 1])));
    if (wordStart)
      snake += '_';
    snake += std::tolower(name[i]);
  }
  return snake;
}

/// Whether the field `name` of `T` is not exposed to Python.
template <typename T>
bool isHidden(std::string_view name) {
  return std::is_same_v<T, cudaq::CompileTarget> &&
         name == "resetOnRandomSeedSet";
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

/// The nanobind name of `T`.
template <typename T>
constexpr auto typeDescr = []<std::size_t... I>(std::index_sequence<I...>) {
  return nb::detail::descr<sizeof...(I)>(typeName<T>[I]...);
}(std::make_index_sequence<typeName<T>.size()>());

} // namespace

namespace nanobind::detail {

/// Converts the struct `T` to and from the dataclass of the same name.
template <typename T>
struct reflected_caster {
  NB_TYPE_CASTER(T, typeDescr<T>)

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    try {
      if (!isinstance(src, pyType(typeName<T>)))
        return false;
      // The dataclass may check its own invariants, which could have been
      // broken since it was constructed.
      if (hasattr(src, "verify"))
        src.attr("verify")();
      bool ok = true;
      cudaq::reflection::forEachField(value, [&](std::string_view name,
                                                 auto &field) {
        using Field = std::remove_reference_t<decltype(field)>;
        if (!ok || isHidden<T>(name))
          return;
        make_caster<Field> caster;
        ok = caster.from_python(getattr(src, pythonName(name).c_str(), none()),
                                flags, cleanup) &&
             caster.template can_cast<Field>();
        if (ok)
          field = caster.operator cast_t<Field>();
      });
      return ok;
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(const T &value, rv_policy, cleanup_list *) noexcept {
    return releaseOrNull([&] {
      dict kwargs;
      cudaq::reflection::forEachField(
          value, [&](std::string_view name, const auto &field) {
            if (!isHidden<T>(name))
              kwargs[pythonName(name).c_str()] = cast(field);
          });
      return pyType(typeName<T>)(**kwargs);
    });
  }
};

/// Converts the enum `E` to and from the Python enum of the same name. Accepts
/// a member of the enum or its name.
template <typename E>
struct reflected_enum_caster {
  NB_TYPE_CASTER(E, typeDescr<E>)

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    try {
      std::string name;
      if (!try_cast(src, name)) {
        if (!isinstance(src, pyType(typeName<E>)))
          return false;
        name = cast<std::string>(src.attr("name"));
      }
      auto enumerator = cudaq::reflection::enumFromName<E>(name);
      if (enumerator)
        value = *enumerator;
      return enumerator.has_value();
    } catch (...) {
      return false;
    }
  }

  static handle from_cpp(E value, rv_policy, cleanup_list *) noexcept {
    return releaseOrNull([&]() -> object {
      return pyType(typeName<E>)
          .attr(std::string(cudaq::reflection::enumName(value)).c_str());
    });
  }
};

template <typename T>
struct type_caster<
    T, enable_if_t<cudaq::reflection::isReflected<T> && std::is_class_v<T>>>
    : reflected_caster<T> {};

// Enums are listed one by one, as a partial specialization would be ambiguous
// with nanobind's enum caster.
template <>
struct type_caster<cudaq::NoiseSupport>
    : reflected_enum_caster<cudaq::NoiseSupport> {};
template <>
struct type_caster<cudaq::config::CostModel>
    : reflected_enum_caster<cudaq::config::CostModel> {};
template <>
struct type_caster<cudaq::config::LoopUnrolling>
    : reflected_enum_caster<cudaq::config::LoopUnrolling> {};

/// `MeasurementForm` is a `str` in Python.
template <>
struct type_caster<cudaq::config::MeasurementForm>
    : reflected_enum_caster<cudaq::config::MeasurementForm> {
  static constexpr auto Name = const_name("str");

  static handle from_cpp(cudaq::config::MeasurementForm value, rv_policy,
                         cleanup_list *) noexcept {
    auto name = cudaq::reflection::enumName(value);
    return PyUnicode_FromStringAndSize(name.data(), name.size());
  }
};

/// `InferredLowering` is `None` in Python.
template <>
struct type_caster<cudaq::config::InferredLowering>
    : none_caster<cudaq::config::InferredLowering> {};

/// Also accepts the emission format as a string.
template <>
struct type_caster<cudaq::config::Codegen>
    : reflected_caster<cudaq::config::Codegen> {
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (std::string format; try_cast(src, format)) {
      value = {std::move(format), ""};
      return true;
    }
    return reflected_caster::from_python(src, flags, cleanup);
  }
};

bool type_caster<cudaq::CompileTarget>::from_python(
    handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
  reflected_caster<cudaq::CompileTarget> caster;
  if (!caster.from_python(src, flags, cleanup))
    return false;
  value = std::move(caster.value);
  return true;
}

handle
type_caster<cudaq::CompileTarget>::from_cpp(const cudaq::CompileTarget &value,
                                            rv_policy policy,
                                            cleanup_list *cleanup) noexcept {
  return reflected_caster<cudaq::CompileTarget>::from_cpp(value, policy,
                                                          cleanup);
}

} // namespace nanobind::detail
