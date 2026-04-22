/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>

namespace py_ext {

/// Extended python complex object.
///
/// Includes `complex`, `numpy.complex64`, `numpy.complex128`.
class Complex : public nanobind::object {
public:
  NB_OBJECT_DEFAULT(Complex, object, "complex", isComplex_)

  // NOLINTNEXTLINE(google-explicit-constructor)
  Complex(const nanobind::object &o)
      : object(nanobind::steal(convert_(o.ptr()))) {
    if (!m_ptr)
      throw nanobind::python_error();
  }

  Complex(double real, double imag)
      : object(nanobind::steal(PyComplex_FromDoubles(real, imag))) {
    if (!m_ptr) {
      throw std::runtime_error("Could not allocate complex object!");
    }
  }

  // Allow implicit conversion from complex<double>/complex<float>:
  // NOLINTNEXTLINE(google-explicit-constructor)
  Complex(std::complex<double> value)
      : Complex((double)value.real(), (double)value.imag()) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  Complex(std::complex<float> value)
      : Complex((double)value.real(), (double)value.imag()) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::complex<double>() {
    auto value = PyComplex_AsCComplex(m_ptr);
    return std::complex<double>(value.real, value.imag);
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::complex<float>() {
    auto value = PyComplex_AsCComplex(m_ptr);
    return std::complex<float>(value.real, value.imag);
  }

  static bool isComplex_(PyObject *o) {
    if (PyComplex_Check(o)) {
      return true;
    }
    PyTypeObject *type = Py_TYPE(o);
    std::string name = std::string(type->tp_name);
    if (name == "numpy.complex64" || name == "numpy.complex128") {
      return true;
    }
    return false;
  }

  static PyObject *convert_(PyObject *o) {
    PyObject *ret = nullptr;
    if (isComplex_(o)) {
      double real = PyComplex_RealAsDouble(o);
      double imag = PyComplex_ImagAsDouble(o);
      ret = PyComplex_FromDoubles(real, imag);
    } else {
      PyErr_SetString(PyExc_TypeError, "Unexpected type");
    }
    return ret;
  }
};

/// Extended python float object.
///
/// Includes `float`, `numpy.float64`, `numpy.float32`.
class Float : public nanobind::object {
public:
  NB_OBJECT_DEFAULT(Float, object, "float", isFloat_)

  // Converting constructor
  // NOLINTNEXTLINE(google-explicit-constructor)
  Float(const nanobind::object &o)
      : object(nanobind::steal(convert_(o.ptr()))) {
    if (!m_ptr)
      throw nanobind::python_error();
  }

  // Allow implicit conversion from float/double:
  // NOLINTNEXTLINE(google-explicit-constructor)
  Float(float value)
      : object(nanobind::steal(PyFloat_FromDouble((double)value))) {
    if (!m_ptr) {
      throw std::runtime_error("Could not allocate float object!");
    }
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  Float(double value = .0)
      : object(nanobind::steal(PyFloat_FromDouble((double)value))) {
    if (!m_ptr) {
      throw std::runtime_error("Could not allocate float object!");
    }
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator float() const { return (float)PyFloat_AsDouble(m_ptr); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator double() const { return (double)PyFloat_AsDouble(m_ptr); }

  static bool isFloat_(PyObject *o) {
    if (PyFloat_Check(o)) {
      return true;
    }
    PyTypeObject *type = Py_TYPE(o);
    std::string name = std::string(type->tp_name);
    if (name == "numpy.float32" || name == "numpy.float64") {
      return true;
    }
    return false;
  }

  static PyObject *convert_(PyObject *o) {
    PyObject *ret = nullptr;
    if (isFloat_(o)) {
      ret = PyFloat_FromDouble(PyFloat_AsDouble(o));
    } else {
      PyErr_SetString(PyExc_TypeError, "Unexpected type");
    }
    return ret;
  }
};

/// Extended python int object.
///
/// Includes `int`, `numpy.intXXX`.
class Int : public nanobind::object {
public:
  NB_OBJECT_DEFAULT(Int, object, "int", isInt_)

  // Converting constructor
  // NOLINTNEXTLINE(google-explicit-constructor)
  Int(const nanobind::object &o) : object(nanobind::steal(convert_(o.ptr()))) {
    if (!m_ptr)
      throw nanobind::python_error();
  }

  // Allow implicit conversion from int:
  // NOLINTNEXTLINE(google-explicit-constructor)
  Int(long value) : object(nanobind::steal(PyLong_FromLong((long)value))) {
    if (!m_ptr) {
      throw std::runtime_error("Could not allocate int object!");
    }
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::int8_t() const { return (std::int8_t)PyLong_AsLong(m_ptr); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::int16_t() const { return (std::int16_t)PyLong_AsLong(m_ptr); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::int32_t() const { return (std::int32_t)PyLong_AsLong(m_ptr); }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::int64_t() const { return (std::int64_t)PyLong_AsLong(m_ptr); }

  static bool isInt_(PyObject *o) {
    if (PyLong_Check(o)) {
      return true;
    }
    PyTypeObject *type = Py_TYPE(o);
    std::string name = std::string(type->tp_name);
    if (name == "numpy.int8" || name == "numpy.int16" ||
        name == "numpy.int32" || name == "numpy.int64") {
      return true;
    }
    return false;
  }

  static PyObject *convert_(PyObject *o) {
    PyObject *ret = nullptr;
    if (isInt_(o)) {
      ret = PyLong_FromLong(PyLong_AsLong(o));
    } else {
      PyErr_SetString(PyExc_TypeError, "Unexpected type");
    }
    return ret;
  }
};

template <typename T>
inline char const *typeName() {
  return typeid(T).name();
}
template <>
inline char const *typeName<py_ext::Float>() {
  return "float";
}
template <>
inline char const *typeName<py_ext::Complex>() {
  return "complex";
}
template <>
inline char const *typeName<py_ext::Int>() {
  return "long";
}
template <>
inline char const *typeName<nanobind::int_>() {
  return "long";
}
template <>
inline char const *typeName<nanobind::bool_>() {
  return "bool";
}
template <>
inline char const *typeName<nanobind::list>() {
  return "list";
}

template <typename T, std::enable_if_t<
                          std::is_base_of<nanobind::object, T>::value, int> = 0>
inline bool isConvertible(nanobind::handle o) {
  return nanobind::isinstance<T>(o);
}
template <>
inline bool isConvertible<Complex>(nanobind::handle o) {
  return nanobind::isinstance<Complex>(o) || nanobind::isinstance<Float>(o) ||
         nanobind::isinstance<nanobind::int_>(o);
}
template <>
inline bool isConvertible<Float>(nanobind::handle o) {
  return nanobind::isinstance<Float>(o) ||
         nanobind::isinstance<nanobind::int_>(o);
}

template <typename T>
inline nanobind::object convert(T value) = delete;

template <>
inline nanobind::object convert(bool value) {
  return nanobind::bool_(value);
}

template <>
inline nanobind::object convert(std::int8_t value) {
  return nanobind::int_(value);
}

template <>
inline nanobind::object convert(std::int16_t value) {
  return nanobind::int_(value);
}

template <>
inline nanobind::object convert(std::int32_t value) {
  return nanobind::int_(value);
}

template <>
inline nanobind::object convert(std::int64_t value) {
  return nanobind::int_(value);
}

template <>
inline nanobind::object convert(float value) {
  return Float(value);
}

template <>
inline nanobind::object convert(double value) {
  return Float(value);
}

template <>
inline nanobind::object convert(std::complex<float> value) {
  return Complex(value);
}

template <>
inline nanobind::object convert(std::complex<double> value) {
  return Complex(value);
}

} // namespace py_ext
