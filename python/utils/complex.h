/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class complex_ : public py::object {
public:
    // Python does not provide a `PyNumber_Complex` function, so we provide a partial
    // functionality compared to a py::float_ and other similar object classes.
    // PYBIND11_OBJECT_CVT(complex_, object, PyComplex_Check, PyNumber_Complex)
    
    complex_(std::complex<double> value) : object(py::detail::type_caster<std::complex<double>>::cast(
        value, py::return_value_policy::automatic, nullptr), stolen_t{}) {}

    complex_(std::complex<float> value) : object(py::detail::type_caster<std::complex<float>>::cast(
        value, py::return_value_policy::automatic, nullptr), stolen_t{}) {}

    static bool check_(handle h) { return h.ptr() != nullptr && _PyObject_TypeCheck(((PyObject*)(h.ptr())), &PyComplex_Type); }

    operator std::complex<double>() {
      auto value = PyComplex_AsCComplex(m_ptr);
      return std::complex<double>(value.real, value.imag);
    }
    operator std::complex<float>() {
      auto value = PyComplex_AsCComplex(m_ptr);
      return std::complex<float>(value.real, value.imag);
    }
};