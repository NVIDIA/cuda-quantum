/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "operator_leafs.h"
#include <complex>
#include <type_traits>

#include "cudaq/boson_op.h"
#include "cudaq/fermion_op.h"
#include "cudaq/matrix_op.h"
#include "cudaq/spin_op.h"

namespace cudaq {

template <typename HandlerTy>
class product_op;

template <typename HandlerTy>
class sum_op;

#define TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)                             \
  std::enable_if_t<!std::is_same<LHtype, RHtype>::value &&                     \
                       !std::is_same<matrix_handler, LHtype>::value &&         \
                       std::is_base_of<operator_handler, LHtype>::value &&     \
                       std::is_base_of<operator_handler, RHtype>::value,       \
                   bool>

template <typename HandlerTy>
product_op<HandlerTy> operator*(const scalar_operator &other,
                                const product_op<HandlerTy> &self);
template <typename HandlerTy>
product_op<HandlerTy> operator*(const scalar_operator &other,
                                product_op<HandlerTy> &&self);
template <typename HandlerTy>
sum_op<HandlerTy> operator+(const scalar_operator &other,
                            const product_op<HandlerTy> &self);
template <typename HandlerTy>
sum_op<HandlerTy> operator+(const scalar_operator &other,
                            product_op<HandlerTy> &&self);
template <typename HandlerTy>
sum_op<HandlerTy> operator-(const scalar_operator &other,
                            const product_op<HandlerTy> &self);
template <typename HandlerTy>
sum_op<HandlerTy> operator-(const scalar_operator &other,
                            product_op<HandlerTy> &&self);

template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
product_op<matrix_handler> operator*(const product_op<LHtype> &other,
                                     const product_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator+(const product_op<LHtype> &other,
                                 const product_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator-(const product_op<LHtype> &other,
                                 const product_op<RHtype> &self);

template <typename HandlerTy>
sum_op<HandlerTy> operator*(const scalar_operator &other,
                            const sum_op<HandlerTy> &self);
template <typename HandlerTy>
sum_op<HandlerTy> operator*(const scalar_operator &other,
                            sum_op<HandlerTy> &&self);
template <typename HandlerTy>
sum_op<HandlerTy> operator+(const scalar_operator &other,
                            const sum_op<HandlerTy> &self);
template <typename HandlerTy>
sum_op<HandlerTy> operator+(const scalar_operator &other,
                            sum_op<HandlerTy> &&self);
template <typename HandlerTy>
sum_op<HandlerTy> operator-(const scalar_operator &other,
                            const sum_op<HandlerTy> &self);
template <typename HandlerTy>
sum_op<HandlerTy> operator-(const scalar_operator &other,
                            sum_op<HandlerTy> &&self);

template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator*(const sum_op<LHtype> &other,
                                 const product_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator+(const sum_op<LHtype> &other,
                                 const product_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator-(const sum_op<LHtype> &other,
                                 const product_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator*(const product_op<LHtype> &other,
                                 const sum_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator+(const product_op<LHtype> &other,
                                 const sum_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator-(const product_op<LHtype> &other,
                                 const sum_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator*(const sum_op<LHtype> &other,
                                 const sum_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator+(const sum_op<LHtype> &other,
                                 const sum_op<RHtype> &self);
template <typename LHtype, typename RHtype,
          TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
sum_op<matrix_handler> operator-(const sum_op<LHtype> &other,
                                 const sum_op<RHtype> &self);

#ifndef CUDAQ_INSTANTIATE_TEMPLATES
#define EXTERN_TEMPLATE_SPECIALIZATIONS(HandlerTy)                             \
                                                                               \
  extern template product_op<HandlerTy> operator*(                             \
      const scalar_operator &other, const product_op<HandlerTy> &self);        \
  extern template product_op<HandlerTy> operator*(                             \
      const scalar_operator &other, product_op<HandlerTy> &&self);             \
  extern template sum_op<HandlerTy> operator+(                                 \
      const scalar_operator &other, const product_op<HandlerTy> &self);        \
  extern template sum_op<HandlerTy> operator+(const scalar_operator &other,    \
                                              product_op<HandlerTy> &&self);   \
  extern template sum_op<HandlerTy> operator-(                                 \
      const scalar_operator &other, const product_op<HandlerTy> &self);        \
  extern template sum_op<HandlerTy> operator-(const scalar_operator &other,    \
                                              product_op<HandlerTy> &&self);   \
                                                                               \
  extern template sum_op<HandlerTy> operator*(const scalar_operator &other,    \
                                              const sum_op<HandlerTy> &self);  \
  extern template sum_op<HandlerTy> operator*(const scalar_operator &other,    \
                                              sum_op<HandlerTy> &&self);       \
  extern template sum_op<HandlerTy> operator+(const scalar_operator &other,    \
                                              const sum_op<HandlerTy> &self);  \
  extern template sum_op<HandlerTy> operator+(const scalar_operator &other,    \
                                              sum_op<HandlerTy> &&self);       \
  extern template sum_op<HandlerTy> operator-(const scalar_operator &other,    \
                                              const sum_op<HandlerTy> &self);  \
  extern template sum_op<HandlerTy> operator-(const scalar_operator &other,    \
                                              sum_op<HandlerTy> &&self);

EXTERN_TEMPLATE_SPECIALIZATIONS(matrix_handler);
EXTERN_TEMPLATE_SPECIALIZATIONS(spin_handler);
EXTERN_TEMPLATE_SPECIALIZATIONS(boson_handler);

#define EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(op, returnTy)               \
                                                                               \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<spin_handler> &other,                                   \
      const product_op<matrix_handler> &self);                                 \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<boson_handler> &other,                                  \
      const product_op<matrix_handler> &self);                                 \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<fermion_handler> &other,                                \
      const product_op<matrix_handler> &self);                                 \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<spin_handler> &other,                                   \
      const product_op<boson_handler> &self);                                  \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<boson_handler> &other,                                  \
      const product_op<spin_handler> &self);                                   \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<spin_handler> &other,                                   \
      const product_op<fermion_handler> &self);                                \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<fermion_handler> &other,                                \
      const product_op<spin_handler> &self);                                   \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<boson_handler> &other,                                  \
      const product_op<fermion_handler> &self);                                \
  extern template returnTy<matrix_handler> operator op(                        \
      const product_op<fermion_handler> &other,                                \
      const product_op<boson_handler> &self);                                  \
                                                                               \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<spin_handler> &other,                                       \
      const product_op<matrix_handler> &self);                                 \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<boson_handler> &other,                                      \
      const product_op<matrix_handler> &self);                                 \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<fermion_handler> &other,                                    \
      const product_op<matrix_handler> &self);                                 \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<spin_handler> &other,                                       \
      const product_op<boson_handler> &self);                                  \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<boson_handler> &other,                                      \
      const product_op<spin_handler> &self);                                   \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<spin_handler> &other,                                       \
      const product_op<fermion_handler> &self);                                \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<fermion_handler> &other,                                    \
      const product_op<spin_handler> &self);                                   \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<boson_handler> &other,                                      \
      const product_op<fermion_handler> &self);                                \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<fermion_handler> &other,                                    \
      const product_op<boson_handler> &self);                                  \
                                                                               \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<spin_handler> &other,                                   \
      const sum_op<matrix_handler> &self);                                     \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<boson_handler> &other,                                  \
      const sum_op<matrix_handler> &self);                                     \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<fermion_handler> &other,                                \
      const sum_op<matrix_handler> &self);                                     \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<spin_handler> &other,                                   \
      const sum_op<boson_handler> &self);                                      \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<boson_handler> &other,                                  \
      const sum_op<spin_handler> &self);                                       \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<spin_handler> &other,                                   \
      const sum_op<fermion_handler> &self);                                    \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<fermion_handler> &other,                                \
      const sum_op<spin_handler> &self);                                       \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<boson_handler> &other,                                  \
      const sum_op<fermion_handler> &self);                                    \
  extern template sum_op<matrix_handler> operator op(                          \
      const product_op<fermion_handler> &other,                                \
      const sum_op<boson_handler> &self);                                      \
                                                                               \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<spin_handler> &other, const sum_op<matrix_handler> &self);  \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<boson_handler> &other, const sum_op<matrix_handler> &self); \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<fermion_handler> &other,                                    \
      const sum_op<matrix_handler> &self);                                     \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<spin_handler> &other, const sum_op<boson_handler> &self);   \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<boson_handler> &other, const sum_op<spin_handler> &self);   \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<spin_handler> &other, const sum_op<fermion_handler> &self); \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<fermion_handler> &other, const sum_op<spin_handler> &self); \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<boson_handler> &other,                                      \
      const sum_op<fermion_handler> &self);                                    \
  extern template sum_op<matrix_handler> operator op(                          \
      const sum_op<fermion_handler> &other,                                    \
      const sum_op<boson_handler> &self);

EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(*, product_op);
EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(+, sum_op);
EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(-, sum_op);
#endif

// templates for arithmetics with callables

#define PRODUCT_FUNCTION_ARITHMETICS_RHS(op, returnTy)                         \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(product_op<HandlerTy> &&prod,                \
                                  Callable &&fct) {                            \
    return std::move(prod)                                                     \
        op scalar_operator(scalar_callback(std::forward<Callable>(fct)));      \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(const product_op<HandlerTy> &prod,           \
                                  Callable &&fct) {                            \
    return prod op scalar_operator(                                            \
        scalar_callback(std::forward<Callable>(fct)));                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(product_op<HandlerTy> &&prod,                \
                                  scalar value) {                              \
    return std::move(prod) op scalar_operator(value);                          \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(const product_op<HandlerTy> &prod,           \
                                  scalar value) {                              \
    return prod op scalar_operator(value);                                     \
  }

#define PRODUCT_FUNCTION_ARITHMETICS_LHS(op, returnTy)                         \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(Callable &&fct,                              \
                                  product_op<HandlerTy> &&prod) {              \
    return scalar_operator(scalar_callback(std::forward<Callable>(fct)))       \
        op std::move(prod);                                                    \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(Callable &&fct,                              \
                                  const product_op<HandlerTy> &prod) {         \
    return scalar_operator(scalar_callback(std::forward<Callable>(fct)))       \
        op prod;                                                               \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(scalar value,                                \
                                  product_op<HandlerTy> &&prod) {              \
    return scalar_operator(value) op std::move(prod);                          \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  returnTy<HandlerTy> operator op(scalar value,                                \
                                  const product_op<HandlerTy> &prod) {         \
    return scalar_operator(value) op prod;                                     \
  }

#define PRODUCT_FUNCTION_ARITHMETICS_ASSIGNMENT(op)                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  product_op<HandlerTy> &operator op(product_op<HandlerTy> &prod,              \
                                     Callable &&fct) {                         \
    return prod op scalar_operator(                                            \
        scalar_callback(std::forward<Callable>(fct)));                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  product_op<HandlerTy> &operator op(product_op<HandlerTy> &prod,              \
                                     scalar value) {                           \
    return prod op scalar_operator(value);                                     \
  }

PRODUCT_FUNCTION_ARITHMETICS_RHS(*, product_op);
PRODUCT_FUNCTION_ARITHMETICS_RHS(/, product_op);
PRODUCT_FUNCTION_ARITHMETICS_RHS(+, sum_op);
PRODUCT_FUNCTION_ARITHMETICS_RHS(-, sum_op);
PRODUCT_FUNCTION_ARITHMETICS_LHS(*, product_op);
PRODUCT_FUNCTION_ARITHMETICS_LHS(+, sum_op);
PRODUCT_FUNCTION_ARITHMETICS_LHS(-, sum_op);
PRODUCT_FUNCTION_ARITHMETICS_ASSIGNMENT(*=);
PRODUCT_FUNCTION_ARITHMETICS_ASSIGNMENT(/=);

#define SUM_FUNCTION_ARITHMETICS_RHS(op)                                       \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  sum_op<HandlerTy> operator op(sum_op<HandlerTy> &&sum, Callable &&fct) {     \
    return std::move(sum)                                                      \
        op scalar_operator(scalar_callback(std::forward<Callable>(fct)));      \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  sum_op<HandlerTy> operator op(const sum_op<HandlerTy> &sum,                  \
                                Callable &&fct) {                              \
    return sum op scalar_operator(                                             \
        scalar_callback(std::forward<Callable>(fct)));                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  sum_op<HandlerTy> operator op(sum_op<HandlerTy> &&sum, scalar value) {       \
    return std::move(sum) op scalar_operator(value);                           \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  sum_op<HandlerTy> operator op(const sum_op<HandlerTy> &sum, scalar value) {  \
    return sum op scalar_operator(value);                                      \
  }

#define SUM_FUNCTION_ARITHMETICS_LHS(op)                                       \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  sum_op<HandlerTy> operator op(Callable &&fct, sum_op<HandlerTy> &&sum) {     \
    return scalar_operator(scalar_callback(std::forward<Callable>(fct)))       \
        op std::move(sum);                                                     \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  sum_op<HandlerTy> operator op(Callable &&fct,                                \
                                const sum_op<HandlerTy> &sum) {                \
    return scalar_operator(scalar_callback(std::forward<Callable>(fct)))       \
        op sum;                                                                \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  sum_op<HandlerTy> operator op(scalar value, sum_op<HandlerTy> &&sum) {       \
    return scalar_operator(value) op std::move(sum);                           \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  sum_op<HandlerTy> operator op(scalar value, const sum_op<HandlerTy> &sum) {  \
    return scalar_operator(value) op sum;                                      \
  }

#define SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(op)                                \
                                                                               \
  template <typename HandlerTy, typename Callable,                             \
            std::enable_if_t<                                                  \
                std::is_constructible<scalar_callback, Callable>::value,       \
                bool> = true>                                                  \
  sum_op<HandlerTy> &operator op(sum_op<HandlerTy> &prod, Callable &&fct) {    \
    return prod op scalar_operator(                                            \
        scalar_callback(std::forward<Callable>(fct)));                         \
  }                                                                            \
                                                                               \
  template <typename HandlerTy, typename scalar,                               \
            std::enable_if_t<                                                  \
                std::is_constructible<std::complex<double>, scalar>::value,    \
                bool> = true>                                                  \
  sum_op<HandlerTy> &operator op(sum_op<HandlerTy> &prod, scalar value) {      \
    return prod op scalar_operator(value);                                     \
  }

SUM_FUNCTION_ARITHMETICS_RHS(*);
SUM_FUNCTION_ARITHMETICS_RHS(/);
SUM_FUNCTION_ARITHMETICS_RHS(+);
SUM_FUNCTION_ARITHMETICS_RHS(-);
SUM_FUNCTION_ARITHMETICS_LHS(*);
SUM_FUNCTION_ARITHMETICS_LHS(+);
SUM_FUNCTION_ARITHMETICS_LHS(-);
SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(*=);
SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(/=);
SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(+=);
SUM_FUNCTION_ARITHMETICS_ASSIGNMENT(-=);

} // namespace cudaq
