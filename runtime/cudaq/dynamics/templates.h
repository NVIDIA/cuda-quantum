/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <concepts>

#include "operator_leafs.h"
#include "matrix_operators.h"
#include "spin_operators.h"
#include "boson_operators.h"

namespace cudaq {

template <typename HandlerTy> 
class product_operator;

template <typename HandlerTy>
class operator_sum;

#define TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype)                                                \
  std::enable_if_t<!std::is_same<LHtype, RHtype>::value &&                                        \
                   !std::is_same<matrix_operator, LHtype>::value &&                               \
                   std::is_base_of<operator_handler, LHtype>::value &&                            \
                   std::is_base_of<operator_handler, RHtype>::value, bool>

template <typename HandlerTy> 
product_operator<HandlerTy> operator*(double other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(double other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(double other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
product_operator<HandlerTy> operator*(std::complex<double> other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(std::complex<double> other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(std::complex<double> other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
product_operator<HandlerTy> operator*(const scalar_operator &other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(const scalar_operator &other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(const scalar_operator &other, const product_operator<HandlerTy> &self);

template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
product_operator<matrix_operator> operator*(const product_operator<LHtype> &other, const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator+(const product_operator<LHtype> &other, const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator-(const product_operator<LHtype> &other, const product_operator<RHtype> &self);

template <typename HandlerTy> 
operator_sum<HandlerTy> operator*(double other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(double other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(double other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator*(std::complex<double> other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(std::complex<double> other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(std::complex<double> other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator*(const scalar_operator &other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(const scalar_operator &other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(const scalar_operator &other, const operator_sum<HandlerTy> &self);

template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator*(const operator_sum<LHtype> &other, const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator+(const operator_sum<LHtype> &other, const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator-(const operator_sum<LHtype> &other, const product_operator<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator*(const product_operator<LHtype> &other, const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator+(const product_operator<LHtype> &other, const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator-(const product_operator<LHtype> &other, const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator*(const operator_sum<LHtype> &other, const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator+(const operator_sum<LHtype> &other, const operator_sum<RHtype> &self);
template <typename LHtype, typename RHtype, TYPE_CONVERSION_CONSTRAINT(LHtype, RHtype) = true>
operator_sum<matrix_operator> operator-(const operator_sum<LHtype> &other, const operator_sum<RHtype> &self);


#ifndef CUDAQ_INSTANTIATE_TEMPLATES
#define EXTERN_TEMPLATE_SPECIALIZATIONS(HandlerTy)                                                                  \
                                                                                                                    \
    extern template                                                                                                 \
    product_operator<HandlerTy> operator*(double other, const product_operator<HandlerTy> &self);                   \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator+(double other, const product_operator<HandlerTy> &self);                       \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator-(double other, const product_operator<HandlerTy> &self);                       \
    extern template                                                                                                 \
    product_operator<HandlerTy> operator*(std::complex<double> other, const product_operator<HandlerTy> &self);     \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator+(std::complex<double> other, const product_operator<HandlerTy> &self);         \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator-(std::complex<double> other, const product_operator<HandlerTy> &self);         \
    extern template                                                                                                 \
    product_operator<HandlerTy> operator*(const scalar_operator &other, const product_operator<HandlerTy> &self);   \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator+(const scalar_operator &other, const product_operator<HandlerTy> &self);       \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator-(const scalar_operator &other, const product_operator<HandlerTy> &self);       \
                                                                                                                    \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator*(double other, const operator_sum<HandlerTy> &self);                           \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator+(double other, const operator_sum<HandlerTy> &self);                           \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator-(double other, const operator_sum<HandlerTy> &self);                           \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator*(std::complex<double> other, const operator_sum<HandlerTy> &self);             \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator+(std::complex<double> other, const operator_sum<HandlerTy> &self);             \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator-(std::complex<double> other, const operator_sum<HandlerTy> &self);             \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator*(const scalar_operator &other, const operator_sum<HandlerTy> &self);           \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator+(const scalar_operator &other, const operator_sum<HandlerTy> &self);           \
    extern template                                                                                                 \
    operator_sum<HandlerTy> operator-(const scalar_operator &other, const operator_sum<HandlerTy> &self);

EXTERN_TEMPLATE_SPECIALIZATIONS(matrix_operator);
EXTERN_TEMPLATE_SPECIALIZATIONS(spin_operator);
EXTERN_TEMPLATE_SPECIALIZATIONS(boson_operator);

#define EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(op, returnTy)                                                    \
                                                                                                                    \
    extern template                                                                                                 \
    returnTy<matrix_operator> operator op(const product_operator<spin_operator> &other,                             \
                                          const product_operator<matrix_operator> &self);                           \
    extern template                                                                                                 \
    returnTy<matrix_operator> operator op(const product_operator<boson_operator> &other,                            \
                                          const product_operator<matrix_operator> &self);                           \
    extern template                                                                                                 \
    returnTy<matrix_operator> operator op(const product_operator<spin_operator> &other,                             \
                                          const product_operator<boson_operator> &self);                            \
    extern template                                                                                                 \
    returnTy<matrix_operator> operator op(const product_operator<boson_operator> &other,                            \
                                          const product_operator<spin_operator> &self);                             \
                                                                                                                    \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,                             \
                                              const product_operator<matrix_operator> &self);                       \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,                            \
                                              const product_operator<matrix_operator> &self);                       \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,                             \
                                              const product_operator<boson_operator> &self);                        \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,                            \
                                              const product_operator<spin_operator> &self);                         \
                                                                                                                    \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const product_operator<spin_operator> &other,                         \
                                              const operator_sum<matrix_operator> &self);                           \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const product_operator<boson_operator> &other,                        \
                                              const operator_sum<matrix_operator> &self);                           \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const product_operator<spin_operator> &other,                         \
                                              const operator_sum<boson_operator> &self);                            \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const product_operator<boson_operator> &other,                        \
                                              const operator_sum<spin_operator> &self);                             \
                                                                                                                    \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,                             \
                                              const operator_sum<matrix_operator> &self);                           \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,                            \
                                              const operator_sum<matrix_operator> &self);                           \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const operator_sum<spin_operator> &other,                             \
                                              const operator_sum<boson_operator> &self);                            \
    extern template                                                                                                 \
    operator_sum<matrix_operator> operator op(const operator_sum<boson_operator> &other,                            \
                                              const operator_sum<spin_operator> &self);                             \

EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(*, product_operator);
EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(+, operator_sum);
EXTERN_CONVERSION_TEMPLATE_SPECIALIZATIONS(-, operator_sum);
#endif

} // namespace cudaq