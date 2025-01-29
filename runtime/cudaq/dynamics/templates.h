/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//#include <type_traits>
#include <concepts> 
#include <complex>

namespace cudaq {

class scalar_operator;

class elementary_operator;

template <typename HandlerTy> 
class product_operator;

template <typename HandlerTy> 
class operator_sum;

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
template <typename HandlerTy> 
product_operator<HandlerTy> operator*(const HandlerTy &other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(const HandlerTy &other, const product_operator<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(const HandlerTy &other, const product_operator<HandlerTy> &self);

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
template <typename HandlerTy> 
operator_sum<HandlerTy> operator*(const HandlerTy &other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator+(const HandlerTy &other, const operator_sum<HandlerTy> &self);
template <typename HandlerTy> 
operator_sum<HandlerTy> operator-(const HandlerTy &other, const operator_sum<HandlerTy> &self);


#ifndef CUDAQ_INSTANTIATE_TEMPLATES
extern template 
product_operator<elementary_operator> operator*(double other, const product_operator<elementary_operator> &self);
extern template 
operator_sum<elementary_operator> operator+(double other, const product_operator<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator-(double other, const product_operator<elementary_operator> &self);
extern template
product_operator<elementary_operator> operator*(std::complex<double> other, const product_operator<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator+(std::complex<double> other, const product_operator<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator-(std::complex<double> other, const product_operator<elementary_operator> &self);
extern template
product_operator<elementary_operator> operator*(const scalar_operator &other, const product_operator<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator+(const scalar_operator &other, const product_operator<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator-(const scalar_operator &other, const product_operator<elementary_operator> &self);
extern template
product_operator<elementary_operator> operator*(const elementary_operator &other, const product_operator<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator+(const elementary_operator &other, const product_operator<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator-(const elementary_operator &other, const product_operator<elementary_operator> &self);

extern template
operator_sum<elementary_operator> operator*(double other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator+(double other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator-(double other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator*(std::complex<double> other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator+(std::complex<double> other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator-(std::complex<double> other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator*(const scalar_operator &other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator+(const scalar_operator &other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator-(const scalar_operator &other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator*(const elementary_operator &other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator+(const elementary_operator &other, const operator_sum<elementary_operator> &self);
extern template
operator_sum<elementary_operator> operator-(const elementary_operator &other, const operator_sum<elementary_operator> &self);
#endif

}