/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <concepts> 
#include <set>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <numeric>

namespace cudaq {

// operator sum right-hand arithmetics

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(double other) {
  *this *= scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(double other) {
  *this += scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(double other) {
  *this -= scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(std::complex<double> other) {
  *this *= scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(std::complex<double> other) {
  *this += scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(std::complex<double> other) {
  *this -= scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const scalar_operator &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const scalar_operator &other) {
  *this = *this + other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const scalar_operator &other) {
  *this = *this - other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const HandlerTy &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const HandlerTy &other) {
  this->coefficients.push_back(1.);
  this->terms.push_back({other});
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const HandlerTy &other) {
  this->coefficients.push_back(-1.);
  this->terms.push_back({other});
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const product_operator<HandlerTy> &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const product_operator<HandlerTy> &other) {
  *this = *this + other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const product_operator<HandlerTy> &other) {
  *this = *this - other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const operator_sum<HandlerTy> &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const operator_sum<HandlerTy> &other) {
  *this = *this - other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const operator_sum<HandlerTy> &other) {
  *this = *this + other;
  return *this;
}


// product operator right-hand arithmetics

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(double other) {
  *this = *this * scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(std::complex<double> other) {
  *this = *this * scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const scalar_operator &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const HandlerTy &other) {
  *this = *this * other;
  return *this;
}


template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const product_operator<HandlerTy> &other) {
  *this = *this * other;
  return *this;
}


// instantiations

template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(double other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(double other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(double other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(std::complex<double> other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(std::complex<double> other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(std::complex<double> other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(const scalar_operator &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(const scalar_operator &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(const scalar_operator &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(const elementary_operator &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(const elementary_operator &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(const elementary_operator &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(const product_operator<elementary_operator> &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(const product_operator<elementary_operator> &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(const product_operator<elementary_operator> &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(const operator_sum<elementary_operator> &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(const operator_sum<elementary_operator> &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(const operator_sum<elementary_operator> &other);

template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(double other);
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(std::complex<double> other);
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(const scalar_operator &other);
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(const elementary_operator &other);
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(const product_operator<elementary_operator> &other);
}