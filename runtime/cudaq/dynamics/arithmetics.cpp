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
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const product_operator<HandlerTy> &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = this->get_terms();
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const product_operator<HandlerTy> &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = this->get_terms();
  combined_terms.push_back(other);
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const product_operator<HandlerTy> &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = this->get_terms();
  combined_terms.push_back(other * (-1.));
  return operator_sum(combined_terms);
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
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const operator_sum<HandlerTy> &other) const {
  auto self_terms = this->get_terms();
  std::vector<product_operator<HandlerTy>> product_terms;
  auto other_terms = other.get_terms();
  for (auto &term : self_terms) {
    for (auto &other_term : other_terms) {
      product_terms.push_back(term * other_term);
    }
  }
  return operator_sum(product_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const operator_sum<HandlerTy> &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = std::move(this->get_terms());
  std::vector<product_operator<HandlerTy>> other_terms = std::move(other.get_terms());
  combined_terms.insert(combined_terms.end(), std::make_move_iterator(other_terms.begin()), std::make_move_iterator(other_terms.end()));
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const operator_sum<HandlerTy> &other) const {
  return *this + (other * (-1));
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
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator*(const product_operator<elementary_operator> &other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator+(const product_operator<elementary_operator> &other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator-(const product_operator<elementary_operator> &other) const;
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(const product_operator<elementary_operator> &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(const product_operator<elementary_operator> &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(const product_operator<elementary_operator> &other);
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator*(const operator_sum<elementary_operator> &other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator+(const operator_sum<elementary_operator> &other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator-(const operator_sum<elementary_operator> &other) const;
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