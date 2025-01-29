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
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(double other) const {
  return *this * scalar_operator(other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(double other) const {
  return *this + scalar_operator(other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(double other) const {
  return *this - scalar_operator(other);
}

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
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(std::complex<double> other) const {
  return *this * scalar_operator(other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(std::complex<double> other) const {
  return *this + scalar_operator(other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(std::complex<double> other) const {
  return *this - scalar_operator(other);
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
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const scalar_operator &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = std::move(this->get_terms());
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const scalar_operator &other) const {
  // FIXME: reserve length
  auto combined_terms = std::move(this->get_terms()); 
  combined_terms.push_back(product_operator<HandlerTy>(other));
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const scalar_operator &other) const {
  return *this + (other * (-1.0));
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
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const HandlerTy &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = this->get_terms();
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const HandlerTy &other) const {
  auto combined_terms = std::move(this->get_terms()); 
  combined_terms.push_back(product_operator<HandlerTy>(1., other));
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const HandlerTy &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = std::move(this->get_terms());
  combined_terms.push_back(product_operator<HandlerTy>(-1., other));
  return operator_sum(combined_terms);
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
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(double other) const {
  return *this * scalar_operator(other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(double other) const {
  return *this + scalar_operator(other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(double other) const {
  return *this - scalar_operator(other);
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(double other) {
  *this = *this * scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(std::complex<double> other) const {
  return *this * scalar_operator(other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(std::complex<double> other) const {
  return *this + scalar_operator(other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(std::complex<double> other) const {
  return *this - scalar_operator(other);
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(std::complex<double> other) {
  *this = *this * scalar_operator(other);
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(const scalar_operator &other) const {
  return product_operator(this->coefficients[0] * other, this->terms[0]);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(const scalar_operator &other) const {
  product_operator<HandlerTy> coefficient(other);
  return operator_sum<HandlerTy>(coefficient, *this);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(const scalar_operator &other) const {
  product_operator<HandlerTy> coefficient(-1. * other);
  return operator_sum<HandlerTy>(coefficient, *this);
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const scalar_operator &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(const HandlerTy &other) const {
  auto combined_terms = this->terms[0];
  combined_terms.push_back(other);
  return product_operator(1., combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(const HandlerTy &other) const {
  return operator_sum<HandlerTy>(*this, product_operator<HandlerTy>(1., other));
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(const HandlerTy &other) const {
  return operator_sum<HandlerTy>(*this, product_operator<HandlerTy>(-1., other));
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const HandlerTy &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator*(const product_operator<HandlerTy> &other) const {
  auto combined_terms = this->terms[0];
  combined_terms.insert(combined_terms.end(), other.terms[0].begin(), other.terms[0].end());
  return product_operator(1., combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(const product_operator<HandlerTy> &other) const {
  return operator_sum<HandlerTy>(*this, other);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(const product_operator<HandlerTy> &other) const {
  return operator_sum<HandlerTy>(*this, other * (-1.));
}

template <typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator*=(const product_operator<HandlerTy> &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator*(const operator_sum<HandlerTy> &other) const {
  std::vector<product_operator<HandlerTy>> other_terms = other.get_terms();
  for (auto &term : other_terms) {
    term = *this * term;
  }
  return operator_sum<HandlerTy>(other_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator+(const operator_sum<HandlerTy> &other) const {
  std::vector<product_operator> other_terms = other.get_terms();
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum<HandlerTy>(other_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> product_operator<HandlerTy>::operator-(const operator_sum<HandlerTy> &other) const {
  auto negative_other = other * (-1.);
  std::vector<product_operator<HandlerTy>> other_terms = negative_other.get_terms();
  other_terms.insert(other_terms.begin(), *this);
  return operator_sum<HandlerTy>(other_terms);
}


// instantiations

template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator*(double other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator+(double other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator-(double other) const;
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(double other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(double other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(double other);
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator*(std::complex<double> other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator+(std::complex<double> other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator-(std::complex<double> other) const;
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(std::complex<double> other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(std::complex<double> other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(std::complex<double> other);
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator*(const scalar_operator &other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator+(const scalar_operator &other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator-(const scalar_operator &other) const;
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator*=(const scalar_operator &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator+=(const scalar_operator &other);
template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator-=(const scalar_operator &other);
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator*(const elementary_operator &other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator+(const elementary_operator &other) const;
template
operator_sum<elementary_operator> operator_sum<elementary_operator>::operator-(const elementary_operator &other) const;
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
product_operator<elementary_operator> product_operator<elementary_operator>::operator*(double other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator+(double other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator-(double other) const;
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(double other);
template
product_operator<elementary_operator> product_operator<elementary_operator>::operator*(std::complex<double> other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator+(std::complex<double> other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator-(std::complex<double> other) const;
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(std::complex<double> other);
template
product_operator<elementary_operator> product_operator<elementary_operator>::operator*(const scalar_operator &other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator+(const scalar_operator &other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator-(const scalar_operator &other) const;
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(const scalar_operator &other);
template
product_operator<elementary_operator> product_operator<elementary_operator>::operator*(const elementary_operator &other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator+(const elementary_operator &other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator-(const elementary_operator &other) const;
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(const elementary_operator &other);
template
product_operator<elementary_operator> product_operator<elementary_operator>::operator*(const product_operator<elementary_operator> &other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator+(const product_operator<elementary_operator> &other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator-(const product_operator<elementary_operator> &other) const;
template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator*=(const product_operator<elementary_operator> &other);
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator*(const operator_sum<elementary_operator> &other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator+(const operator_sum<elementary_operator> &other) const;
template
operator_sum<elementary_operator> product_operator<elementary_operator>::operator-(const operator_sum<elementary_operator> &other) const;

}