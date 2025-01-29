/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "cudaq/operators.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>

namespace cudaq {

// private methods

template<typename HandlerTy>
void product_operator<HandlerTy>::aggregate_terms() {}

template<typename HandlerTy>
template <typename ... Args>
void product_operator<HandlerTy>::aggregate_terms(const HandlerTy &head, Args&& ... args) {
  this->terms[0].push_back(head);
  aggregate_terms(std::forward<Args>(args)...);
}

template
void product_operator<elementary_operator>::aggregate_terms(const elementary_operator &item1, 
                                                            const elementary_operator &item2);

template
void product_operator<elementary_operator>::aggregate_terms(const elementary_operator &item1, 
                                                            const elementary_operator &item2,
                                                            const elementary_operator &item3);

// read-only properties

template<typename HandlerTy>
std::vector<int> product_operator<HandlerTy>::degrees() const {
  std::set<int> unsorted_degrees;
  for (const HandlerTy &term : this->terms[0]) {
    unsorted_degrees.insert(term.degrees.begin(), term.degrees.end());
  }
  auto degrees = std::vector<int>(unsorted_degrees.begin(), unsorted_degrees.end());
  std::sort(degrees.begin(), degrees.end()); // FIXME: DELEGATE ANY CONVENTION RELATED ORDERING TO A GENERAL HELPER FUNCTION
  return degrees;
}

template<typename HandlerTy>
int product_operator<HandlerTy>::n_terms() const { 
  return this->terms[0].size(); 
}

template<typename HandlerTy>
std::vector<HandlerTy> product_operator<HandlerTy>::get_terms() const { 
  return this->terms[0]; 
}

template<typename HandlerTy>
scalar_operator product_operator<HandlerTy>::get_coefficient() const { 
  return this->coefficients[0]; 
}

template
std::vector<int> product_operator<elementary_operator>::degrees() const;

template
int product_operator<elementary_operator>::n_terms() const;

template
std::vector<elementary_operator> product_operator<elementary_operator>::get_terms() const;

template
scalar_operator product_operator<elementary_operator>::get_coefficient() const;

// constructors

template<typename HandlerTy>
template<class... Args, class>
product_operator<HandlerTy>::product_operator(scalar_operator coefficient, const Args&... args) {
  this->coefficients.push_back(std::move(coefficient));
  std::vector<HandlerTy> ops = {};
  ops.reserve(sizeof...(Args));
  this->terms.push_back(ops);
  aggregate_terms(args...);
}

template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(scalar_operator coefficient, const std::vector<HandlerTy> &atomic_operators) { 
  this->terms.push_back(atomic_operators);
  this->coefficients.push_back(std::move(coefficient));
}

template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(scalar_operator coefficient, std::vector<HandlerTy> &&atomic_operators) {
  this->terms.push_back(std::move(atomic_operators));
  this->coefficients.push_back(std::move(coefficient));
}

template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(const product_operator<HandlerTy> &other) {
  this->terms = other.terms;
  this->coefficients = other.coefficients;
}

template<typename HandlerTy>
product_operator<HandlerTy>::product_operator(product_operator<HandlerTy> &&other) {
  this->terms = std::move(other.terms);
  this->coefficients = std::move(other.coefficients);
}

template 
product_operator<elementary_operator>::product_operator(scalar_operator coefficient);

template 
product_operator<elementary_operator>::product_operator(scalar_operator coefficient,
                                                        const elementary_operator &item1);

template 
product_operator<elementary_operator>::product_operator(scalar_operator coefficient,
                                                        const elementary_operator &item1,
                                                        const elementary_operator &item2);

template 
product_operator<elementary_operator>::product_operator(scalar_operator coefficient,
                                                        const elementary_operator &item1,
                                                        const elementary_operator &item2,
                                                        const elementary_operator &item3);

template
product_operator<elementary_operator>::product_operator(scalar_operator coefficient, const std::vector<elementary_operator> &atomic_operators);

template
product_operator<elementary_operator>::product_operator(scalar_operator coefficient, std::vector<elementary_operator> &&atomic_operators);

template
product_operator<elementary_operator>::product_operator(const product_operator<elementary_operator> &other);

template
product_operator<elementary_operator>::product_operator(product_operator<elementary_operator> &&other);

// assignments

template<typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator=(const product_operator<HandlerTy> &other) {
  if (this != &other) {
    this->terms = other.terms;
    this->coefficients = other.coefficients;
  }
  return *this;
}

template<typename HandlerTy>
product_operator<HandlerTy>& product_operator<HandlerTy>::operator=(product_operator<HandlerTy> &&other) {
  if (this != &other) {
    this->coefficients = std::move(other.coefficients);
    this->terms = std::move(other.terms);
  }
  return *this;
}

template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator=(const product_operator<elementary_operator> &other);

template
product_operator<elementary_operator>& product_operator<elementary_operator>::operator=(product_operator<elementary_operator> &&other);

// evaluations

template<typename HandlerTy>
std::string product_operator<HandlerTy>::to_string() const {
  throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
matrix_2 product_operator<HandlerTy>::to_matrix(std::map<int, int> dimensions,
                                                std::map<std::string, std::complex<double>> parameters) const {
  if (this->get_coefficient() != scalar_operator(1.) || this->n_terms() != 1)
    throw std::runtime_error("not implemented");
  return this->get_terms()[0].to_matrix(dimensions, parameters);
}

template
std::string product_operator<elementary_operator>::to_string() const;

template
matrix_2 product_operator<elementary_operator>::to_matrix(std::map<int, int> dimensions,
                                                          std::map<std::string, std::complex<double>> parameters) const;

// comparisons

template<typename HandlerTy>
bool product_operator<HandlerTy>::operator==(const product_operator<HandlerTy> &other) const {
  throw std::runtime_error("not implemented");
}

template
bool product_operator<elementary_operator>::operator==(const product_operator<elementary_operator> &other) const;

// unary operators

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator-() const {
  return product_operator<HandlerTy>(-1. * this->coefficients[0], this->terms[0]);
}

template <typename HandlerTy>
product_operator<HandlerTy> product_operator<HandlerTy>::operator+() const {
  return *this;
}

template 
product_operator<elementary_operator> product_operator<elementary_operator>::operator-() const;

template
product_operator<elementary_operator> product_operator<elementary_operator>::operator+() const;

// left-hand arithmetics

#define PRODUCT_MULTIPLICATION_REVERSE(otherTy)                                         \
  template <typename HandlerTy>                                                         \
  product_operator<HandlerTy> operator*(otherTy other,                                  \
                                        const product_operator<HandlerTy> &self) {      \
    return product_operator<HandlerTy>(other * self.coefficients[0], self.terms[0]);    \
  }

PRODUCT_MULTIPLICATION_REVERSE(double);
PRODUCT_MULTIPLICATION_REVERSE(std::complex<double>);
PRODUCT_MULTIPLICATION_REVERSE(const scalar_operator &);

#define PRODUCT_ADDITION_REVERSE(otherTy, op)                                           \
  template <typename HandlerTy>                                                         \
  operator_sum<HandlerTy> operator op(otherTy other,                                    \
                                      const product_operator<HandlerTy> &self) {        \
    return operator_sum<HandlerTy>(product_operator<HandlerTy>(other), op self);        \
  }

PRODUCT_ADDITION_REVERSE(double, +);
PRODUCT_ADDITION_REVERSE(double, -);
PRODUCT_ADDITION_REVERSE(std::complex<double>, +);
PRODUCT_ADDITION_REVERSE(std::complex<double>, -);
PRODUCT_ADDITION_REVERSE(const scalar_operator &, +);
PRODUCT_ADDITION_REVERSE(const scalar_operator &, -);

template <typename HandlerTy>
product_operator<HandlerTy> operator*(const HandlerTy &other, const product_operator<HandlerTy> &self) {
  std::vector<HandlerTy> terms;
  terms.reserve(self.terms[0].size() + 1);
  terms.push_back(other);
  for (auto term : self.terms[0])
    terms.push_back(term);
  return product_operator<HandlerTy>(self.coefficients[0], std::move(terms));
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator+(const HandlerTy &other, const product_operator<HandlerTy> &self) {
  return operator_sum<HandlerTy>(product_operator<HandlerTy>(1., other), self);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator-(const HandlerTy &other, const product_operator<HandlerTy> &self) {
  return operator_sum<HandlerTy>(product_operator<HandlerTy>(1., other), -self);
}

template
product_operator<elementary_operator> operator*(double other, const product_operator<elementary_operator> &self);
template
product_operator<elementary_operator> operator*(std::complex<double> other, const product_operator<elementary_operator> &self);
template
product_operator<elementary_operator> operator*(const scalar_operator &other, const product_operator<elementary_operator> &self);
template
product_operator<elementary_operator> operator*(const elementary_operator &other, const product_operator<elementary_operator> &self);
template
operator_sum<elementary_operator> operator+(double other, const product_operator<elementary_operator> &self);
template
operator_sum<elementary_operator> operator+(std::complex<double> other, const product_operator<elementary_operator> &self);
template
operator_sum<elementary_operator> operator+(const scalar_operator &other, const product_operator<elementary_operator> &self);
template
operator_sum<elementary_operator> operator+(const elementary_operator &other, const product_operator<elementary_operator> &self);
template
operator_sum<elementary_operator> operator-(double other, const product_operator<elementary_operator> &self);
template
operator_sum<elementary_operator> operator-(std::complex<double> other, const product_operator<elementary_operator> &self);
template
operator_sum<elementary_operator> operator-(const scalar_operator &other, const product_operator<elementary_operator> &self);
template
operator_sum<elementary_operator> operator-(const elementary_operator &other, const product_operator<elementary_operator> &self);

} // namespace cudaq