/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/EigenDense.h"
#include "cudaq/operators.h"

#include <iostream>
#include <set>
#include <concepts>
#include <type_traits>

namespace cudaq {

// private methods

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
std::vector<std::tuple<scalar_operator, HandlerTy>> operator_sum<HandlerTy>::canonicalize_product(product_operator<HandlerTy> &prod) const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
std::vector<std::tuple<scalar_operator, HandlerTy>> operator_sum<HandlerTy>::_canonical_terms() const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
void operator_sum<HandlerTy>::aggregate_terms(const product_operator<HandlerTy> &head) {
    this->terms.push_back(head.terms[0]);
    this->coefficients.push_back(head.coefficients[0]);
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
template <typename ... Args>
void operator_sum<HandlerTy>::aggregate_terms(const product_operator<HandlerTy> &head, Args&& ... args) {
    this->terms.push_back(head.terms[0]);
    this->coefficients.push_back(head.coefficients[0]);
    aggregate_terms(std::forward<Args>(args)...);
}

template
std::vector<std::tuple<scalar_operator, elementary_operator>> operator_sum<elementary_operator>::canonicalize_product(product_operator<elementary_operator> &prod) const;

template
std::vector<std::tuple<scalar_operator, elementary_operator>> operator_sum<elementary_operator>::_canonical_terms() const;

template
void operator_sum<elementary_operator>::aggregate_terms(const product_operator<elementary_operator> &item1, 
                                                        const product_operator<elementary_operator> &item2);

template
void operator_sum<elementary_operator>::aggregate_terms(const product_operator<elementary_operator> &item1, 
                                                        const product_operator<elementary_operator> &item2,
                                                        const product_operator<elementary_operator> &item3);

// read-only properties

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
std::vector<int> operator_sum<HandlerTy>::degrees() const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
int operator_sum<HandlerTy>::n_terms() const { 
    return this->terms.size(); 
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
std::vector<product_operator<HandlerTy>> operator_sum<HandlerTy>::get_terms() const { 
    std::vector<product_operator<HandlerTy>> prods;
    prods.reserve(this->terms.size());
    for (size_t i = 0; i < this->terms.size(); ++i) {
        prods.push_back(product_operator<HandlerTy>(this->coefficients[i], this->terms[i]));
    }
    return prods; 
}

template
std::vector<int> operator_sum<elementary_operator>::degrees() const;

template
int operator_sum<elementary_operator>::n_terms() const;

template
std::vector<product_operator<elementary_operator>> operator_sum<elementary_operator>::get_terms() const;

// constructors

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
template<class... Args, class>
operator_sum<HandlerTy>::operator_sum(const Args&... args) {
    this->terms.reserve(sizeof...(Args));
    this->coefficients.reserve(sizeof...(Args));
    aggregate_terms(args...);
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
operator_sum<HandlerTy>::operator_sum(const std::vector<product_operator<HandlerTy>> &terms) { 
    this->terms.reserve(terms.size());
    this->coefficients.reserve(terms.size());
    for (const product_operator<HandlerTy>& term : terms) {
        this->terms.push_back(term.terms[0]);
        this->coefficients.push_back(term.coefficients[0]);
    }
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
operator_sum<HandlerTy>::operator_sum(std::vector<product_operator<HandlerTy>> &&terms) { 
    this->terms.reserve(terms.size());
    for (const product_operator<HandlerTy>& term : terms) {
        this->terms.push_back(std::move(term.terms[0]));
        this->coefficients.push_back(std::move(term.coefficients[0]));
    }
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
operator_sum<HandlerTy>::operator_sum(const operator_sum<HandlerTy> &other)
    : coefficients(other.coefficients), terms(other.terms) {}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
operator_sum<HandlerTy>::operator_sum(operator_sum<HandlerTy> &&other) 
    : coefficients(std::move(other.coefficients)), terms(std::move(other.terms)) {}

template 
operator_sum<elementary_operator>::operator_sum(const product_operator<elementary_operator> &item1);

template 
operator_sum<elementary_operator>::operator_sum(const product_operator<elementary_operator> &item1,
                                                const product_operator<elementary_operator> &item2);

template 
operator_sum<elementary_operator>::operator_sum(const product_operator<elementary_operator> &item1,
                                                const product_operator<elementary_operator> &item2,
                                                const product_operator<elementary_operator> &item3);

template
operator_sum<elementary_operator>::operator_sum(const std::vector<product_operator<elementary_operator>> &terms);

template
operator_sum<elementary_operator>::operator_sum(std::vector<product_operator<elementary_operator>> &&terms);

template
operator_sum<elementary_operator>::operator_sum(const operator_sum<elementary_operator> &other);

template
operator_sum<elementary_operator>::operator_sum(operator_sum<elementary_operator> &&other);

// assignments

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(const operator_sum<HandlerTy> &other) {
    if (this != &other) {
        coefficients = other.coefficients;
        terms = other.terms;
    }
    return *this;
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator=(operator_sum<HandlerTy> &&other) {
    if (this != &other) {
        coefficients = std::move(other.coefficients);
        terms = std::move(other.terms);
    }
    return *this;
}

template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator=(const operator_sum<elementary_operator>& other);

template
operator_sum<elementary_operator>& operator_sum<elementary_operator>::operator=(operator_sum<elementary_operator> &&other);

// evaluations

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
std::string operator_sum<HandlerTy>::to_string() const {
    throw std::runtime_error("not implemented");
}

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
matrix_2 operator_sum<HandlerTy>::to_matrix(const std::map<int, int> &dimensions,
                                            const std::map<std::string, double> &params) const {
    throw std::runtime_error("not implemented");
}

template
std::string operator_sum<elementary_operator>::to_string() const;

template
matrix_2 operator_sum<elementary_operator>::to_matrix(const std::map<int, int> &dimensions,
                                                      const std::map<std::string, double> &params) const;

// comparisons

template<typename HandlerTy>
requires std::derived_from<elementary_operator, HandlerTy>
bool operator_sum<HandlerTy>::operator==(const operator_sum<HandlerTy> &other) const {
    throw std::runtime_error("not implemented");
}

template
bool operator_sum<elementary_operator>::operator==(const operator_sum<elementary_operator> &other) const;

} // namespace cudaq