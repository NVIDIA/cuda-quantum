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

namespace cudaq {

// std::vector<std::tuple<scalar_operator, HandlerTy>>
// operator_sum<HandlerTy>::canonicalize_product(product_operator<HandlerTy> &prod) const {
//   std::vector<std::tuple<scalar_operator, HandlerTy>>
//       canonicalized_terms;

// std::vector<int> all_degrees;
// std::vector<scalar_operator> scalars;
// std::vector<HandlerTy> non_scalars;

// for (const auto &op : prod.get_operators()) {
//   if (std::holds_alternative<scalar_operator>(op)) {
//     scalars.push_back(*std::get<scalar_operator>(op));
//   } else {
//     non_scalars.push_back(*std::get<HandlerTy>(op));
//     all_degrees.insert(all_degrees.end(),
//                        std::get<HandlerTy>(op).degrees.begin(),
//                        std::get<HandlerTy>(op).degrees.end());
//   }
// }

// if (all_degrees.size() ==
//     std::set<int>(all_degrees.begin(), all_degrees.end()).size()) {
//   std::sort(non_scalars.begin(), non_scalars.end(),
//             [](const HandlerTy &a, const HandlerTy &b) {
//               return a.degrees < b.degrees;
//             });
// }

// for (size_t i = 0; std::min(scalars.size(), non_scalars.size()); i++) {
//   canonicalized_terms.push_back(std::make_tuple(scalars[i], non_scalars[i]));
// }

//   return canonicalized_terms;
// }

// std::vector<std::tuple<scalar_operator, HandlerTy>>
// operator_sum<HandlerTy>::_canonical_terms() const {
//   std::vector<std::tuple<scalar_operator, HandlerTy>> terms;
//   // for (const auto &term : terms) {
//   //   auto canonicalized = canonicalize_product(term);
//   //   terms.insert(terms.end(), canonicalized.begin(), canonicalized.end());
//   // }

//   // std::sort(terms.begin(), terms.end(), [](const auto &a, const auto &b) {
//   //   // return std::to_string(product_operator(a)) <
//   //   //        std::to_string(product_operator(b));
//   //   return product_operator(a).to_string() <
//   product_operator(b).to_string();
//   // });

//   return terms;
// }

// operator_sum<HandlerTy> operator_sum<HandlerTy>::canonicalize() const {
//   std::vector<product_operator> canonical_terms;
//   for (const auto &term : _canonical_terms()) {
//     canonical_terms.push_back(product_operator(term));
//   }
//   return operator_sum(canonical_terms);
// }

// bool operator_sum<HandlerTy>::operator==(const operator_sum<HandlerTy> &other) const {
// return _canonical_terms() == other._canonical_terms();
// }

// // Degrees property
// std::vector<int> operator_sum<HandlerTy>::degrees() const {
//   std::set<int> unique_degrees;
//   for (const auto &term : terms) {
//     for (const auto &op : term.get_operators()) {
//       unique_degrees.insert(op.get_degrees().begin(),
//       op.get_degrees().end());
//     }
//   }

//   return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
// }

// // Parameters property
// std::map<std::string, std::string> operator_sum<HandlerTy>::parameters() const {
//   std::map<std::string, std::string> param_map;
//   for (const auto &term : terms) {
//     for (const auto &op : term.get_operators()) {
//       auto op_params = op.parameters();
//       param_map.insert(op_params.begin(), op.params.end());
//     }
//   }

//   return param_map;
// }

// // Check if all terms are spin operators
// bool operator_sum<HandlerTy>::_is_spinop() const {
//   return std::all_of(
//       terms.begin(), terms.end(), [](product_operator<HandlerTy> &term) {
//         return std::all_of(term.get_operators().begin(),
//                            term.get_operators().end(),
//                            [](const Operator &op) { return op.is_spinop();
//                            });
//       });
// }

// evaluations

/// FIXME:
// tensor
// operator_sum<HandlerTy>::to_matrix(const std::map<int, int> &dimensions,
//                         const std::map<std::string, double> &params) const {
// // todo
// }

// std::string operator_sum<HandlerTy>::to_string() const {
//   std::string result;
//   // for (const auto &term : terms) {
//   //   result += term.to_string() + " + ";
//   // }
//   // // Remove last " + "
//   // if (!result.empty())
//   //   result.pop_back();
//   return result;
// }

// left-hand arithmetics

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(double other, const operator_sum<HandlerTy> &self) {
  return scalar_operator(other) * self;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator+(double other, const operator_sum<HandlerTy> &self) {
  return scalar_operator(other) + self;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator-(double other, const operator_sum<HandlerTy> &self) {
  return scalar_operator(other) - self;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(std::complex<double> other, const operator_sum<HandlerTy> &self) {
  return scalar_operator(other) * self;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator+(std::complex<double> other, const operator_sum<HandlerTy> &self) {
  return scalar_operator(other) + self;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator-(std::complex<double> other, const operator_sum<HandlerTy> &self) {
  return scalar_operator(other) - self;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(const scalar_operator &other, const operator_sum<HandlerTy> &self) {
  std::vector<product_operator<HandlerTy>> terms = self.get_terms();
  for (auto &term : terms)
    term = other * term;
  return operator_sum(terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator+(const scalar_operator &other, const operator_sum<HandlerTy> &self) {
  std::vector<product_operator<HandlerTy>> terms = self.get_terms();
  terms.insert(terms.begin(), other);
  return operator_sum(terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator-(const scalar_operator &other, const operator_sum<HandlerTy> &self) {
  auto negative_self = (-1. * self);
  std::vector<product_operator<HandlerTy>> terms = negative_self.get_terms();
  terms.insert(terms.begin(), other);
  return operator_sum(terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator*(const HandlerTy &other, const operator_sum<HandlerTy> &self) {
  std::vector<std::variant<scalar_operator, HandlerTy>> new_term = {other};
  std::vector<product_operator<HandlerTy>> _prods = {product_operator(new_term)};
  auto selfOpSum = operator_sum(_prods);
  return selfOpSum * self;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator+(const HandlerTy &other, const operator_sum<HandlerTy> &self) {
  std::vector<std::variant<scalar_operator, HandlerTy>> new_term = {other};
  std::vector<product_operator<HandlerTy>> _prods = {product_operator(new_term)};
  auto selfOpSum = operator_sum(_prods);
  return selfOpSum + self;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator-(const HandlerTy &other, const operator_sum<HandlerTy> &self) {
  std::vector<std::variant<scalar_operator, HandlerTy>> new_term = {other};
  std::vector<product_operator<HandlerTy>> _prods = {product_operator(new_term)};
  auto selfOpSum = operator_sum(_prods);
  return selfOpSum - self;
}

// right-hand arithmetics

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
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const scalar_operator &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
  std::vector<std::variant<scalar_operator, HandlerTy>> _other = {
      other};
  combined_terms.push_back(product_operator(_other));
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const scalar_operator &other) const {
  return *this + (-1.0 * other);
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
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const HandlerTy &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
  std::vector<std::variant<scalar_operator, HandlerTy>> _other = {
      other};
  combined_terms.push_back(product_operator(_other));
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const HandlerTy &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
  combined_terms.push_back((-1. * other));
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator*=(const HandlerTy &other) {
  *this = *this * other;
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator+=(const HandlerTy &other) {
  std::vector<std::variant<scalar_operator, HandlerTy>> _other = {
      other};
  *this = *this + product_operator(_other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy>& operator_sum<HandlerTy>::operator-=(const HandlerTy &other) {
  std::vector<std::variant<scalar_operator, HandlerTy>> _other = {
      other};
  *this = *this - product_operator(_other);
  return *this;
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator*(const product_operator<HandlerTy> &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
  for (auto &term : combined_terms) {
    term *= other;
  }
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator+(const product_operator<HandlerTy> &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
  combined_terms.push_back(other);
  return operator_sum(combined_terms);
}

template <typename HandlerTy>
operator_sum<HandlerTy> operator_sum<HandlerTy>::operator-(const product_operator<HandlerTy> &other) const {
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
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
  auto self_terms = terms;
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
  std::vector<product_operator<HandlerTy>> combined_terms = terms;
  combined_terms.insert(combined_terms.end(),
                        std::make_move_iterator(other.terms.begin()),
                        std::make_move_iterator(other.terms.end()));
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

} // namespace cudaq