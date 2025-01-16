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


} // namespace cudaq