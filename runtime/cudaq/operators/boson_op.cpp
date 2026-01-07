/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <complex>
#include <functional>
#include <unordered_map>
#include <vector>

#include "common/EigenSparse.h"
#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"
#include "helpers.h"

#include "cudaq/boson_op.h"

namespace cudaq {

// private helpers

// used internally and externally to identity the operator -
// use a human friendly string code to make it more comprehensible
std::string boson_handler::op_code_to_string() const {
  // Note that we can (and should) have the same op codes across boson, fermion,
  // and spin ops, since individual operators with the same op codes are
  // actually equal. Note that the matrix definition for creation, annihilation
  // and number operators are equal despite the different
  // commutation/anticommutation relations; what makes them behave differently
  // is effectively the "finite size effects" for fermions. Specifically, if we
  // were to blindly multiply the matrices for d=2 for bosons, we would get the
  // same behavior as we have for a single fermion due to the finite size of the
  // matrix. To avoid this, we ensure that we reorder the operators for bosons
  // appropriately as part of the in-place multiplication, whereas for fermions,
  // this effect is desired/correct.
  if (this->additional_terms == 0 && this->number_offsets.size() == 0)
    return "I";
  std::string str;
  for (auto offset : this->number_offsets) {
    if (offset == 0)
      str += "N";
    else if (offset > 0)
      str += "(N+" + std::to_string(offset) + ")";
    else
      str += "(N" + std::to_string(offset) + ")";
  }
  if (this->additional_terms > 0) {
    str += "Ad";
    if (this->additional_terms > 1)
      str += "^" + std::to_string(this->additional_terms);
  } else if (this->additional_terms < 0) {
    str += "A";
    if (-this->additional_terms > 1)
      str += "^" + std::to_string(-this->additional_terms);
  }
  return str;
}

// used internally for canonical evaluation -
// use a encoding that makes it convenient to reconstruct the operator
std::string boson_handler::canonical_form(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    std::vector<std::int64_t> &relevant_dims) const {
  auto it = dimensions.find(this->degree);
  if (it == dimensions.end())
    throw std::runtime_error("missing dimension for degree " +
                             std::to_string(this->degree));
  relevant_dims.push_back(it->second);

  if (this->additional_terms == 0 && this->number_offsets.size() == 0)
    return "I_";
  std::string str = std::to_string(this->additional_terms);
  for (auto offset : this->number_offsets)
    str += "." + std::to_string(offset);
  return str + "_";
}

void boson_handler::inplace_mult(const boson_handler &other) {
  this->number_offsets.reserve(this->number_offsets.size() +
                               other.number_offsets.size());

  // first permute all number operators of RHS to the left; for x = #
  // permutations, if we have "unpaired" creation operators, the number operator
  // becomes (N + x), if we have "unpaired" annihilation operators, the number
  // operator becomes (N - x).
  auto it = this->number_offsets
                .cbegin(); // we will sort the offsets from biggest to smallest
  for (auto offset : other.number_offsets) {
    while (it != this->number_offsets.cend() &&
           *it >= offset - this->additional_terms)
      ++it;
    this->number_offsets.insert(it, offset - this->additional_terms);
  }

  // now we can combine the creation and annihilation operators;
  if (this->additional_terms > 0) { // we have "unpaired" creation operators
    // using ad*a = N and ad*N = (N - 1)*ad, each created number operator has an
    // offset of -(x - 1 - i), where x is the number of creation operators, and
    // i is the number of creation operators we already combined
    it = this->number_offsets.cbegin();
    for (auto i = std::min(this->additional_terms, -other.additional_terms);
         i > 0; --i) {
      // we make sure to have offsets get smaller as we go to keep the sorting
      // cheap
      while (it != this->number_offsets.cend() &&
             *it >= i - this->additional_terms)
        ++it;
      this->number_offsets.insert(it, i - this->additional_terms);
    }
  } else if (this->additional_terms <
             0) { // we have "unpaired" annihilation operators
    // using a*ad = (N + 1) and a*N = (N + 1)*a, each created number operator
    // has an offset of (x - i), where x is the number of annihilation
    // operators, and i is the number of annihilation operators we already
    // combined
    it = this->number_offsets.cbegin();
    for (auto i = 0; i > this->additional_terms && i > -other.additional_terms;
         --i) {
      // we make sure to have offsets get smaller as we go to keep the sorting
      // cheap
      while (it != this->number_offsets.cend() &&
             *it >= i - this->additional_terms)
        ++it;
      this->number_offsets.insert(it, i - this->additional_terms);
    }
  }

  // finally, we update the number of remaining unpaired operators
  this->additional_terms += other.additional_terms;

#if !defined(NDEBUG)
  // we sort the number offsets, such that the equality comparison and the
  // operator id perfectly reflects the mathematical evaluation of the operator
  auto sorted_offsets = this->number_offsets;
  std::sort(sorted_offsets.begin(), sorted_offsets.end(), std::greater<int>());
  assert(sorted_offsets == this->number_offsets);
#endif
}

// read-only properties

std::string boson_handler::unique_id() const {
  return this->op_code_to_string() + std::to_string(this->degree);
}

std::vector<std::size_t> boson_handler::degrees() const {
  return {this->degree};
}

std::size_t boson_handler::target() const { return this->degree; }

// constructors

boson_handler::boson_handler(std::size_t target)
    : degree(target), additional_terms(0) {}

boson_handler::boson_handler(std::size_t target, int op_id)
    : degree(target), additional_terms(0) {
  assert(0 <= op_id && op_id < 4);
  if (op_id == 1) // create
    this->additional_terms = 1;
  else if (op_id == 2) // annihilate
    this->additional_terms = -1;
  else if (op_id == 3) // number
    this->number_offsets.push_back(0);
}

// evaluations

void boson_handler::create_matrix(
    const std::string &boson_word, const std::vector<std::int64_t> &dimensions,
    const std::function<void(std::size_t, std::size_t, std::complex<double>)>
        &process_element,
    bool invert_order) {
  auto tokenize = [](std::string s, char delim) {
    std::vector<std::string> tokens;
    std::size_t start = 0, end = 0;
    while ((end = s.find(delim, start)) != std::string::npos) {
      tokens.push_back(s.substr(start, end - start));
      start = end + 1;
    }
    tokens.push_back(s.substr(start));
    return tokens;
  };

  auto map_state = [&tokenize](std::string encoding, std::int64_t old_state) {
    if (encoding == "I")
      return std::pair<double, std::int64_t>{1., old_state};
    auto ops = tokenize(encoding, '.');
    assert(ops.size() > 0);

    auto it = ops.cbegin();
    int additional_terms = std::stol(*it);
    std::int64_t new_state = old_state + additional_terms;
    if (new_state < 0)
      return std::pair<double, std::int64_t>{0., old_state};

    double value = 1.;
    if (additional_terms > 0)
      for (auto offset = additional_terms; offset > 0; --offset)
        value *= std::sqrt(old_state + offset);
    if (additional_terms < 0)
      for (auto offset = -additional_terms; offset > 0; --offset)
        value *= std::sqrt(new_state + offset);
    while (++it != ops.cend())
      value *= (new_state + std::stol(*it));
    return std::pair<double, std::int64_t>{value, new_state};
  };

  auto states = cudaq::detail::generate_all_states(dimensions);
  if (states.size() == 0)
    process_element(0, 0, 1.);
  std::vector<std::string> boson_terms = tokenize(boson_word, '_');
  std::size_t old_state_idx = 0;
  for (const auto &old_state : states) {
    std::vector<std::int64_t> new_state(old_state.size(), 0);
    std::complex<double> entry = 1.;
    for (std::size_t degree = 0; degree < old_state.size(); ++degree) {
      auto state = old_state[degree];
      auto op =
          boson_terms[invert_order ? old_state.size() - 1 - degree : degree];
      auto mapped = map_state(op, state);
      entry *= mapped.first;
      if (mapped.second >= dimensions[degree])
        entry = 0.;
      else
        new_state[degree] = mapped.second;
    }

    if (entry != 0.) {
      auto new_state_idx = 0;
      for (std::size_t idx = 0; idx < new_state.size(); ++idx) {
        auto offset = 1;
        for (std::size_t d = 0; d < idx; ++d)
          offset *= dimensions[d];
        new_state_idx += new_state[idx] * offset;
      }
      process_element(new_state_idx, old_state_idx, entry);
    }
    old_state_idx += 1;
  }
}

cudaq::detail::EigenSparseMatrix
boson_handler::to_sparse_matrix(const std::string &boson_word,
                                const std::vector<std::int64_t> &dimensions,
                                std::complex<double> coeff, bool invert_order) {
  std::int64_t dim = 1;
  for (auto d : dimensions)
    dim *= d;
  return cudaq::detail::create_sparse_matrix(
      dim, coeff,
      [&boson_word, &dimensions, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(boson_word, dimensions, process_entry, invert_order);
      });
}

complex_matrix
boson_handler::to_matrix(const std::string &boson_word,
                         const std::vector<std::int64_t> &dimensions,
                         std::complex<double> coeff, bool invert_order) {
  std::int64_t dim = 1;
  for (auto d : dimensions)
    dim *= d;
  return cudaq::detail::create_matrix(
      dim, coeff,
      [&boson_word, &dimensions, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(boson_word, dimensions, process_entry, invert_order);
      });
}

complex_matrix boson_handler::to_matrix(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  std::vector<std::int64_t> relevant_dims;
  auto boson_word = this->canonical_form(dimensions, relevant_dims);
  return boson_handler::to_matrix(boson_word, relevant_dims);
}

mdiag_sparse_matrix boson_handler::to_diagonal_matrix(
    std::unordered_map<std::size_t, std::int64_t> &dimensions,
    const std::unordered_map<std::string, std::complex<double>> &parameters)
    const {
  std::vector<std::int64_t> relevant_dims;
  return boson_handler::to_diagonal_matrix(
      this->canonical_form(dimensions, relevant_dims), relevant_dims);
}

mdiag_sparse_matrix boson_handler::to_diagonal_matrix(
    const std::string &boson_word, const std::vector<std::int64_t> &dimensions,
    std::complex<double> coeff, bool invert_order) {
  std::int64_t dim = 1;
  for (auto d : dimensions)
    dim *= d;
  return cudaq::detail::create_mdiag_sparse_matrix(
      dim, coeff,
      [&boson_word, &dimensions, invert_order](
          const std::function<void(std::size_t, std::size_t,
                                   std::complex<double>)> &process_entry) {
        create_matrix(boson_word, dimensions, process_entry, invert_order);
      });
}

std::string boson_handler::to_string(bool include_degrees) const {
  if (include_degrees)
    return this->unique_id(); // unique id for consistency with keys in some
                              // user facing maps
  else
    return this->op_code_to_string();
}

// comparisons

bool boson_handler::operator==(const boson_handler &other) const {
  return this->degree == other.degree &&
         this->additional_terms == other.additional_terms &&
         this->number_offsets == other.number_offsets;
}

// defined operators

boson_handler boson_handler::create(std::size_t degree) {
  return boson_handler(degree, 1);
}

boson_handler boson_handler::annihilate(std::size_t degree) {
  return boson_handler(degree, 2);
}

boson_handler boson_handler::number(std::size_t degree) {
  return boson_handler(degree, 3);
}

namespace boson {
product_op<boson_handler> create(std::size_t target) {
  return product_op(boson_handler::create(target));
}
product_op<boson_handler> annihilate(std::size_t target) {
  return product_op(boson_handler::annihilate(target));
}
product_op<boson_handler> number(std::size_t target) {
  return product_op(boson_handler::number(target));
}
sum_op<boson_handler> position(std::size_t target) {
  return sum_op<boson_handler>(0.5 * create(target), 0.5 * annihilate(target));
}
sum_op<boson_handler> momentum(std::size_t target) {
  return sum_op<boson_handler>(std::complex<double>(0., 0.5) * create(target),
                               std::complex<double>(0., -0.5) *
                                   annihilate(target));
}
} // namespace boson

} // namespace cudaq
