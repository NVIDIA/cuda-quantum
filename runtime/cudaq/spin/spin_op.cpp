/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include <cudaq/spin_op.h>
#include <stdint.h>
#ifdef CUDAQ_HAS_OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace cudaq {

namespace details {

/// @brief Compute the action
std::pair<std::string, std::complex<double>>
actionOnBra(spin_op &term, const std::string &bitConfiguration) {
  auto coeff = term.get_coefficient();
  auto newConfiguration = bitConfiguration;
  std::complex<double> i(0, 1);

  term.for_each_pauli([&](pauli p, std::size_t idx) {
    if (p == pauli::Z) {
      coeff *= (newConfiguration[idx] == '1' ? -1 : 1);
    } else if (p == pauli::X) {
      newConfiguration[idx] = newConfiguration[idx] == '1' ? '0' : '1';
    } else if (p == pauli::Y) {
      coeff *= (newConfiguration[idx] == '1' ? i : -i);
      newConfiguration[idx] = (newConfiguration[idx] == '1' ? '0' : '1');
    }
  });

  return std::make_pair(newConfiguration, coeff);
}

std::pair<std::complex<double>, std::vector<bool>>
mult(std::vector<bool> row, std::vector<bool> other_row,
     std::complex<double> &rowCoeff, std::complex<double> &otherCoeff) {
  // This is term_i * otherTerm_j
  std::vector<bool> tmp(row.size()), tmp2(row.size());
  std::size_t numQubits = row.size() / 2;

  for (std::size_t i = 0; i < 2 * numQubits; i++)
    tmp[i] = row[i] ^ other_row[i];

  for (std::size_t i = 0; i < numQubits; i++)
    tmp2[i] = row[i] && other_row[numQubits + i];

  int orig_phase = 0, other_phase = 0;
  for (std::size_t i = 0; i < numQubits; i++) {
    if (row[i] && row[i + numQubits])
      orig_phase++;

    if (other_row[i] && other_row[i + numQubits])
      other_phase++;
  }

  auto _phase = orig_phase + other_phase;
  int sum = 0;
  for (auto a : tmp2)
    if (a)
      sum++;

  _phase += 2 * sum;
  // Based on the phase, figure out an extra coeff to apply
  for (std::size_t i = 0; i < numQubits; i++)
    if (tmp[i] && tmp[i + numQubits])
      _phase -= 1;

  _phase %= 4;
  std::complex<double> imaginary(0, 1);
  std::array<std::complex<double>, 4> phaseCoeffArr{1.0, -1. * imaginary, -1.0,
                                                    imaginary};
  auto phase_coeff = phaseCoeffArr[_phase];
  auto coeff = rowCoeff;
  coeff *= phase_coeff * otherCoeff;
  return std::make_pair(coeff, tmp);
}
} // namespace details

spin_op::spin_op() {
  std::vector<bool> init(2);
  terms.emplace(init, 1.0);
}

spin_op::spin_op(
    const std::unordered_map<spin_op_term, std::complex<double>> &_terms)
    : terms(_terms) {}

spin_op::spin_op(std::size_t numQubits) {
  std::vector<bool> init(2 * numQubits);
  terms.emplace(init, 1.0);
}

spin_op::spin_op(const spin_op_term &term, const std::complex<double> &coeff) {
  terms.emplace(term, coeff);
}

spin_op::spin_op(const std::vector<spin_op_term> &bsf,
                 const std::vector<std::complex<double>> &coeffs) {
  for (std::size_t i = 0; auto &t : bsf)
    terms.emplace(t, coeffs[i++]);
}

spin_op::spin_op(pauli type, const std::size_t idx,
                 std::complex<double> coeff) {
  auto numQubits = idx + 1;
  std::vector<bool> d(2 * numQubits);

  if (type == pauli::X)
    d[idx] = 1;
  else if (type == pauli::Y) {
    d[idx] = 1;
    d[idx + numQubits] = 1;
  } else if (type == pauli::Z)
    d[idx + numQubits] = 1;

  terms.emplace(d, coeff);
}

spin_op::spin_op(const spin_op &o) : terms(o.terms) {}

spin_op::spin_op(
    std::pair<const spin_op_term, std::complex<double>> &termData) {
  terms.emplace(termData);
}
spin_op::spin_op(
    const std::pair<const spin_op_term, std::complex<double>> &termData) {
  terms.emplace(termData);
}

spin_op::iterator<spin_op> spin_op::begin() {
  auto startIter = terms.begin();
  return iterator<spin_op>(startIter);
}

spin_op::iterator<spin_op> spin_op::end() {
  auto endIter = terms.end();
  return iterator<spin_op>(endIter);
}


spin_op::iterator<const spin_op> spin_op::begin() const {
  auto startIter = terms.cbegin();
  return iterator<const spin_op>(startIter);
}

spin_op::iterator<const spin_op> spin_op::end() const {
  auto endIter = terms.cend();
  return iterator<const spin_op>(endIter);
}

complex_matrix spin_op::to_matrix() const {
  auto n = num_qubits();
  auto dim = 1UL << n;
  auto getBitStrForIdx = [&](std::size_t i) {
    std::stringstream s;
    for (int k = n - 1; k >= 0; k--)
      s << ((i >> k) & 1);
    return s.str();
  };

  // To construct the matrix, we are looping over every
  // row, computing the binary representation for that index,
  // e.g <100110|, and then we will compute the action of
  // each pauli term on that binary configuration, returning a new
  // product state and coefficient. Call this new state <colState|,
  // we then compute <rowState | Paulis | colState> and set it in the matrix
  // data.

  complex_matrix A(dim, dim);
  A.set_zero();
  auto rawData = A.data();
#pragma omp parallel for shared(rawData)
  for (std::size_t rowIdx = 0; rowIdx < dim; rowIdx++) {
    auto rowBitStr = getBitStrForIdx(rowIdx);
    for_each_term([&](spin_op &term) {
      auto [res, coeff] = details::actionOnBra(term, rowBitStr);
      auto colIdx = std::stol(res, nullptr, 2);
      rawData[rowIdx * dim + colIdx] += coeff;
    });
  }
  return A;
}

std::complex<double> spin_op::get_coefficient() const {
  if (terms.size() != 1)
    throw std::runtime_error(
        "spin_op::get_coefficient called on spin_op with > 1 terms.");
  return terms.begin()->second;
}

void spin_op::for_each_term(std::function<void(spin_op &)> &&functor) const {
  for (auto iter = terms.begin(), e = terms.end(); iter != e; ++iter) {
    const auto &pair = *iter;
    spin_op tmp(pair);
    functor(tmp);
  }
}
void spin_op::for_each_pauli(
    std::function<void(pauli, std::size_t)> &&functor) const {
  if (num_terms() != 1)
    throw std::runtime_error(
        "spin_op::for_each_pauli on valid for spin_op with n_terms == 1.");

  auto nQ = num_qubits();
  auto bsf = terms.begin()->first;
  for (std::size_t i = 0; i < nQ; i++) {
    if (bsf[i] && bsf[i + nQ]) {
      functor(pauli::Y, i);
    } else if (bsf[i]) {
      functor(pauli::X, i);
    } else if (bsf[i + nQ]) {
      functor(pauli::Z, i);
    } else {
      functor(pauli::I, i);
    }
  }
}

spin_op spin_op::random(std::size_t nQubits, std::size_t nTerms) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<std::complex<double>> coeffs(nTerms, 1.0);
  std::vector<spin_op_term> randomTerms;
  for (std::size_t i = 0; i < nTerms; i++) {
    std::vector<bool> termData(2 * nQubits);
    std::fill_n(termData.begin(), termData.size() * (1 - .5), 1);
    std::shuffle(termData.begin(), termData.end(), gen);
    randomTerms.push_back(termData);
  }

  return spin_op(randomTerms, coeffs);
}

void spin_op::expandToNQubits(const std::size_t numQubits) {
  auto iter = terms.begin();
  while (iter != terms.end()) {
    auto coeff = iter->second;
    std::vector<bool> tmp = iter->first;
    if (tmp.size() == numQubits * 2) {
      iter++;
      continue;
    }

    auto newSize = numQubits * 2 - tmp.size();
    for (std::size_t i = 0; i < newSize / 2; i++) {
      tmp.insert(tmp.begin() + tmp.size() / 2, 0);
      tmp.insert(tmp.begin() + tmp.size(), 0);
    }

    terms.erase(iter++);
    terms.emplace(tmp, coeff);
  }
}

spin_op &spin_op::operator+=(const spin_op &v) noexcept {
  auto otherNumQubits = v.num_qubits();

  spin_op tmpv = v;
  if (otherNumQubits > num_qubits())
    expandToNQubits(otherNumQubits);
  else if (otherNumQubits < num_qubits())
    tmpv.expandToNQubits(num_qubits());

  for (auto [term, coeff] : tmpv.terms) {
    auto iter = terms.find(term);
    if (iter != terms.end())
      iter->second += coeff;
    else
      terms.emplace(term, coeff);
  }

  return *this;
}

// spin_op spin_op::operator[](const std::size_t term_idx) const {
//   auto start = terms.begin();
//   std::advance(start, term_idx);
//   return spin_op(start->first, start->second);
// }

spin_op &spin_op::operator-=(const spin_op &v) noexcept {
  return operator+=(-1.0 * v);
}

spin_op &spin_op::operator*=(const spin_op &v) noexcept {
  spin_op copy = v;
  if (v.num_qubits() > num_qubits())
    expandToNQubits(copy.num_qubits());
  else if (v.num_qubits() < num_qubits())
    copy.expandToNQubits(num_qubits());

  std::unordered_map<std::vector<bool>, std::complex<double>> newTerms;
  std::size_t ourRow = 0, theirRow = 0;
  std::vector<std::complex<double>> composedCoeffs(num_terms() *
                                                   copy.num_terms());
  std::vector<std::vector<bool>> composition(num_terms() * copy.num_terms());
  std::map<std::size_t, std::pair<std::size_t, std::size_t>> indexMap;
  auto nElements = composition.size();
  for (std::size_t i = 0; i < nElements; i++) {
    auto pair = std::make_pair(ourRow, theirRow);
    indexMap.emplace(i, pair);
    if (theirRow == copy.num_terms() - 1) {
      theirRow = 0;
      ourRow++;
    } else
      theirRow++;
  }

#pragma omp parallel for shared(composition)
  for (std::size_t i = 0; i < nElements; i++) {
    auto [j, k] = indexMap[i];
    auto s = terms.begin();
    auto t = copy.terms.begin();
    std::advance(s, j);
    std::advance(t, k);
    auto res = details::mult(s->first, t->first, s->second, t->second);
    composition[i] = res.second;
    composedCoeffs[i] = res.first;
  }

  for (std::size_t i = 0; i < nElements; i++) {
    auto iter = newTerms.find(composition[i]);
    if (iter == newTerms.end())
      newTerms.emplace(composition[i], composedCoeffs[i]);
    else
      iter->second += composedCoeffs[i];
  }

  terms = newTerms;

  return *this;
}

bool spin_op::is_identity() const {
  for (auto &[row, c] : terms)
    for (auto e : row)
      if (e)
        return false;

  return true;
}

bool spin_op::operator==(const spin_op &v) const noexcept {
  // Could be that the term is identity with all zeros
  bool isId1 = true, isId2 = true;
  for (auto &[row, c] : terms)
    for (auto e : row)
      if (e) {
        isId1 = false;
        break;
      }

  for (auto &[row, c] : v.terms)
    for (auto e : row)
      if (e) {
        isId2 = false;
        break;
      }

  if (isId1 && isId2)
    return true;

  for (auto &[k, c] : terms) {
    if (v.terms.find(k) == v.terms.end())
      return false;
  }
  return true;
}

spin_op &spin_op::operator*=(const double v) noexcept {
  for (auto &[term, coeff] : terms)
    coeff *= v;

  return *this;
}

spin_op &spin_op::operator*=(const std::complex<double> v) noexcept {
  for (auto &[term, coeff] : terms)
    coeff *= v;

  return *this;
}

std::size_t spin_op::num_qubits() const {
  if (terms.empty())
    return 0;
  return terms.begin()->first.size() / 2;
}
std::size_t spin_op::num_terms() const { return terms.size(); }

spin_op spin_op::slice(const std::size_t startIdx, const std::size_t count) {
  auto nTerms = num_terms();
  if (nTerms <= count)
    throw std::runtime_error("Cannot request slice with " +
                             std::to_string(count) + " terms on spin_op with " +
                             std::to_string(nTerms) + " terms.");
  auto start = terms.begin();
  std::advance(start, startIdx);
  std::unordered_map<spin_op_term, std::complex<double>> sliced;
  std::copy_n(start, count, std::inserter(sliced, sliced.end()));
  return spin_op(sliced);
}

std::string spin_op::to_string(bool printCoeffs) const {
  std::stringstream ss;
  std::vector<std::string> printOut;
  for (auto &[term, coeff] : terms) {
    for (std::size_t i = 0; i < term.size() / 2; i++) {
      if (term[i] && term[i + term.size() / 2])
        printOut.push_back("Y");
      else if (term[i])
        printOut.push_back("X");
      else if (term[i + term.size() / 2])
        printOut.push_back("Z");
      else
        printOut.push_back("I");
    }

    if (printCoeffs)
      ss << fmt::format("[{}{}{}j]", coeff.real(),
                        coeff.imag() < 0.0 ? "-" : "+", std::fabs(coeff.imag()))
         << " ";

    ss << fmt::format("{}", fmt::join(printOut, ""));

    if (printCoeffs)
      ss << "\n";
    printOut.clear();
  }

  return ss.str();
}

void spin_op::dump() const {
  auto str = to_string();
  std::cout << str;
}

spin_op::spin_op(std::vector<double> &input_vec, std::size_t nQubits) {
  auto n_terms = (int)input_vec.back();
  if (nQubits != ((input_vec.size() - 2 * n_terms) / n_terms))
    throw std::runtime_error("Invalid data representation for construction "
                             "spin_op. Number of data elements is incorrect.");

  for (std::size_t i = 0; i < input_vec.size() - 1; i += nQubits + 2) {
    std::vector<bool> tmpv(2 * nQubits);
    for (std::size_t j = 0; j < nQubits; j++) {
      double intPart;
      if (std::modf(input_vec[j + i], &intPart) != 0.0)
        throw std::runtime_error(
            "Invalid pauli data element, must be integer value.");

      int val = (int)input_vec[j + i];
      if (val == 1) { // X
        tmpv[j] = 1;
      } else if (val == 2) { // Z
        tmpv[j + nQubits] = 1;
      } else if (val == 3) { // Y
        tmpv[j + nQubits] = 1;
        tmpv[j] = 1;
      }
    }
    auto el_real = input_vec[i + nQubits];
    auto el_imag = input_vec[i + nQubits + 1];
    terms.emplace(tmpv, std::complex<double>{el_real, el_imag});
  }
}

std::pair<std::vector<spin_op::spin_op_term>, std::vector<std::complex<double>>>
spin_op::get_raw_data() const {
  std::vector<spin_op_term> data;
  std::vector<std::complex<double>> coeffs;
  for (auto &[term, c] : terms) {
    data.push_back(term);
    coeffs.push_back(c);
  }

  return std::make_pair(data, coeffs);
}

spin_op &spin_op::operator=(const spin_op &other) {
  terms = other.terms;
  return *this;
}

spin_op operator+(double coeff, spin_op op) {
  return spin_op(op.num_qubits()) * coeff + op;
}
spin_op operator+(spin_op op, double coeff) {
  return op + spin_op(op.num_qubits()) * coeff;
}
spin_op operator-(double coeff, spin_op op) {
  return spin_op(op.num_qubits()) * coeff - op;
}
spin_op operator-(spin_op op, double coeff) {
  return op - spin_op(op.num_qubits()) * coeff;
}

namespace spin {
spin_op i(const std::size_t idx) { return spin_op(pauli::I, idx); }
spin_op x(const std::size_t idx) { return spin_op(pauli::X, idx); }
spin_op y(const std::size_t idx) { return spin_op(pauli::Y, idx); }
spin_op z(const std::size_t idx) { return spin_op(pauli::Z, idx); }
} // namespace spin

std::vector<double> spin_op::getDataRepresentation() {
  std::vector<double> dataVec;
  for (auto &[term, coeff] : terms) {
    auto nq = terms.size() / 2;
    for (std::size_t i = 0; i < nq; i++) {
      if (term[i] && term[i + nq]) {
        dataVec.push_back(3.);
      } else if (term[i]) {
        dataVec.push_back(1.);
      } else if (term[i + nq]) {
        dataVec.push_back(2.);
      } else {
        dataVec.push_back(0.);
      }
    }
    dataVec.push_back(coeff.real());
    dataVec.push_back(coeff.imag());
  }
  dataVec.push_back(num_terms());
  return dataVec;
}

spin_op binary_spin_op_reader::read(const std::string &data_filename) {
  std::ifstream input(data_filename, std::ios::binary);
  if (input.fail())
    throw std::runtime_error(data_filename + " does not exist.");

  input.seekg(0, std::ios_base::end);
  std::size_t size = input.tellg();
  input.seekg(0, std::ios_base::beg);
  std::vector<double> input_vec(size / sizeof(double));
  input.read((char *)&input_vec[0], size);
  auto n_terms = (int)input_vec.back();
  auto nQubits = (input_vec.size() - 2 * n_terms) / n_terms;
  spin_op s(input_vec, nQubits);
  return s;
}
} // namespace cudaq
