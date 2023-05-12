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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#include <Eigen/Dense>
#pragma clang diagnostic pop
#include <algorithm>
#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace cudaq {

/// @brief Compute the action
/// @param term
/// @param bitConfiguration
/// @return
std::pair<std::string, std::complex<double>>
actionOnBra(spin_op &term, const std::string &bitConfiguration) {
  auto coeff = term.get_coefficients()[0];
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

complex_matrix spin_op::to_matrix() const {
  auto n = n_qubits();
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
      auto [res, coeff] = actionOnBra(term, rowBitStr);
      auto colIdx = std::stol(res, nullptr, 2);
      rawData[rowIdx * dim + colIdx] += coeff;
    });
  }
  return A;
}

void spin_op::for_each_term(std::function<void(spin_op &)> &&functor) const {
  for (std::size_t i = 0; i < n_terms(); i++) {
    auto term = operator[](i);
    functor(term);
  }
}
void spin_op::for_each_pauli(
    std::function<void(pauli, std::size_t)> &&functor) const {
  if (n_terms() != 1)
    throw std::runtime_error(
        "spin_op::for_each_pauli on valid for spin_op with n_terms == 1.");

  auto nQ = n_qubits();
  auto bsf = get_bsf()[0];
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
  std::vector<std::complex<double>> coeff(nTerms, 1.0);
  std::vector<std::vector<bool>> randomTerms;
  for (std::size_t i = 0; i < nTerms; i++) {
    std::vector<bool> termData(2 * nQubits);
    std::fill_n(termData.begin(), termData.size() * (1 - .5), 1);
    std::shuffle(termData.begin(), termData.end(), gen);
    randomTerms.push_back(termData);
  }

  return cudaq::spin_op::from_binary_symplectic(randomTerms, coeff);
}

void spin_op::expandToNQubits(const std::size_t n_q) {
  for (auto &row : data) {
    std::vector<bool> tmp(n_q * 2);
    for (std::size_t k = 0; k < m_n_qubits; k++)
      if (row[k] && row[k + m_n_qubits]) {
        tmp[k] = 1;
        tmp[k + n_q] = 1;
      } else if (row[k])
        tmp[k] = row[k];
      else if (row[k + m_n_qubits])
        tmp[k + n_q] = true;
    row = tmp;
  }
  m_n_qubits = n_q;
}

spin_op::spin_op() {
  coefficients.push_back(1.0);
  // Should initialize with 2 elements for a 1 qubit Identity.
  data.push_back(std::vector<bool>(2));
}

spin_op::spin_op(BinarySymplecticForm d,
                 std::vector<std::complex<double>> coeffs)
    : data(d), coefficients(coeffs) {
  m_n_qubits = data[0].size() / 2.;
}

spin_op::spin_op(pauli type, const std::size_t idx,
                 std::complex<double> coeff) {
  m_n_qubits = idx + 1;
  [[maybe_unused]] int p = 0;
  data.push_back(std::vector<bool>(2 * m_n_qubits));
  if (type == pauli::X) {
    data.back()[idx] = true;
  } else if (type == pauli::Y) {
    p++;
    data.back()[idx] = true;
    data.back()[idx + m_n_qubits] = true;
  } else if (type == pauli::Z) {
    data.back()[idx + m_n_qubits] = true;
  }

  coefficients.push_back(coeff);
}

spin_op::spin_op(const spin_op &o)
    : data(o.data), coefficients(o.coefficients), m_n_qubits(o.m_n_qubits) {}

spin_op &spin_op::operator+=(const spin_op &v) noexcept {
  spin_op tmpv = v;
  if (v.m_n_qubits > m_n_qubits) {
    // If we are adding a op that has more qubits than we do
    // then we need to resize, making sure to ensure the
    // correct 1/0 positions.
    expandToNQubits(v.m_n_qubits);
  } else if (v.m_n_qubits < m_n_qubits) {
    tmpv.expandToNQubits(m_n_qubits);
  }

  // Add the rows from v to this, if
  // the row already exists, we should just add the coeffs
  for (auto [i, row] : enumerate(tmpv.data)) {
    auto it = std::find(data.begin(), data.end(), row);
    if (it != data.end()) {
      auto idx = std::distance(data.begin(), it);
      coefficients[idx] += tmpv.coefficients[i];
    } else {
      data.push_back(row);
      coefficients.push_back(tmpv.coefficients[i]);
    }
  }

  // mark any rows for deletion if coeff = (0,0)
  std::vector<int> marked;
  for (std::size_t i = 0; i < data.size(); i++)
    if (std::abs(coefficients[i]) < 1e-12)
      marked.push_back(i);

  std::sort(marked.begin(), marked.end(), std::greater<int>());
  for (auto m : marked) {
    data.erase(data.begin() + m);
    coefficients.erase(coefficients.begin() + m);
  }

  return *this;
}

std::vector<std::complex<double>> spin_op::get_coefficients() const {
  return coefficients;
}

spin_op spin_op::operator[](const std::size_t term_idx) const {
  std::vector<bool> term_data = data[term_idx];
  auto term_coeff = coefficients[term_idx];
  BinarySymplecticForm f{term_data};
  std::vector<std::complex<double>> c{term_coeff};
  return spin_op(f, c);
}

spin_op &spin_op::operator-=(const spin_op &v) noexcept {
  return operator+=(-1.0 * v);
}

spin_op &spin_op::operator*=(const spin_op &v) noexcept {
  spin_op copy = v;
  if (v.m_n_qubits > m_n_qubits) {
    // If we are adding a op that has more qubits than we do
    // then we need to resize, making sure to ensure the
    // correct 1/0 positions.
    expandToNQubits(v.m_n_qubits);
  } else if (v.m_n_qubits < m_n_qubits) {
    copy.expandToNQubits(m_n_qubits);
  }

  int counter = 0;
  for (auto &row : data) {
    int inner_counter = 0;
    for (auto &other_row : copy.data) {
      // This is term * otherTerm
      std::vector<bool> tmp(2 * m_n_qubits), tmp2(2 * m_n_qubits);
      for (std::size_t i = 0; i < 2 * m_n_qubits; i++)
        tmp[i] = row[i] ^ other_row[i];

      for (std::size_t i = 0; i < m_n_qubits; i++)
        tmp2[i] = row[i] && other_row[m_n_qubits + i];

      int orig_phase = 0, other_phase = 0;
      for (std::size_t i = 0; i < m_n_qubits; i++) {
        if (row[i] && row[i + m_n_qubits])
          orig_phase++;

        if (other_row[i] && other_row[i + m_n_qubits])
          other_phase++;
      }

      auto _phase = orig_phase + other_phase;
      int sum = 0;
      for (auto a : tmp2)
        if (a)
          sum++;

      _phase += 2 * sum;
      // Based on the phase, figure out an extra coeff to apply
      for (std::size_t i = 0; i < m_n_qubits; i++)
        if (tmp[i] && tmp[i + m_n_qubits])
          _phase -= 1;

      _phase %= 4;
      std::complex<double> imaginary(0, 1);
      std::map<int, std::complex<double>> phase_coeff_map{
          {0, 1.0}, {1, -1. * imaginary}, {2, -1.0}, {3, imaginary}};
      auto phase_coeff = phase_coeff_map[_phase];
      coefficients[counter] *= phase_coeff * copy.coefficients[inner_counter];
      inner_counter++;
      row = tmp;
    }
    counter++;
  }
  return *this;
}

bool spin_op::is_identity() const {
  for (auto &row : data)
    for (auto e : row)
      if (e)
        return false;

  return true;
}

bool spin_op::operator==(const spin_op &v) const noexcept {
  // Could be that the term is identity with all zeros
  bool isId1 = true, isId2 = true;
  for (auto &row : data)
    for (auto e : row)
      if (e) {
        isId1 = false;
        break;
      }

  for (auto &row : v.data)
    for (auto e : row)
      if (e) {
        isId2 = false;
        break;
      }

  if (isId1 && isId2)
    return true;

  return data == v.data;
}

spin_op &spin_op::operator*=(const double v) noexcept {
  for (auto &c : coefficients)
    c *= v;

  return *this;
}
spin_op &spin_op::operator*=(const std::complex<double> v) noexcept {
  for (auto &c : coefficients)
    c *= v;

  return *this;
}

std::size_t spin_op::n_qubits() const { return m_n_qubits; }
std::size_t spin_op::n_terms() const { return data.size(); }
std::complex<double>
spin_op::get_term_coefficient(const std::size_t idx) const {
  return coefficients[idx];
}

spin_op spin_op::slice(const std::size_t startIdx, const std::size_t count) {
  auto nTerms = n_terms();
  if (nTerms <= count)
    throw std::runtime_error("Cannot request slice with " +
                             std::to_string(count) + " terms on spin_op with " +
                             std::to_string(nTerms) + " terms.");

  std::vector<std::complex<double>> newCoeffs;
  BinarySymplecticForm newData;
  for (std::size_t i = startIdx; i < startIdx + count; ++i) {
    if (i == data.size())
      break;
    newData.push_back(data[i]);
    newCoeffs.push_back(coefficients[i]);
  }
  return spin_op(newData, newCoeffs);
}

std::string spin_op::to_string(bool printCoeffs) const {
  if (data.empty())
    return "";

  auto first = data[0];
  std::stringstream ss;
  if (printCoeffs)
    ss << coefficients[0] << " ";
  for (std::size_t i = 0; i < m_n_qubits; i++) {
    if (first[i] && first[i + m_n_qubits])
      ss << "Y" << i;
    else if (first[i])
      ss << "X" << i;
    else if (first[i + m_n_qubits])
      ss << "Z" << i;
    else
      ss << "I" << i;
  }

  for (std::size_t j = 1; j < data.size(); j++) {
    ss << " + ";
    first = data[j];
    if (printCoeffs)
      ss << coefficients[j] << " ";
    for (std::size_t i = 0; i < m_n_qubits; i++) {
      if (first[i] && first[i + m_n_qubits])
        ss << "Y" << i;
      else if (first[i])
        ss << "X" << i;
      else if (first[i + m_n_qubits])
        ss << "Z" << i;
      else
        ss << "I" << i;
    }
  }

  return ss.str();
}

void spin_op::dump() const {
  auto str = to_string();
  printf("%s\n", str.c_str());
}

spin_op::spin_op(std::vector<double> &input_vec, std::size_t nQubits) {
  auto n_terms = (int)input_vec.back();
  if (nQubits != ((input_vec.size() - 2 * n_terms) / n_terms))
    throw std::runtime_error("Invalid data representation for construction "
                             "spin_op. Number of data elements is incorrect.");

  m_n_qubits = nQubits;
  for (std::size_t i = 0; i < input_vec.size() - 1; i += m_n_qubits + 2) {
    std::vector<bool> tmpv(2 * m_n_qubits);
    for (std::size_t j = 0; j < m_n_qubits; j++) {
      double intPart;
      if (std::modf(input_vec[j + i], &intPart) != 0.0)
        throw std::runtime_error(
            "Invalid pauli data element, must be integer value.");

      int val = (int)input_vec[j + i];
      if (val == 1) { // X
        tmpv[j] = 1;
      } else if (val == 2) { // Z
        tmpv[j + m_n_qubits] = 1;
      } else if (val == 3) { // Y
        tmpv[j + m_n_qubits] = 1;
        tmpv[j] = 1;
      }
    }
    data.push_back(tmpv);
    auto el_real = input_vec[i + m_n_qubits];
    auto el_imag = input_vec[i + m_n_qubits + 1];
    coefficients.emplace_back(el_real, el_imag);
  }
}

spin_op::BinarySymplecticForm spin_op::get_bsf() const { return data; }

spin_op &spin_op::operator=(const spin_op &other) {
  data = other.data;
  coefficients = other.coefficients;
  m_n_qubits = other.m_n_qubits;
  return *this;
}

spin_op operator+(double coeff, spin_op op) { return spin_op() * coeff + op; }
spin_op operator+(spin_op op, double coeff) { return op + spin_op() * coeff; }
spin_op operator-(double coeff, spin_op op) { return spin_op() * coeff - op; }
spin_op operator-(spin_op op, double coeff) { return op - spin_op() * coeff; }

namespace spin {
spin_op i(const std::size_t idx) { return spin_op(pauli::I, idx); }
spin_op x(const std::size_t idx) { return spin_op(pauli::X, idx); }
spin_op y(const std::size_t idx) { return spin_op(pauli::Y, idx); }
spin_op z(const std::size_t idx) { return spin_op(pauli::Z, idx); }
} // namespace spin

std::vector<double> spin_op::getDataRepresentation() {
  std::vector<double> dataVec;
  int counter = 0;
  for (auto &term : data) {
    auto nq = n_qubits();
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
    dataVec.push_back(coefficients[counter].real());
    dataVec.push_back(coefficients[counter].imag());
    counter++;
  }
  dataVec.push_back(n_terms());
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
