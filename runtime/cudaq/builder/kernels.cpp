/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "kernels.h"
#include "common/EigenDense.h"

namespace cudaq::details {

std::vector<std::string> grayCode(std::size_t rank) {
  std::function<void(std::vector<std::string> &, std::size_t)> grayCodeRecurse;
  grayCodeRecurse = [&grayCodeRecurse](std::vector<std::string> &g,
                                       std::size_t rank) {
    auto k = g.size();
    if (rank <= 0)
      return;

    for (int i = k - 1; i >= 0; --i) {
      auto c = "1" + g[i];
      g.push_back(c);
    }
    for (int i = k - 1; i >= 0; --i) {
      g[i] = "0" + g[i];
    }

    grayCodeRecurse(g, rank - 1);
  };

  std::vector<std::string> g{"0", "1"};
  grayCodeRecurse(g, rank - 1);

  return g;
}

std::vector<std::size_t> getControlIndices(std::size_t grayRank) {
  auto code = grayCode(grayRank);
  std::vector<std::size_t> ctrlIds;
  for (std::size_t i = 0; i < code.size(); i++) {
    auto a = std::stoi(code[i], nullptr, 2);
    auto b = std::stoi(code[(i + 1) % code.size()], nullptr, 2);
    auto c = a ^ b % code.size();
    ctrlIds.emplace_back(std::log2(c));
  }
  return ctrlIds;
}

int mEntry(std::size_t row, std::size_t col) {
  auto b_and_g = row & ((col >> 1) ^ col);
  std::size_t sum_of_ones = 0;
  while (b_and_g > 0) {
    if (b_and_g & 0b1)
      sum_of_ones += 1;

    b_and_g = b_and_g >> 1;
  }
  return std::pow(-1, sum_of_ones);
}

std::vector<double> computeAngle(const std::vector<double> &alpha) {
  auto ln = alpha.size();
  std::size_t k = std::log2(ln);

  Eigen::MatrixXi mTrans(ln, ln);
  mTrans.setZero();
  for (Eigen::Index i = 0; i < mTrans.rows(); i++)
    for (Eigen::Index j = 0; j < mTrans.cols(); j++)
      mTrans(i, j) = mEntry(i, j);

  Eigen::VectorXd alphaVec = Eigen::Map<Eigen::VectorXd>(
      const_cast<double *>(alpha.data()), alpha.size());
  Eigen::VectorXd thetas = (1. / (1UL << k)) * mTrans.cast<double>() * alphaVec;

  std::vector<double> ret(thetas.size());
  Eigen::VectorXd::Map(&ret[0], ret.size()) = thetas;
  return ret;
}

std::vector<double> getAlphaZ(const std::span<double> data,
                              std::size_t numQubits, std::size_t k) {

  std::vector<std::vector<std::size_t>> in1, in2;
  auto twoNmK = (1ULL << (numQubits - k)), twoKmOne = (1ULL << (k - 1));
  for (std::size_t j = 1; j < twoNmK + 1; j++) {
    std::vector<std::size_t> local;
    for (std::size_t l = 1; l < twoKmOne + 1; l++)
      local.push_back((2 * j - 1) * twoKmOne + l - 1);

    in1.push_back(local);
  }

  for (std::size_t j = 1; j < twoNmK + 1; j++) {
    std::vector<std::size_t> local;
    for (std::size_t l = 1; l < twoKmOne + 1; l++)
      local.push_back((2 * j - 2) * twoKmOne + l - 1);

    in2.push_back(local);
  }

  std::vector<std::vector<double>> term1, term2;
  for (std::size_t i = 0; auto &el : in1) {

    std::vector<double> local1, local2;
    for (auto &eel : el)
      local1.push_back(data[eel]);
    term1.push_back(local1);

    for (auto &eel : in2[i])
      local2.push_back(data[eel]);
    term2.push_back(local2);

    i++;
  }

  Eigen::MatrixXd term1Mat(term1.size(), term1[0].size()),
      term2Mat(term2.size(), term2[0].size());
  for (Eigen::Index i = 0; i < term1Mat.rows(); i++)
    for (Eigen::Index j = 0; j < term1Mat.cols(); j++) {
      term1Mat(i, j) = term1[i][j];
      term2Mat(i, j) = term2[i][j];
    }

  Eigen::MatrixXd diff = (1. / (1ULL << (k - 1))) * (term1Mat - term2Mat);
  std::vector<double> res(diff.rows());

  for (Eigen::Index i = 0; i < diff.rows(); i++) {
    double sum = 0.0;
    for (Eigen::Index j = 0; j < diff.cols(); j++)
      sum += diff(i, j);

    res[i] = sum;
  }

  return res;
}

std::vector<double> getAlphaY(const std::span<double> data,
                              std::size_t numQubits, std::size_t k) {
  std::vector<std::vector<std::size_t>> inNum, inDenom;
  auto twoNmK = (1ULL << (numQubits - k)), twoK = (1ULL << k),
       twoKmOne = (1ULL << (k - 1));
  for (auto j : cudaq::range(twoNmK)) {
    std::vector<std::size_t> local;
    for (auto l : cudaq::range(twoKmOne))
      local.push_back((2 * (j + 1) - 1) * twoKmOne + l);

    inNum.push_back(local);
  }

  std::vector<double> numeratorSums;
  for (auto &el : inNum) {
    double sum = 0.0;
    for (auto &i : el)
      sum += std::pow(std::fabs(data[i]), 2);

    numeratorSums.push_back(sum);
  }

  for (auto j : cudaq::range(twoNmK)) {
    std::vector<std::size_t> local;
    for (auto l : cudaq::range(twoK))
      local.push_back(j * twoK + l);

    inDenom.push_back(local);
  }

  std::vector<double> denomSums;
  for (auto &el : inDenom) {
    double sum = 0.0;
    for (auto &i : el)
      sum += std::pow(std::fabs(data[i]), 2);

    denomSums.push_back(sum);
  }

  std::vector<double> res(denomSums.size());
  for (std::size_t i = 0; i < denomSums.size(); i++) {
    if (std::fabs(denomSums[i]) > 1e-12)
      res[i] = 2 * std::asin(std::sqrt(numeratorSums[i] / denomSums[i]));
  }

  return res;
}
} // namespace cudaq::details