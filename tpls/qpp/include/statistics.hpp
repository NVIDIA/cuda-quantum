/*
 * This file is part of Quantum++.
 *
 * Copyright (c) 2013 - 2022 softwareQ Inc. All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * \file statistics.hpp
 * \brief Statistics functions
 */

#ifndef STATISTICS_HPP_
#define STATISTICS_HPP_

namespace qpp {
/**
 * \brief Uniform probability distribution vector
 *
 * \param N Size of the alphabet
 * \return Real vector consisting of a uniform distribution of size \a N
 */
inline std::vector<double> uniform(idx N) {
    // EXCEPTION CHECKS

    if (N == 0)
        throw exception::ZeroSize("qpp::uniform()", "N");
    // END EXCEPTION CHECKS

    return std::vector<double>(N, 1. / static_cast<double>(N));
}

/**
 * \brief Marginal distribution
 *
 * \param probXY Real matrix representing the joint probability distribution of
 * \a X and \a Y in lexicographical order (\a X labels the rows, \a Y labels the
 * columns)
 * \return Real vector consisting of the marginal distribution of \a X
 */
inline std::vector<double> marginalX(const dmat& probXY) {
    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(probXY))
        throw exception::ZeroSize("qpp::marginalX()", "probXY");
    // END EXCEPTION CHECKS

    std::vector<double> result(probXY.rows(), 0);
    for (idx i = 0; i < static_cast<idx>(probXY.rows()); ++i) {
        for (idx j = 0; j < static_cast<idx>(probXY.cols()); ++j) {
            result[i] += probXY(i, j);
        }
    }

    return result;
}

/**
 * \brief Marginal distribution
 *
 * \param probXY Real matrix representing the joint probability distribution of
 * \a X and \a Y in lexicographical order (\a X labels the rows, \a Y labels the
 * columns)
 * \return Real vector consisting of the marginal distribution of \a Y
 */
inline std::vector<double> marginalY(const dmat& probXY) {
    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(probXY))
        throw exception::ZeroSize("qpp::marginalY()", "probXY");
    // END EXCEPTION CHECKS

    return marginalX(probXY.transpose());
}

/**
 * \brief Average
 *
 * \param prob Real probability vector representing the probability distribution
 * of \a X
 *
 * \param X Real random variable values represented by an STL-like container
 * \return Average of \a X
 */
template <typename Container>
double
avg(const std::vector<double>& prob, const Container& X,
    typename std::enable_if<is_iterable<Container>::value>::type* = nullptr) {
    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(prob))
        throw exception::ZeroSize("qpp::avg()", "prob");
    if (!internal::check_matching_sizes(prob, X))
        throw exception::SizeMismatch("qpp::avg()", "prob/X");
    // END EXCEPTION CHECKS

    double result = 0;
    for (idx i = 0; i < prob.size(); ++i)
        result += prob[i] * X[i];

    return result;
}

/**
 * \brief Covariance
 *
 * \param probXY Real matrix representing the joint probability distribution of
 * \a X and \a Y in lexicographical order (\a X labels the rows, \a Y labels the
 * columns)
 * \param X Real random variable values represented by an STL-like container
 * \param Y Real random variable values represented by an STL-like container
 * \return Covariance of \a X and \a Y
 */
template <typename Container>
double
cov(const dmat& probXY, const Container& X, const Container& Y,
    typename std::enable_if<is_iterable<Container>::value>::type* = nullptr) {
    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(X))
        throw exception::ZeroSize("qpp::cov()", "X");
    if (!internal::check_nonzero_size(Y))
        throw exception::ZeroSize("qpp::cov()", "Y");
    if (static_cast<idx>(probXY.rows()) != X.size())
        throw exception::SizeMismatch("qpp::cov()", "probXY/X");
    if (static_cast<idx>(probXY.cols()) != Y.size())
        throw exception::SizeMismatch("qpp::cov()", "probXY/Y");
    // END EXCEPTION CHECKS

    std::vector<double> probX = marginalX(probXY); // marginals
    std::vector<double> probY = marginalY(probXY); // marginals

    double result = 0;
    for (idx i = 0; i < X.size(); ++i) {
        for (idx j = 0; j < Y.size(); ++j) {
            result += probXY(i, j) * X[i] * Y[j];
        }
    }

    return result - avg(probX, X) * avg(probY, Y);
}

/**
 * \brief Variance
 *
 * \param prob Real probability vector representing the probability distribution
 * of \a X
 * \param X Real random variable values represented by an STL-like container
 * \return Variance of \a X
 */
template <typename Container>
double
var(const std::vector<double>& prob, const Container& X,
    typename std::enable_if<is_iterable<Container>::value>::type* = nullptr) {
    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(prob))
        throw exception::ZeroSize("qpp::var()", "prob");
    if (!internal::check_matching_sizes(prob, X))
        throw exception::SizeMismatch("qpp::var()", "prob/X");
    // END EXCEPTION CHECKS

    Eigen::VectorXd diag(prob.size());
    for (idx i = 0; i < prob.size(); ++i)
        diag(i) = prob[i];
    dmat probXX = diag.asDiagonal();

    return cov(probXX, X, X);
}

/**
 * \brief Standard deviation
 *
 * \param prob Real probability vector representing the probability distribution
 * of \a X
 * \param X Real random variable values represented by an STL-like container
 * \return Standard deviation of \a X
 */
template <typename Container>
double
sigma(const std::vector<double>& prob, const Container& X,
      typename std::enable_if<is_iterable<Container>::value>::type* = nullptr) {
    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(prob))
        throw exception::ZeroSize("qpp::sigma()", "prob");
    if (!internal::check_matching_sizes(prob, X))
        throw exception::SizeMismatch("qpp::sigma()", "prob/X");
    // END EXCEPTION CHECKS

    return std::sqrt(var(prob, X));
}

/**
 * \brief Correlation
 *
 * \param probXY Real matrix representing the joint probability distribution of
 * \a X and \a Y in lexicographical order (\a X labels the rows, \a Y labels the
 * columns)
 * \param X Real random variable values represented by an STL-like container
 * \param Y Real random variable values represented by an STL-like container
 * \return Correlation of \a X and \a Y
 */
template <typename Container>
double
cor(const dmat& probXY, const Container& X, const Container& Y,
    typename std::enable_if<is_iterable<Container>::value>::type* = nullptr) {
    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(X))
        throw exception::ZeroSize("qpp::cor()", "X");
    if (!internal::check_nonzero_size(Y))
        throw exception::ZeroSize("qpp::cor()", "Y");
    if (static_cast<idx>(probXY.rows()) != X.size())
        throw exception::SizeMismatch("qpp::cor()", "probXY/X");
    if (static_cast<idx>(probXY.cols()) != Y.size())
        throw exception::SizeMismatch("qpp::cor()", "probXY/Y");
    // END EXCEPTION CHECKS

    return cov(probXY, X, Y) /
           (sigma(marginalX(probXY), X) * sigma(marginalY(probXY), Y));
}

} /* namespace qpp */

#endif /* STATISTICS_HPP_ */
