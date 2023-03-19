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
 * \file entropies.hpp
 * \brief Entropy functions
 */

#ifndef ENTROPY_HPP_
#define ENTROPY_HPP_

namespace qpp {
/**
 * \brief von-Neumann entropy of the density matrix \a A
 *
 * \param A Eigen expression
 * \return von-Neumann entropy, with the logarithm in base 2
 */
template <typename Derived>
double entropy(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::entropy()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::entropy()", "A");
    // END EXCEPTION CHECKS

    dmat ev = svals(rA); // get the singular values
    double result = 0;
    for (idx i = 0; i < static_cast<idx>(ev.rows()); ++i)
        if (ev(i) != 0) // not identically zero
            result -= ev(i) * std::log2(ev(i));

    return result;
}

/**
 * \brief Shannon entropy of the probability distribution \a prob
 *
 * \param prob Real probability vector
 * \return Shannon entropy, with the logarithm in base 2
 */
inline double entropy(const std::vector<double>& prob) {
    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(prob))
        throw exception::ZeroSize("qpp::entropy()", "prob");
    // END EXCEPTION CHECKS

    double result = 0;
    for (double p : prob)
        if (std::abs(p) != 0) // not identically zero
            result -= std::abs(p) * std::log2(std::abs(p));

    return result;
}

/**
 * \brief Renyi-\f$\alpha\f$ entropy of the density matrix \a A,
 * for \f$\alpha\geq 0\f$
 *
 * \note When \f$\alpha\to 1\f$ the Renyi entropy converges to the von-Neumann
 * entropy, with the logarithm in base 2
 *
 * \param A Eigen expression
 * \param alpha Non-negative real number, use qpp::infty for
 * \f$\alpha = \infty\f$
 * \return Renyi-\f$\alpha\f$ entropy, with the logarithm in base 2
 */
template <typename Derived>
double renyi(const Eigen::MatrixBase<Derived>& A, double alpha) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::renyi()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::renyi()", "A");

    if (alpha < 0)
        throw exception::OutOfRange("qpp::renyi()", "alpha");
    // END EXCEPTION CHECKS

    if (alpha == 0) // H max
        return std::log2(rA.rows());

    if (alpha == 1) // Shannon/von-Neumann
        return entropy(rA);

    if (alpha == infty) // H min
        return -std::log2(svals(rA)[0]);

    dmat sv = svals(rA); // get the singular values
    double result = 0;
    for (idx i = 0; i < static_cast<idx>(sv.rows()); ++i)
        if (sv(i) != 0) // not identically zero
            result += std::pow(sv(i), alpha);

    return std::log2(result) / (1 - alpha);
}

/**
 * \brief Renyi-\f$\alpha\f$ entropy of the probability distribution \a prob,
 * for \f$\alpha\geq 0\f$
 *
 * \note When \f$\alpha\to 1\f$ the Renyi entropy converges to the Shannon
 * entropy, with the logarithm in base 2
 *
 * \param prob Real probability vector
 * \param alpha Non-negative real number, use qpp::infty for
 * \f$\alpha = \infty\f$
 * \return Renyi-\f$\alpha\f$ entropy, with the logarithm in base 2
 */
inline double renyi(const std::vector<double>& prob, double alpha) {
    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(prob))
        throw exception::ZeroSize("qpp::renyi()", "prob");

    if (alpha < 0)
        throw exception::OutOfRange("qpp::renyi()", "alpha");

    if (alpha == 0) // H max
        return std::log2(prob.size());

    if (alpha == 1) // Shannon/von-Neumann
        return entropy(prob);
    // END EXCEPTION CHECKS

    if (alpha == infty) // H min
    {
        double max = 0;
        for (double p : prob)
            if (std::abs(p) > max)
                max = std::abs(p);

        return -std::log2(max);
    }

    double result = 0;
    for (double p : prob)
        if (std::abs(p) != 0) // not identically zero
            result += std::pow(std::abs(p), alpha);

    return std::log2(result) / (1 - alpha);
}

/**
 * \brief Tsallis-\f$q\f$ entropy of the density matrix \a A, for \f$q\geq 0\f$
 *
 * \note When \f$q\to 1\f$ the Tsallis entropy converges to the von-Neumann
 * entropy, with the logarithm in base \f$e\f$
 *
 * \param A Eigen expression
 * \param q Non-negative real number
 * \return Tsallis-\f$q\f$ entropy
 */
template <typename Derived>
double tsallis(const Eigen::MatrixBase<Derived>& A, double q) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::tsallis()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::tsallis()", "A");

    if (q < 0)
        throw exception::OutOfRange("qpp::tsallis()", "q");
    // END EXCEPTION CHECKS

    if (q == 1) // Shannon/von-Neumann with base e logarithm
        return entropy(rA) * std::log(2);

    dmat ev = svals(rA); // get the singular values
    double result = 0;
    for (idx i = 0; i < static_cast<idx>(ev.rows()); ++i)
        if (ev(i) != 0) // not identically zero
            result += std::pow(ev(i), q);

    return (result - 1) / (1 - q);
}

/**
 * \brief Tsallis-\f$q\f$ entropy of the probability distribution \a prob, for
 * \f$q\geq 0\f$
 *
 * \note When \f$q\to 1\f$ the Tsallis entropy converges to the Shannon
 * entropy, with the logarithm in base \f$e\f$
 *
 * \param prob Real probability vector
 * \param q Non-negative real number
 * \return Tsallis-\f$q\f$ entropy
 */
inline double tsallis(const std::vector<double>& prob, double q) {
    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(prob))
        throw exception::ZeroSize("qpp::tsallis()", "prob");

    if (q < 0)
        throw exception::OutOfRange("qpp::tsallis()", "q");
    // END EXCEPTION CHECKS

    if (q == 1) // Shannon/von-Neumann with base e logarithm
        return entropy(prob) * std::log(2);

    double result = 0;
    for (double p : prob)
        if (std::abs(p) != 0) // not identically zero
            result += std::pow(std::abs(p), q);

    return (result - 1) / (1 - q);
}

/**
 * \brief Quantum mutual information between 2 subsystems of a composite system
 *
 * \param A Eigen expression
 * \param subsysA Indexes of the first subsystem
 * \param subsysB Indexes of the second subsystem
 * \param dims Dimensions of the multi-partite system
 * \return Mutual information between the 2 subsystems
 */
template <typename Derived>
double qmutualinfo(const Eigen::MatrixBase<Derived>& A,
                   const std::vector<idx>& subsysA,
                   const std::vector<idx>& subsysB,
                   const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::qmutualinfo()", "A");

    // check that dims is a valid dimension vector
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::qmutualinfo()", "dims");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::qmutualinfo()", "A");

    // check that dims match the dimension of A
    if (!internal::check_dims_match_mat(dims, rA))
        throw exception::DimsMismatchMatrix("qpp::qmutualinfo()", "A/dims");

    // check that subsys are valid
    if (!internal::check_subsys_match_dims(subsysA, dims))
        throw exception::SubsysMismatchDims("qpp::qmutualinfo()", "subsysA");
    if (!internal::check_subsys_match_dims(subsysB, dims))
        throw exception::SubsysMismatchDims("qpp::qmutualinfo()", "subsysB");
    // END EXCEPTION CHECKS

    // The full system indexes {0,1,...,n-1}
    std::vector<idx> full_system(dims.size());
    std::iota(std::begin(full_system), std::end(full_system), 0);

    // Sorted input subsystems
    std::vector<idx> subsysAsorted{subsysA};
    std::vector<idx> subsysBsorted{subsysB};

    // sort the input subsystems (as needed by std::set_union)
    std::sort(std::begin(subsysAsorted), std::end(subsysAsorted));
    std::sort(std::begin(subsysBsorted), std::end(subsysBsorted));

    // construct the union of A and B
    std::vector<idx> subsysAB;
    std::set_union(std::begin(subsysAsorted), std::end(subsysAsorted),
                   std::begin(subsysBsorted), std::end(subsysBsorted),
                   std::back_inserter(subsysAB));

    // construct the complements
    std::vector<idx> subsysA_bar = complement(subsysA, dims.size());
    std::vector<idx> subsysB_bar = complement(subsysB, dims.size());
    std::vector<idx> subsysAB_bar = complement(subsysAB, dims.size());

    cmat rhoA = ptrace(rA, subsysA_bar, dims);
    cmat rhoB = ptrace(rA, subsysB_bar, dims);
    cmat rhoAB = ptrace(rA, subsysAB_bar, dims);

    return entropy(rhoA) + entropy(rhoB) - entropy(rhoAB);
}

/**
 * \brief Quantum mutual information between 2 subsystems of a composite system
 *
 * \param A Eigen expression
 * \param subsysA Indexes of the first subsystem
 * \param subsysB Indexes of the second subsystem
 * \param d Subsystem dimensions
 * \return Mutual information between the 2 subsystems
 */
template <typename Derived>
double qmutualinfo(const Eigen::MatrixBase<Derived>& A,
                   const std::vector<idx>& subsysA,
                   const std::vector<idx>& subsysB, idx d = 2) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::qmutualinfo()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::qmutualinfo()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return qmutualinfo(rA, subsysA, subsysB, dims);
}

} /* namespace qpp */

#endif /* ENTROPY_HPP_ */
