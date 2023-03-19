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
 * \file instruments.hpp
 * \brief Measurement functions
 */

#ifndef INSTRUMENTS_HPP_
#define INSTRUMENTS_HPP_

namespace qpp {
/**
 * \brief Generalized inner product
 *
 * \param phi Column vector Eigen expression
 * \param psi Column vector Eigen expression
 * \param subsys Subsystem indexes over which \a phi is defined
 * \param dims Dimensions of the multi-partite system
 * \return Inner product \f$\langle \phi_{subsys}|\psi\rangle\f$, as a scalar or
 * column vector over the remaining Hilbert space
 */
template <typename Derived>
[[qpp::critical, qpp::parallel]] dyn_col_vect<typename Derived::Scalar>
ip(const Eigen::MatrixBase<Derived>& phi, const Eigen::MatrixBase<Derived>& psi,
   const std::vector<idx>& subsys, const std::vector<idx>& dims) {
    const dyn_col_vect<typename Derived::Scalar>& rphi = phi.derived();
    const dyn_col_vect<typename Derived::Scalar>& rpsi = psi.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rphi))
        throw exception::ZeroSize("qpp::ip()", "phi");

    // check zero-size
    if (!internal::check_nonzero_size(rpsi))
        throw exception::ZeroSize("qpp::ip()", "psi");

    // check column vector
    if (!internal::check_cvector(rphi))
        throw exception::MatrixNotCvector("qpp::ip()", "phi");

    // check column vector
    if (!internal::check_cvector(rpsi))
        throw exception::MatrixNotCvector("qpp::ip()", "psi");

    // check that dims is a valid dimension vector
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::ip()");

    // check that subsys are valid w.r.t. dims
    if (!internal::check_subsys_match_dims(subsys, dims))
        throw exception::SubsysMismatchDims("qpp::ip()", "dims/subsys");

    // check that dims match psi column vector
    if (!internal::check_dims_match_cvect(dims, rpsi))
        throw exception::DimsMismatchCvector("qpp::ip()", "dims/psi");

    // check that subsys match phi column vector
    std::vector<idx> subsys_dims(subsys.size());
    for (idx i = 0; i < subsys.size(); ++i)
        subsys_dims[i] = dims[subsys[i]];
    if (!internal::check_dims_match_cvect(subsys_dims, rphi))
        throw exception::DimsMismatchCvector("qpp::ip()", "dims/phi");
    // END EXCEPTION CHECKS

    idx Dsubsys = prod(std::begin(subsys_dims), std::end(subsys_dims));

    idx D = static_cast<idx>(rpsi.rows());
    idx Dsubsys_bar = D / Dsubsys;

    idx n = dims.size();
    idx n_subsys = subsys.size();
    idx n_subsys_bar = n - n_subsys;

    idx Cdims[internal::maxn];
    idx Csubsys[internal::maxn];
    idx Cdimssubsys[internal::maxn];
    idx Csubsys_bar[internal::maxn];
    idx Cdimssubsys_bar[internal::maxn];

    std::vector<idx> subsys_bar = complement(subsys, n);
    std::copy(std::begin(subsys_bar), std::end(subsys_bar),
              std::begin(Csubsys_bar));

    for (idx i = 0; i < n; ++i) {
        Cdims[i] = dims[i];
    }
    for (idx i = 0; i < n_subsys; ++i) {
        Csubsys[i] = subsys[i];
        Cdimssubsys[i] = dims[subsys[i]];
    }
    for (idx i = 0; i < n_subsys_bar; ++i) {
        Cdimssubsys_bar[i] = dims[subsys_bar[i]];
    }

    auto worker = [&](idx b) noexcept -> typename Derived::Scalar {
        idx Cmidxrow[internal::maxn];
        idx Cmidxrowsubsys[internal::maxn];
        idx Cmidxcolsubsys_bar[internal::maxn];

        /* get the col multi-indexes of the complement */
        internal::n2multiidx(b, n_subsys_bar, Cdimssubsys_bar,
                             Cmidxcolsubsys_bar);
        /* write it in the global row multi-index */
        for (idx k = 0; k < n_subsys_bar; ++k) {
            Cmidxrow[Csubsys_bar[k]] = Cmidxcolsubsys_bar[k];
        }

        typename Derived::Scalar result = 0;
        for (idx a = 0; a < Dsubsys; ++a) {
            /* get the row multi-indexes of the subsys */
            internal::n2multiidx(a, n_subsys, Cdimssubsys, Cmidxrowsubsys);
            /* write it in the global row multi-index */
            for (idx k = 0; k < n_subsys; ++k) {
                Cmidxrow[Csubsys[k]] = Cmidxrowsubsys[k];
            }
            // compute the row index
            idx i = internal::multiidx2n(Cmidxrow, n, Cdims);

            result += std::conj(rphi(a)) * rpsi(i);
        }

        return result;
    }; /* end worker */

    dyn_col_vect<typename Derived::Scalar> result(Dsubsys_bar);
#ifdef HAS_OPENMP
// NOLINTNEXTLINE
#pragma omp parallel for
#endif // HAS_OPENMP
    for (idx m = 0; m < Dsubsys_bar; ++m)
        result(m) = worker(m);

    return result;
}

/**
 * \brief Generalized inner product
 *
 * \param phi Column vector Eigen expression
 * \param psi Column vector Eigen expression
 * \param subsys Subsystem indexes over which \a phi is defined
 * \param d Subsystem dimensions
 * \return Inner product \f$\langle \phi_{subsys}|\psi\rangle\f$, as a scalar or
 * column vector over the remaining Hilbert space
 */
template <typename Derived>
dyn_col_vect<typename Derived::Scalar>
ip(const Eigen::MatrixBase<Derived>& phi, const Eigen::MatrixBase<Derived>& psi,
   const std::vector<idx>& subsys, idx d = 2) {
    const dyn_col_vect<typename Derived::Scalar>& rphi = phi.derived();
    const dyn_col_vect<typename Derived::Scalar>& rpsi = psi.derived();

    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(rpsi))
        throw exception::ZeroSize("qpp::ip()", "psi");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::ip()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rpsi.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return ip(phi, psi, subsys, dims);
}

// full measurements
/**
 * \brief Measures the state vector or density operator \a A using the set of
 * Kraus operators \a Ks
 *
 * \note The Kraus operators can have their range different from their domain
 * (i.e., they can be rectangular matrices).
 *
 * \param A Eigen expression
 * \param Ks Set of Kraus operators
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
[[qpp::critical]] std::tuple<idx, std::vector<double>,
                             std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A, const std::vector<cmat>& Ks) {
    const expr_t<Derived>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure()", "A");

    // check the Kraus operators
    if (Ks.empty())
        throw exception::ZeroSize("qpp::measure()", "Ks");
    if (Ks[0].cols() != rA.rows())
        throw exception::SizeMismatch("qpp::measure()", "Ks[0]/A");
    for (auto&& elem : Ks)
        if (elem.rows() != Ks[0].rows() || elem.cols() != Ks[0].cols())
            throw exception::DimsNotEqual("qpp::measure()", "K");
    // END EXCEPTION CHECKS

    // probabilities
    std::vector<double> probs(Ks.size());
    // resulting states
    std::vector<expr_t<Derived>> outstates(Ks.size());

    idx Dout = Ks[0].rows();
    //************ density matrix ************//
    if (internal::check_square_mat(rA)) // square matrix
    {
        for (idx i = 0; i < Ks.size(); ++i) {
            outstates[i] = cmat::Zero(Dout, Dout);
            cmat tmp = Ks[i] * rA * adjoint(Ks[i]); // un-normalized;
            probs[i] = std::abs(trace(tmp));        // probability
            if (probs[i] > 0) {
                outstates[i] = tmp / probs[i]; // normalized
            }
        }
    }
    //************ ket ************//
    else if (internal::check_cvector(rA)) // column vector
    {
        for (idx i = 0; i < Ks.size(); ++i) {
            outstates[i] = ket::Zero(Dout);
            ket tmp = Ks[i] * rA; // un-normalized;
            // probability
            probs[i] = std::pow(norm(tmp), 2);
            if (probs[i] > 0) {
                outstates[i] = tmp / std::sqrt(probs[i]); // normalized
            }
        }
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::measure()", "A");

    // sample from the probability distribution
    assert(probs != decltype(probs)(probs.size(), 0)); // not all zeros
    std::discrete_distribution<idx> dd(std::begin(probs), std::end(probs));
    auto& gen = RandomDevices::get_instance().get_prng();
    idx result = dd(gen);

    return std::make_tuple(result, probs, outstates);
}

// std::initializer_list overload, avoids ambiguity for 2-element lists, see
// https://stackoverflow.com/questions/26750039/ambiguity-when-using-initializer-list-as-parameter
/**
 * \brief Measures the state vector or density matrix \a A using the set of
 * Kraus operators \a Ks
 *
 * \param A Eigen expression
 * \param Ks Set of Kraus operators
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
std::tuple<idx, std::vector<double>, std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A,
        const std::initializer_list<cmat>& Ks) {
    return measure(A, std::vector<cmat>(Ks));
}

/**
 * \brief Measures the state vector or density matrix \a A in the orthonormal
 * basis specified by the unitary matrix \a U
 * \see qpp::sample()
 *
 * \note This measurement is equivalent to measuring \a A in the
 * \f$U^\dagger\f$ basis
 *
 * \param A Eigen expression
 * \param U Unitary matrix whose columns represent the measurement basis vectors
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
std::tuple<idx, std::vector<double>, std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A, const cmat& U) {
    const expr_t<Derived>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure()", "A");

    // check the unitary basis matrix U
    if (!internal::check_nonzero_size(U))
        throw exception::ZeroSize("qpp::measure()", "U");
    if (!internal::check_square_mat(U))
        throw exception::MatrixNotSquare("qpp::measure()", "U");
    if (U.rows() != rA.rows())
        throw exception::SizeMismatch("qpp::measure()", "A/U");
    // END EXCEPTION CHECKS

    std::vector<cmat> Ks(U.rows());
    for (idx i = 0; i < static_cast<idx>(U.rows()); ++i)
        Ks[i] = U.col(i) * adjoint(U.col(i));

    return measure(rA, Ks);
}

// partial measurements
/**
 * \brief  Measures the part \a subsys of the multi-partite state vector or
 * density matrix \a A using the set of Kraus operators \a Ks
 *
 * \note The dimension of all \a Ks must match the dimension of \a target. If
 * \a destructive is set to true (by default), the measurement is destructive,
 * i.e., the measured subsystems are traced away.
 *
 * \param A Eigen expression
 * \param Ks Set of Kraus operators (must be square)
 * \param target Subsystem indexes that are measured
 * \param dims Dimensions of the multi-partite system
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
[[qpp::critical]] std::tuple<idx, std::vector<double>,
                             std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A, const std::vector<cmat>& Ks,
        const std::vector<idx>& target, const std::vector<idx>& dims,
        bool destructive = true) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure()", "A");

    // check that dimension is valid
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::measure()", "dims");

    // check that target is valid w.r.t. dims
    if (!internal::check_subsys_match_dims(target, dims))
        throw exception::SubsysMismatchDims("qpp::measure()", "dims/target");

    // check valid state and matching dimensions
    if (internal::check_cvector(rA)) {
        if (!internal::check_dims_match_cvect(dims, rA))
            throw exception::DimsMismatchCvector("qpp::measure()", "A/dims");
    } else if (internal::check_square_mat(rA)) {
        if (!internal::check_dims_match_mat(dims, rA))
            throw exception::DimsMismatchMatrix("qpp::measure()", "A/dims");
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::measure()", "A");

    std::vector<idx> subsys_dims(target.size());
    for (idx i = 0; i < target.size(); ++i)
        subsys_dims[i] = dims[target[i]];

    idx D = prod(std::begin(dims), std::end(dims));
    idx Dsubsys = prod(std::begin(subsys_dims), std::end(subsys_dims));
    idx Dsubsys_bar = D / Dsubsys;

    // check the Kraus operators
    if (Ks.empty())
        throw exception::ZeroSize("qpp::measure()", "Ks");
    if (!internal::check_square_mat(Ks[0]))
        throw exception::MatrixNotSquare("qpp::measure()", "Ks[0]");
    if (Dsubsys != static_cast<idx>(Ks[0].rows()))
        throw exception::MatrixMismatchSubsys("qpp::measure()", "Ks[0]");
    for (auto&& elem : Ks)
        if (elem.rows() != Ks[0].rows() || elem.cols() != Ks[0].cols())
            throw exception::DimsNotEqual("qpp::measure()", "K");
    // END EXCEPTION CHECKS

    // probabilities
    std::vector<double> probs(Ks.size());
    // resulting states
    std::vector<expr_t<Derived>> outstates;

    if (destructive)
        outstates.resize(Ks.size(), cmat::Zero(Dsubsys_bar, Dsubsys_bar));
    else
        outstates.resize(Ks.size(), cmat::Zero(D, D));

    //************ ket ************//
    if (internal::check_cvector(rA)) // column vector
    {
        for (idx i = 0; i < Ks.size(); ++i) {
            ket tmp = apply(rA, Ks[i], target, dims);
            probs[i] = std::pow(norm(tmp), 2);
            if (probs[i] > 0) {
                // normalized output state
                // corresponding to measurement result i
                tmp /= std::sqrt(probs[i]);
                if (destructive)
                    outstates[i] = ptrace(tmp, target, dims);
                else
                    outstates[i] = tmp;
            }
        }
    }
    //************ density matrix ************//
    else // square matrix
    {
        for (idx i = 0; i < Ks.size(); ++i) {
            cmat tmp = apply(rA, Ks[i], target, dims);
            if (destructive)
                tmp = ptrace(tmp, target, dims);
            probs[i] = std::abs(trace(tmp)); // probability
            if (probs[i] > 0) {
                // normalized output state
                // corresponding to measurement result i
                outstates[i] = tmp / probs[i];
            }
        }
    }
    // sample from the probability distribution
    assert(probs != decltype(probs)(probs.size(), 0)); // not all zeros
    std::discrete_distribution<idx> dd(std::begin(probs), std::end(probs));
    auto& gen = RandomDevices::get_instance().get_prng();
    idx result = dd(gen);

    return std::make_tuple(result, probs, outstates);
}

// std::initializer_list overload, avoids ambiguity for 2-element lists, see
// https://stackoverflow.com/questions/26750039/ambiguity-when-using-initializer-list-as-parameter
/**
 * \brief  Measures the part \a target of the multi-partite state vector or
 * density matrix \a A using the set of Kraus operators \a Ks
 *
 * \note The dimension of all \a Ks must match the dimension of \a target. If
 * \a destructive is set to true (by default), the measurement is destructive,
 * i.e., the measured subsystems are traced away.
 *
 * \param A Eigen expression
 * \param Ks Set of Kraus operators
 * \param target Subsystem indexes that are measured
 * \param dims Dimensions of the multi-partite system
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
std::tuple<idx, std::vector<double>, std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A,
        const std::initializer_list<cmat>& Ks, const std::vector<idx>& target,
        const std::vector<idx>& dims, bool destructive = true) {
    return measure(A, std::vector<cmat>(Ks), target, dims, destructive);
}

/**
 * \brief  Measures the part \a target of the multi-partite state vector or
 * density matrix \a A using the set of Kraus operators \a Ks
 *
 * \note The dimension of all \a Ks must match the dimension of \a target. If
 * \a destructive is set to true (by default), the measurement is destructive,
 * i.e., the measured subsystems are traced away.
 *
 * \param A Eigen expression
 * \param Ks Set of Kraus operators (must be square)
 * \param target Subsystem indexes that are measured
 * \param d Subsystem dimensions
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
std::tuple<idx, std::vector<double>, std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A, const std::vector<cmat>& Ks,
        const std::vector<idx>& target, idx d = 2, bool destructive = true) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::measure()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return measure(rA, Ks, target, dims, destructive);
}

// std::initializer_list overload, avoids ambiguity for 2-element lists, see
// https://stackoverflow.com/questions/26750039/ambiguity-when-using-initializer-list-as-parameter
/**
 * \brief  Measures the part \a target of the multi-partite state vector or
 * density matrix \a A using the set of Kraus operators \a Ks
 *
 * \note The dimension of all \a Ks must match the dimension of \a target. If
 * \a destructive is set to true (by default), the measurement is destructive,
 * i.e., the measured subsystems are traced away.
 *
 * \param A Eigen expression
 * \param Ks Set of Kraus operators
 * \param target Subsystem indexes that are measured
 * \param d Subsystem dimensions
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
std::tuple<idx, std::vector<double>, std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A,
        const std::initializer_list<cmat>& Ks, const std::vector<idx>& target,
        idx d = 2, bool destructive = true) {
    return measure(A, std::vector<cmat>(Ks), target, d, destructive);
}

/**
 * \brief Measures the part \a target of the multi-partite state vector or
 * density matrix \a A in the orthonormal basis or rank-1 projectors specified
 * by the columns of the matrix \a V
 * \see qpp::measure_seq(), qpp::sample()
 *
 * \note The dimension of \a V must match the dimension of \a target. If
 * \a destructive is set to true (by default), the measurement is destructive,
 * i.e., the measured subsystems are traced away.
 *
 * \note The number of column vectors of \a V can be smaller than the dimension
 * of \a target. If that is the case, then the measurement probabilities sum up
 * to a number smaller than 1 (i.e., having, in effect, a filtering operation).
 *
 * \param A Eigen expression
 * \param V Matrix whose columns represent the measurement basis vectors or the
 * ket parts of the rank-1 projectors
 * \param target Subsystem indexes that are measured
 * \param dims Dimensions of the multi-partite system
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
[[qpp::critical, qpp::parallel]] std::tuple<idx, std::vector<double>,
                                            std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A, const cmat& V,
        const std::vector<idx>& target, const std::vector<idx>& dims,
        bool destructive = true) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure()", "A");

    // check that dimension is valid
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::measure()", "dims");

    // check that target is valid w.r.t. dims
    if (!internal::check_subsys_match_dims(target, dims))
        throw exception::SubsysMismatchDims("qpp::measure()", "dims/target");

    // check valid state and matching dimensions
    if (internal::check_cvector(rA)) {
        if (!internal::check_dims_match_cvect(dims, rA))
            throw exception::DimsMismatchCvector("qpp::measure()", "A/dims");
    } else if (internal::check_square_mat(rA)) {
        if (!internal::check_dims_match_mat(dims, rA))
            throw exception::DimsMismatchMatrix("qpp::measure()", "A/dims");
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::measure()", "A");

    std::vector<idx> subsys_dims(target.size());
    for (idx i = 0; i < target.size(); ++i)
        subsys_dims[i] = dims[target[i]];

    idx Dsubsys = prod(std::begin(subsys_dims), std::end(subsys_dims));

    // check the matrix V
    if (!internal::check_nonzero_size(V))
        throw exception::ZeroSize("qpp::measure()", "V");
    if (Dsubsys != static_cast<idx>(V.rows()))
        throw exception::MatrixMismatchSubsys("qpp::measure()", "V");
    // END EXCEPTION CHECKS

    // number of basis elements or number of rank-1 projectors
    idx M = static_cast<idx>(V.cols());

    //************ ket ************//
    if (internal::check_cvector(rA)) {
        const ket& rpsi = A.derived();
        std::vector<double> probs(M);              // probabilities
        std::vector<expr_t<Derived>> outstates(M); // resulting states

#ifdef HAS_OPENMP
// NOLINTNEXTLINE
#pragma omp parallel for
#endif // HAS_OPENMP
        for (idx i = 0; i < M; ++i) {
            if (destructive)
                outstates[i] =
                    ip(static_cast<const ket&>(V.col(i)), rpsi, target, dims);
            else
                outstates[i] = apply(rpsi, prj(V.col(i)), target, dims);
        }

        for (idx i = 0; i < M; ++i) {
            double tmp = norm(outstates[i]);
            probs[i] = tmp * tmp;
            if (probs[i] > 0) {
                // normalized output state
                // corresponding to measurement result m
                outstates[i] /= tmp;
            }
        }

        // sample from the probability distribution
        assert(probs != decltype(probs)(probs.size(), 0)); // not all zeros
        std::discrete_distribution<idx> dd(std::begin(probs), std::end(probs));
        auto& gen = RandomDevices::get_instance().get_prng();
        idx result = dd(gen);

        return std::make_tuple(result, probs, outstates);
    }
    //************ density matrix ************//
    else {
        std::vector<cmat> Ks(M);
        for (idx i = 0; i < M; ++i)
            Ks[i] = V.col(i) * adjoint(V.col(i));

        return measure(rA, Ks, target, dims, destructive);
    }
}

/**
 * \brief Measures the part \a target of the multi-partite state vector or
 * density matrix \a A in the orthonormal basis or rank-1 projectors specified
 * by the columns of the matrix \a V
 * \see qpp::measure_seq(), qpp::sample()
 *
 * \note The dimension of \a V must match the dimension of \a target. If
 * \a destructive is set to true (by default), the measurement is destructive,
 * i.e., the measured subsystems are traced away.
 *
 * \note The number of column vectors of \a V can be smaller than the dimension
 * of \a target. If that is the case, then the measurement probabilities sum up
 * to a number smaller than 1 (i.e., having, in effect, a filtering operation).
 *
 * \param A Eigen expression
 * \param V Matrix whose columns represent the measurement basis vectors or the
 * ket parts of the rank-1 projectors
 * \param target Subsystem indexes that are measured
 * \param d Subsystem dimensions
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Result of the measurement, 2. Vector of outcome
 * probabilities, and 3. Vector of post-measurement normalized states
 */
template <typename Derived>
std::tuple<idx, std::vector<double>, std::vector<expr_t<Derived>>>
measure(const Eigen::MatrixBase<Derived>& A, const cmat& V,
        const std::vector<idx>& target, idx d = 2, bool destructive = true) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::measure()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return measure(rA, V, target, dims, destructive);
}

/**
 * \brief Sequentially measures the part \a target of the multi-partite state
 * vector or density matrix \a A in the computational basis
 * \see qpp::measure(), qpp::sample()
 *
 * \note If \a destructive is set to true (by default), the measurement is
 * destructive, i.e., the measured subsystems are traced away.
 *
 * \param A Eigen expression
 * \param target Subsystem indexes that are measured
 * \param dims Dimensions of the multi-partite system
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Vector of outcome results of the measurement (ordered in
 * increasing order with respect to \a target, i.e. first measurement result
 * corresponds to the subsystem with the smallest index), 2. Outcome
 * probability, and 3. Post-measurement normalized state
 */
template <typename Derived>
[[qpp::critical]] std::tuple<std::vector<idx>, double, expr_t<Derived>>
measure_seq(const Eigen::MatrixBase<Derived>& A, std::vector<idx> target,
            std::vector<idx> dims, bool destructive = true) {
    //    typename std::remove_const<
    //            typename Eigen::MatrixBase<Derived>::EvalReturnType
    //    >::type rA = A.derived();

    // typename Eigen::MatrixBase<Derived>::EvalReturnType rA = A.derived();
    expr_t<Derived> rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure_seq()", "A");

    // check that dimension is valid
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::measure_seq()", "dims");

    // check valid state and matching dimensions
    if (internal::check_cvector(rA)) {
        if (!internal::check_dims_match_cvect(dims, rA))
            throw exception::DimsMismatchCvector("qpp::measure_seq()",
                                                 "A/dims");
    } else if (internal::check_square_mat(rA)) {
        if (!internal::check_dims_match_mat(dims, rA))
            throw exception::DimsMismatchMatrix("qpp::measure_seq()", "A/dims");
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::measure_seq()", "A");

    // check that target is valid w.r.t. dims
    if (!internal::check_subsys_match_dims(target, dims))
        throw exception::SubsysMismatchDims("qpp::measure_seq()",
                                            "dims/target");
    // END EXCEPTION CHECKS

    std::vector<idx> result;
    double prob = 1;

    // sort target in decreasing order,
    // the order of measurements does not matter
    std::sort(std::begin(target), std::end(target), std::greater<idx>{});

    //************ ket or density matrix ************//
    while (!target.empty()) {
        auto tmp = measure(
            rA, Gates::get_no_thread_local_instance().Id(dims[target[0]]),
            {target[0]}, dims, destructive);
        idx m = std::get<0>(tmp);
        result.emplace_back(m);
        prob *= std::get<1>(tmp)[m];
        rA = std::get<2>(tmp)[m];

        if (destructive) {
            // remove the subsystem
            dims.erase(std::next(std::begin(dims),
                                 static_cast<std::ptrdiff_t>(target[0])));
        }
        target.erase(std::begin(target));
    }
    // order result in increasing order with respect to target
    std::reverse(std::begin(result), std::end(result));

    return std::make_tuple(result, prob, rA);
}

/**
 * \brief Sequentially measures the part \a target of the multi-partite state
 * vector or density matrix \a A in the computational basis
 * \see qpp::measure(), qpp::sample()
 *
 * \note If \a destructive is set to true (by default), the measurement is
 * destructive, i.e., the measured subsystems are traced away.
 *
 * \param A Eigen expression
 * \param target Subsystem indexes that are measured
 * \param d Subsystem dimensions
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Vector of outcome results of the measurement (ordered in
 * increasing order with respect to \a target, i.e. first measurement result
 * corresponds to the subsystem with the smallest index), 2. Outcome
 * probability, and 3. Post-measurement normalized state
 */
template <typename Derived>
std::tuple<std::vector<idx>, double, expr_t<Derived>>
measure_seq(const Eigen::MatrixBase<Derived>& A, const std::vector<idx>& target,
            idx d = 2, bool destructive = true) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure_seq()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::measure_seq()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return measure_seq(rA, target, dims, destructive);
}

/**
 * \brief Samples from a quantum state in the computational basis (Z-basis)
 * \see qpp::measure()
 *
 * \param A Eigen expression
 * \param target Subsystem indexes that are sampled
 * \param dims Subsystem dimensions
 * \return Vector of outcome results
 */
template <typename Derived>
[[qpp::critical]] std::vector<idx> sample(const Eigen::MatrixBase<Derived>& A,
                                          const std::vector<idx>& target,
                                          const std::vector<idx>& dims) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::sample()", "A");

    // check that dimension is valid
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::sample()", "dims");

    // check valid state and matching dimensions
    if (internal::check_cvector(rA)) {
        if (!internal::check_dims_match_cvect(dims, rA))
            throw exception::DimsMismatchCvector("qpp::sample()", "A/dims");
    } else if (internal::check_square_mat(rA)) {
        if (!internal::check_dims_match_mat(dims, rA))
            throw exception::DimsMismatchMatrix("qpp::sample()", "A/dims");
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::sample()", "A");

    // check that target is valid w.r.t. dims
    if (!internal::check_subsys_match_dims(target, dims))
        throw exception::SubsysMismatchDims("qpp::sample()", "dims/target");
    // END EXCEPTION CHECKS

    idx D = prod(dims); // total dimension
    // std::sort(target.begin(), target.end());

    bool is_ket = internal::check_cvector(rA);
    std::vector<double> pbs;
    if (is_ket) {
        pbs = qpp::abssq(rA);
    } else {
        pbs.resize(D);
        for (idx i = 0; i < D; ++i)
            pbs[i] = std::real(rA(i, i));
    }

    // sample
    std::discrete_distribution dd(pbs.begin(), pbs.end());
    auto& gen = RandomDevices::get_instance().get_prng();
    idx sample_dec = dd(gen);
    auto sample_midx = n2multiidx(sample_dec, dims);

    // measurement result as a vector of dits
    std::vector<idx> measurement_midx(target.size(), 0);
    std::transform(target.begin(), target.end(), measurement_midx.begin(),
                   [&sample_midx = std::as_const(sample_midx)](idx pos) {
                       return sample_midx[pos];
                   });

    return measurement_midx;
}

/**
 * \brief Samples from a quantum state in the computational basis (Z-basis)
 * \see qpp::measure()
 *
 * \param A Eigen expression
 * \param target Subsystem indexes that are sampled
 * \param d Subsystem dimensions
 * \return Vector of outcome results
 */
template <typename Derived>
std::vector<idx> sample(const Eigen::MatrixBase<Derived>& A,
                        const std::vector<idx>& target, idx d = 2) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::sample()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::sample()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return sample(rA, target, dims);
}

/**
 * \brief Samples repeatedly from a quantum state in the computational basis
 * (Z-basis)
 * \see qpp::measure()
 *
 * \param num_samples Number of samples
 * \param A Eigen expression
 * \param target Subsystem indexes that are sampled
 * \param dims Subsystem dimensions
 * \return Map with vector of outcome results and their corresponding number of
 * appearances
 */
template <typename Derived>
[[qpp::critical]] std::map<std::vector<idx>, idx>
sample(idx num_samples, const Eigen::MatrixBase<Derived>& A,
       const std::vector<idx>& target, const std::vector<idx>& dims) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::sample()", "A");

    // check that dimension is valid
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::sample()", "dims");

    // check valid state and matching dimensions
    if (internal::check_cvector(rA)) {
        if (!internal::check_dims_match_cvect(dims, rA))
            throw exception::DimsMismatchCvector("qpp::sample()", "A/dims");
    } else if (internal::check_square_mat(rA)) {
        if (!internal::check_dims_match_mat(dims, rA))
            throw exception::DimsMismatchMatrix("qpp::sample()", "A/dims");
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::sample()", "A");

    // check that target is valid w.r.t. dims
    if (!internal::check_subsys_match_dims(target, dims))
        throw exception::SubsysMismatchDims("qpp::sample()", "dims/target");

    // check non-zero number of samples
    if (num_samples == 0)
        throw exception::OutOfRange("qpp::sample()", "num_samples");
    // END EXCEPTION CHECKS

    idx D = prod(dims); // total dimension
    // std::sort(target.begin(), target.end());

    bool is_ket = internal::check_cvector(rA);
    std::vector<double> pbs;
    if (is_ket) {
        pbs = qpp::abssq(rA);
    } else {
        pbs.resize(D);
        for (idx i = 0; i < D; ++i)
            pbs[i] = std::real(rA(i, i));
    }

    // sample
    std::discrete_distribution dd(pbs.begin(), pbs.end());
    auto& gen = RandomDevices::get_instance().get_prng();

    std::map<std::vector<idx>, idx> result;
    for (idx i = 0; i < num_samples; ++i) {
        idx sample_dec = dd(gen);
        auto sample_midx = n2multiidx(sample_dec, dims);

        // measurement result as a vector of dits
        std::vector<idx> measurement_midx(target.size(), 0);
        std::transform(target.begin(), target.end(), measurement_midx.begin(),
                       [&sample_midx = std::as_const(sample_midx)](idx pos) {
                           return sample_midx[pos];
                       });
        ++result[measurement_midx];
    }

    return result;
}

/**
 * \brief Samples repeatedly from a quantum state (in the computational basis)
 * \see qpp::measure()
 *
 * \param num_samples Number of samples
 * \param A Eigen expression
 * \param target Subsystem indexes that are sampled
 * \param d Subsystem dimensions
 * \return Map with vector of outcome results and their corresponding number of
 * appearances
 */
template <typename Derived>
std::map<std::vector<idx>, idx>
sample(idx num_samples, const Eigen::MatrixBase<Derived>& A,
       const std::vector<idx>& target, idx d = 2) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::sample()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::sample()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return sample(num_samples, rA, target, dims);
}

/**
 * \brief Resets qudits from the multi-partite state vector or density matrix
 * \a A by performing a non-destructive measurement in the computational basis
 * on the \a target qudits and discarding the measurement results, followed by
 * shifting them back to the \f$|0\cdots 0\rangle\f$ state
 *
 * \param A Eigen expression
 * \param target Target qudit indexes that are reset
 * \param dims Dimensions of the multi-partite system
 * \return Reset quantum state
 */
template <typename Derived>
[[qpp::critical]] dyn_mat<typename Derived::Scalar>
reset(const Eigen::MatrixBase<Derived>& A, const std::vector<idx>& target,
      const std::vector<idx>& dims) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::reset()", "A");

    // check that dimension is valid
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::reset()", "dims");

    // check valid state and matching dimensions
    if (internal::check_cvector(rA)) {
        if (!internal::check_dims_match_cvect(dims, rA))
            throw exception::DimsMismatchCvector("qpp::reset()", "A/dims");
    } else if (internal::check_square_mat(rA)) {
        if (!internal::check_dims_match_mat(dims, rA))
            throw exception::DimsMismatchMatrix("qpp::reset()", "A/dims");
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::reset()", "A");

    // check that target is valid w.r.t. dims
    if (!internal::check_subsys_match_dims(target, dims))
        throw exception::SubsysMismatchDims("qpp::reset()", "dims/target");
    // END EXCEPTION CHECKS

    dyn_mat<typename Derived::Scalar> result;
    std::vector<idx> resZ;

    std::tie(resZ, std::ignore, result) = measure_seq(rA, target, dims, false);
    for (idx i = 0; i < target.size(); ++i) {
        cmat correction =
            powm(Gates::get_no_thread_local_instance().Xd(dims[i]),
                 dims[i] - resZ[i]);
        result = apply(result, correction, {target[i]}, dims);
    }

    return result;
}

/**
 * \brief Resets qudits from the multi-partite state vector or density matrix
 * \a A by performing a non-destructive measurement in the computational basis
 * on the \a target qudits and discarding the measurement results, followed by
 * shifting them back to the \f$|0\cdots 0\rangle\f$ state
 *
 * \param A Eigen expression
 * \param target Target qudit indexes that are reset
 * \param d Subsystem dimensions
 * \return Reset quantum state
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> reset(const Eigen::MatrixBase<Derived>& A,
                                        const std::vector<idx>& target,
                                        idx d = 2) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::reset()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::reset()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return reset(rA, target, dims);
}

/**
 * \brief Discards qudits from the multi-partite state vector or density matrix
 * \a A by performing a destructive measurement in the computational basis on
 * the \a target qudits and discarding the measurement results
 *
 * \param A Eigen expression
 * \param target Target qudit indexes that are discarded
 * \param dims Dimensions of the multi-partite system
 * \return Resulting quantum state
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> discard(const Eigen::MatrixBase<Derived>& A,
                                          const std::vector<idx>& target,
                                          const std::vector<idx>& dims) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::discard()", "A");

    // check that dimension is valid
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::discard()", "dims");

    // check valid state and matching dimensions
    if (internal::check_cvector(rA)) {
        if (!internal::check_dims_match_cvect(dims, rA))
            throw exception::DimsMismatchCvector("qpp::discard()", "A/dims");
    } else if (internal::check_square_mat(rA)) {
        if (!internal::check_dims_match_mat(dims, rA))
            throw exception::DimsMismatchMatrix("qpp::discard()", "A/dims");
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::discard()", "A");

    // check that target is valid w.r.t. dims
    if (!internal::check_subsys_match_dims(target, dims))
        throw exception::SubsysMismatchDims("qpp::discard()", "dims/target");
    // END EXCEPTION CHECKS

    dyn_mat<typename Derived::Scalar> result;
    std::tie(std::ignore, std::ignore, result) = measure_seq(rA, target, dims);

    return result;
}

/**
 * \brief Discards qudits from the multi-partite state vector or density matrix
 * \a A by performing a destructive measurement in the computational basis on
 * the \a target qudits and discarding the measurement results
 *
 * \param A Eigen expression
 * \param target Target qudit indexes that are discarded
 * \param d Subsystem dimensions
 * \return Resulting quantum state
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> discard(const Eigen::MatrixBase<Derived>& A,
                                          const std::vector<idx>& target,
                                          idx d = 2) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::discard()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::discard()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return discard(rA, target, dims);
}

} /* namespace qpp */

#endif /* INSTRUMENTS_HPP_ */
