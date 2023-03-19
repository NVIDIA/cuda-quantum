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
 * \file entanglement.hpp
 * \brief Entanglement functions
 */

#ifndef ENTANGLEMENT_HPP_
#define ENTANGLEMENT_HPP_

namespace qpp {
/**
 * \brief Schmidt coefficients of the bi-partite pure state \a A
 *
 * \note The sum of the squares of the Schmidt coefficients equals 1
 * \see qpp::schmidtprobs()
 *
 * \param A Eigen expression
 * \param dims Dimensions of the bi-partite system
 * \return Schmidt coefficients of \a A, ordered in decreasing order, as a real
 * dynamic column vector
 */
template <typename Derived>
dyn_col_vect<double> schmidtcoeffs(const Eigen::MatrixBase<Derived>& A,
                                   const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::schmidtcoeffs()", "A");
    // check bi-partite
    if (dims.size() != 2)
        throw exception::NotBipartite("qpp::schmidtcoeffs()", "dims");
    // check column vector
    if (!internal::check_cvector(rA))
        throw exception::MatrixNotCvector("qpp::schmidtcoeffs()", "A");
    // check matching dimensions
    if (!internal::check_dims_match_cvect(dims, rA))
        throw exception::DimsMismatchCvector("qpp::schmidtcoeffs()", "A/dims");
    // END EXCEPTION CHECKS

    return svals(transpose(reshape(rA, dims[1], dims[0])));
}

/**
 * \brief Schmidt coefficients of the bi-partite pure state \a A
 *
 * \note The sum of the squares of the Schmidt coefficients equals 1
 * \see qpp::schmidtprobs()
 *
 * \param A Eigen expression
 * \param d Subsystem dimensions
 * \return Schmidt coefficients of \a A, ordered in decreasing order, as a real
 * dynamic column vector
 */
template <typename Derived>
dyn_col_vect<double> schmidtcoeffs(const Eigen::MatrixBase<Derived>& A,
                                   idx d = 2) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::schmidtcoeffs()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::schmidtcoeffs()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return schmidtcoeffs(A, dims);
}

/**
 * \brief Schmidt basis on Alice side
 *
 * \param A Eigen expression
 * \param dims Dimensions of the bi-partite system
 * \return Unitary matrix \f$U\f$ whose columns represent the Schmidt basis
 * vectors on Alice side.
 */
template <typename Derived>
cmat schmidtA(const Eigen::MatrixBase<Derived>& A,
              const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::schmidtA()", "A");
    // check bi-partite
    if (dims.size() != 2)
        throw exception::NotBipartite("qpp::schmidtA()", "dims");
    // check column vector
    if (!internal::check_cvector(rA))
        throw exception::MatrixNotCvector("qpp::schmidtA()", "A");
    // check matching dimensions
    if (!internal::check_dims_match_cvect(dims, rA))
        throw exception::DimsMismatchCvector("qpp::schmidtA()", "A/dims");
    // END EXCEPTION CHECKS

    return svdU(transpose(reshape(rA, dims[1], dims[0])));
}

/**
 * \brief Schmidt basis on Alice side
 *
 * \param A Eigen expression
 * \param d Subsystem dimensions
 * \return Unitary matrix \f$U\f$ whose columns represent the Schmidt basis
 * vectors on Alice side.
 */
template <typename Derived>
cmat schmidtA(const Eigen::MatrixBase<Derived>& A, idx d = 2) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::schmidtA()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::schmidtA()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return schmidtA(A, dims);
}

/**
 * \brief Schmidt basis on Bob side
 *
 * \param A Eigen expression
 * \param dims Dimensions of the bi-partite system
 * \return Unitary matrix \f$V\f$ whose columns represent the Schmidt basis
 * vectors on Bob side.
 */
template <typename Derived>
cmat schmidtB(const Eigen::MatrixBase<Derived>& A,
              const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::schmidtB()", "A");
    // check bi-partite
    if (dims.size() != 2)
        throw exception::NotBipartite("qpp::schmidtB()", "dims");
    // check column vector
    if (!internal::check_cvector(rA))
        throw exception::MatrixNotCvector("qpp::schmidtB()", "A");
    // check matching dimensions
    if (!internal::check_dims_match_cvect(dims, rA))
        throw exception::DimsMismatchCvector("qpp::schmidtB()", "A/dims");
    // END EXCEPTION CHECKS

    // by default returns U_B^*, we need U_B, i.e., the complex conjugate,
    // i.e., adjoint(transpose(U_B))
    return svdV(transpose(reshape(conjugate(rA), dims[1], dims[0])));
}

/**
 * \brief Schmidt basis on Bob side
 *
 * \param A Eigen expression
 * \param d Subsystem dimensions
 * \return Unitary matrix \f$V\f$ whose columns represent the Schmidt basis
 * vectors on Bob side.
 */
template <typename Derived>
cmat schmidtB(const Eigen::MatrixBase<Derived>& A, idx d = 2) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::schmidtB()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::schmidtB()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return schmidtB(A, dims);
}

/**
 * \brief Schmidt probabilities of the bi-partite pure state \a A
 *
 * Defined as the squares of the Schmidt coefficients. The sum of the Schmidt
 * probabilities equals 1.
 * \see qpp::schmidtcoeffs()
 *
 * \param A Eigen expression
 * \param dims Dimensions of the bi-partite system
 * \return Real vector consisting of the Schmidt probabilites of \a A, ordered
 * in decreasing order
 */
template <typename Derived>
std::vector<double> schmidtprobs(const Eigen::MatrixBase<Derived>& A,
                                 const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::schmidtprobs()", "A");
    // check bi-partite
    if (dims.size() != 2)
        throw exception::NotBipartite("qpp::schmidtprobs()", "dims");
    // check column vector
    if (!internal::check_cvector(rA))
        throw exception::MatrixNotCvector("qpp::schmidtprobs()", "A");
    // check matching dimensions
    if (!internal::check_dims_match_cvect(dims, rA))
        throw exception::DimsMismatchCvector("qpp::schmidtprobs()", "A/dims");
    // END EXCEPTION CHECKS

    std::vector<double> result;
    dyn_col_vect<double> scf = schmidtcoeffs(rA, dims);
    for (idx i = 0; i < static_cast<idx>(scf.rows()); ++i)
        result.emplace_back(std::pow(scf(i), 2));

    return result;
}

/**
 * \brief Schmidt probabilities of the bi-partite pure state \a A
 *
 * Defined as the squares of the Schmidt coefficients. The sum of the Schmidt
 * probabilities equals 1.
 * \see qpp::schmidtcoeffs()
 *
 * \param A Eigen expression
 * \param d Subsystem dimensions
 * \return Real vector consisting of the Schmidt probabilites of \a A, ordered
 * in decreasing order
 */
template <typename Derived>
std::vector<double> schmidtprobs(const Eigen::MatrixBase<Derived>& A,
                                 idx d = 2) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::schmidtprobs()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::schmidtprobs()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return schmidtprobs(A, dims);
}

/**
 * \brief Schmidt basis on Alice's and Bob's sides, coefficients and
 * probabilities of the bi-partite pure state \a A
 *
 * \see qpp::schmidtA()
 * \see qpp::schmidtB()
 * \see qpp::schmidtcoeffs()
 * \see qpp::schmidtprobs()
 *
 * \param A Eigen expression
 * \param dims Dimensions of the bi-partite system
 * \return Tuple of: 1. Unitary matrix \f$U\f$ whose columns represent the
 * Schmidt basis vectors on Alice side, 2. Unitary matrix \f$V\f$ whose columns
 * represent the Schmidt basis vectors on Bob side, 3. Schmidt coefficients of
 * \a A, ordered in decreasing order, as a real dynamic column vector, and 4.
 * Schmidt probabilites of \a A, ordered in decreasing order, as a real dynamic
 * column vector
 */
template <typename Derived>
std::tuple<cmat, cmat, dyn_col_vect<double>, dyn_col_vect<double>>
schmidt(const Eigen::MatrixBase<Derived>& A, const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::schmidt()", "A");
    // check bi-partite
    if (dims.size() != 2)
        throw exception::NotBipartite("qpp::schmidt()", "dims");
    // check column vector
    if (!internal::check_cvector(rA))
        throw exception::MatrixNotCvector("qpp::schmidt()", "A");
    // check matching dimensions
    if (!internal::check_dims_match_cvect(dims, rA))
        throw exception::DimsMismatchCvector("qpp::schmidt()", "A/dims");
    // END EXCEPTION CHECKS

    auto const sv = svd(transpose(reshape(rA, dims[1], dims[0])));

    return std::make_tuple(std::get<0>(sv), conjugate(std::get<2>(sv)),
                           std::get<1>(sv), std::get<1>(sv).cwiseAbs2().eval());
}

/**
 * \brief Schmidt basis on Alice's and Bob's sides, coefficients and
 * probabilities of the bi-partite pure state \a A
 *
 * \see qpp::schmidtA()
 * \see qpp::schmidtB()
 * \see qpp::schmidtcoeffs()
 * \see qpp::schmidtprobs()
 *
 * \param A Eigen expression
 * \param d Subsystem dimensions
 * \return Tuple of: 1. Unitary matrix \f$U\f$ whose columns represent the
 * Schmidt basis vectors on Alice side, 2. Unitary matrix \f$V\f$ whose columns
 * represent the Schmidt basis vectors on Bob side, 3. Schmidt coefficients of
 * \a A, ordered in decreasing order, as a real dynamic column vector, and 4.
 * Schmidt probabilites of \a A, ordered in decreasing order, as a real dynamic
 * column vector
 */
template <typename Derived>
std::tuple<cmat, cmat, dyn_col_vect<double>, dyn_col_vect<double>>
schmidt(const Eigen::MatrixBase<Derived>& A, idx d = 2) {
    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::schmidt()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::schmidt()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(A.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return schmidt(A, dims);
}

/**
 * \brief Entanglement of the bi-partite pure state \a A
 *
 * Defined as the von-Neumann entropy of the reduced density matrix of one of
 * the subsystems
 * \see qpp::entropy()
 *
 * \param A Eigen expression
 * \param dims Dimensions of the bi-partite system
 * \return Entanglement, with the logarithm in base 2
 */
template <typename Derived>
double entanglement(const Eigen::MatrixBase<Derived>& A,
                    const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::entanglement()", "A");
    // check bi-partite
    if (dims.size() != 2)
        throw exception::NotBipartite("qpp::entanglement()", "dims");
    // check column vector
    if (!internal::check_cvector(rA))
        throw exception::MatrixNotCvector("qpp::entanglement()", "A");
    // check matching dimensions
    if (!internal::check_dims_match_cvect(dims, rA))
        throw exception::DimsMismatchCvector("qpp::entanglement()", "A/dims");
    // END EXCEPTION CHECKS

    return entropy(schmidtprobs(rA, dims));
}

/**
 * \brief Entanglement of the bi-partite pure state \a A
 *
 * Defined as the von-Neumann entropy of the reduced density matrix of one of
 * the subsystems
 * \see qpp::entropy()
 *
 * \param A Eigen expression
 * \param d Subsystem dimensions
 * \return Entanglement, with the logarithm in base 2
 */
template <typename Derived>
double entanglement(const Eigen::MatrixBase<Derived>& A, idx d = 2) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::entanglement()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::entanglement()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return entanglement(A, dims);
}

/**
 * \brief G-concurrence of the bi-partite pure state \a A
 *
 * \note Both local dimensions must be equal
 *
 * Uses qpp::logdet() to avoid overflows
 * \see qpp::logdet()
 *
 * \param A Eigen expression
 * \return G-concurrence
 */
// the G-concurrence
template <typename Derived>
double gconcurrence(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::gconcurrence()", "A");
    // check column vector
    if (!internal::check_cvector(rA))
        throw exception::MatrixNotCvector("qpp::gconcurrence()", "A");

    idx d = internal::get_dim_subsys(static_cast<idx>(rA.rows()), 2);

    // check equal local dimensions
    if (d * d != static_cast<idx>(rA.rows()))
        throw exception::DimsNotEqual("qpp::gconcurrence()", "A");
    // END EXCEPTION CHECKS

    // we compute exp(logdet()) to avoid underflow
    return d * std::abs(std::exp(2. / static_cast<double>(d) *
                                 logdet(reshape(rA, d, d))));
}

/**
 * \brief Negativity of the bi-partite mixed state \a A
 *
 * \param A Eigen expression
 * \param dims Dimensions of the bi-partite system
 * \return Negativity
 */
template <typename Derived>
double negativity(const Eigen::MatrixBase<Derived>& A,
                  const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::negativity()", "A");
    // check bi-partite
    if (dims.size() != 2)
        throw exception::NotBipartite("qpp::negativity()", "dims");
    // check square matrix vector
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::negativity()", "A");
    // check matching dimensions
    if (!internal::check_dims_match_mat(dims, rA))
        throw exception::DimsMismatchMatrix("qpp::negativity()", "A/dims");
    // END EXCEPTION CHECKS

    return (schatten(ptranspose(rA, {0}, dims), 1) - 1.) / 2.;
}

/**
 * \brief Negativity of the bi-partite mixed state \a A
 *
 * \param A Eigen expression
 * \param d Subsystem dimensions
 * \return Negativity
 */
template <typename Derived>
double negativity(const Eigen::MatrixBase<Derived>& A, idx d = 2) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::negativity()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::negativity()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return negativity(A, dims);
}

/**
 * \brief Logarithmic negativity of the bi-partite mixed state \a A
 *
 * \param A Eigen expression
 * \param dims Dimensions of the bi-partite system
 * \return Logarithmic negativity, with the logarithm in base 2
 */
template <typename Derived>
double lognegativity(const Eigen::MatrixBase<Derived>& A,
                     const std::vector<idx>& dims) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::lognegativity()", "A");
    // check bi-partite
    if (dims.size() != 2)
        throw exception::NotBipartite("qpp::lognegativity()", "dims");
    // check square matrix vector
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::lognegativity()", "A");
    // check matching dimensions
    if (!internal::check_dims_match_mat(dims, rA))
        throw exception::DimsMismatchMatrix("qpp::lognegativity()", "A/dims");
    // END EXCEPTION CHECKS

    return std::log2(2 * negativity(rA, dims) + 1);
}

/**
 * \brief Logarithmic negativity of the bi-partite mixed state \a A
 *
 * \param A Eigen expression
 * \param d Subsystem dimensions
 * \return Logarithmic negativity, with the logarithm in base 2
 */
template <typename Derived>
double lognegativity(const Eigen::MatrixBase<Derived>& A, idx d = 2) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::lognegativity()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::lognegativity()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return lognegativity(A, dims);
}

/**
 * \brief Wootters concurrence of the bi-partite qubit mixed state \a A
 *
 * \param A Eigen expression
 * \return Wootters concurrence
 */
template <typename Derived>
double concurrence(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::concurrence()", "A");
    // check square matrix vector
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::concurrence()", "A");
    // check that the state is a 2-qubit state
    if (rA.rows() != 4)
        throw exception::NotQubitSubsys("qpp::concurrence()", "A");
    // END EXCEPTION CHECKS

    cmat sigmaY = Gates::get_no_thread_local_instance().Y;
    dyn_col_vect<double> lambdas =
        evals(rA * kron(sigmaY, sigmaY) * conjugate(rA) * kron(sigmaY, sigmaY))
            .real();

    std::vector<double> lambdas_sorted(lambdas.data(),
                                       lambdas.data() + lambdas.size());

    std::sort(std::begin(lambdas_sorted), std::end(lambdas_sorted),
              std::greater<>());
    std::transform(std::begin(lambdas_sorted), std::end(lambdas_sorted),
                   std::begin(lambdas_sorted), [](double elem) {
                       return std::sqrt(std::abs(elem));
                   }); // chop tiny negatives

    return std::max(0., lambdas_sorted[0] - lambdas_sorted[1] -
                            lambdas_sorted[2] - lambdas_sorted[3]);
}

} /* namespace qpp */

#endif /* ENTANGLEMENT_HPP_ */
