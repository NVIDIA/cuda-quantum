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
 * \file functions.hpp
 * \brief Generic quantum computing functions
 */

#ifndef FUNCTIONS_HPP_
#define FUNCTIONS_HPP_

namespace qpp {
// Eigen function wrappers
/**
 * \brief Transpose
 *
 * \param A Eigen expression
 * \return Transpose of \a A, as a dynamic matrix over the same scalar field as
 * \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar>
transpose(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::transpose()", "A");
    // END EXCEPTION CHECKS

    return rA.transpose();
}

/**
 * \brief Complex conjugate
 *
 * \param A Eigen expression
 * \return Complex conjugate of \a A, as a dynamic matrix over the same scalar
 * field as \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar>
conjugate(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::conjugate()", "A");
    // END EXCEPTION CHECKS

    return rA.conjugate();
}

/**
 * \brief Adjoint
 *
 * \param A Eigen expression
 * \return Adjoint (Hermitian conjugate) of \a A, as a dynamic matrix over the
 * same scalar field as \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> adjoint(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::adjoint()", "A");
    // END EXCEPTION CHECKS

    return rA.adjoint();
}

/**
 * \brief Inverse
 *
 * \param A Eigen expression
 * \return Inverse of \a A, as a dynamic matrix over the same scalar field as
 * \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> inverse(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::inverse()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::inverse()", "A");
    // END EXCEPTION CHECKS

    return rA.inverse();
}

/**
 * \brief Trace
 *
 * \param A Eigen expression
 * \return Trace of \a A, as a scalar over the same scalar field as \a A
 */
template <typename Derived>
typename Derived::Scalar trace(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::trace()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::trace()", "A");
    // END EXCEPTION CHECKS

    return rA.trace();
}

/**
 * \brief Determinant
 *
 * \param A Eigen expression
 * \return Determinant of \a A, as a scalar over the same scalar field as \a A.
 * Returns \f$\pm \infty\f$ when the determinant overflows/underflows.
 */
template <typename Derived>
typename Derived::Scalar det(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::det()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::det()", "A");
    // END EXCEPTION CHECKS

    return rA.determinant();
}

/**
 * \brief Logarithm of the determinant
 *
 * Useful when the determinant overflows/underflows
 *
 * \param A Eigen expression
 * \return Logarithm of the determinant of \a A, as a scalar over the same
 * scalar field as \a A
 */
template <typename Derived>
typename Derived::Scalar logdet(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::logdet()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::logdet()", "A");
    // END EXCEPTION CHECKS

    Eigen::PartialPivLU<dyn_mat<typename Derived::Scalar>> lu(rA);
    dyn_mat<typename Derived::Scalar> U =
        lu.matrixLU().template triangularView<Eigen::Upper>();
    typename Derived::Scalar result = std::log(U(0, 0));

    for (idx i = 1; i < static_cast<idx>(rA.rows()); ++i)
        result += std::log(U(i, i));

    return result;
}

/**
 * \brief Element-wise sum of \a A
 *
 * \param A Eigen expression
 * \return Element-wise sum of \a A, as a scalar over the same scalar field as
 * \a A
 */
template <typename Derived>
typename Derived::Scalar sum(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::sum()", "A");
    // END EXCEPTION CHECKS

    return rA.sum();
}

/**
 * \brief Element-wise product of elements of \a A
 *
 * \param A Eigen expression
 * \return Element-wise product of elements of \a A, as a scalar over the same
 * scalar field as \a A
 */
template <typename Derived>
typename Derived::Scalar prod(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::prod()", "A");
    // END EXCEPTION CHECKS

    return rA.prod();
}

/**
 * \brief Frobenius norm
 *
 * \param A Eigen expression
 * \return Frobenius norm of \a A
 */
template <typename Derived>
double norm(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::norm()", "A");
    // END EXCEPTION CHECKS

    // convert matrix to complex then return its norm
    return (rA.template cast<cplx>()).norm();
}

/**
 * \brief Normalizes state vector (column or row vector) or density matrix
 *
 * \param A Eigen expression
 * \return Normalized state vector or density matrix
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar>
normalize(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::normalize()", "A");
    // END EXCEPTION CHECKS

    dyn_mat<typename Derived::Scalar> result;

    if (internal::check_cvector(rA) || internal::check_rvector(rA)) {
        double normA = norm(rA);
        if (normA == 0) {
            throw std::overflow_error("qpp::normalize(): Division by zero!");
        }
        result = rA / normA;
    } else if (internal::check_square_mat(rA)) {
        typename Derived::Scalar traceA = trace(rA);
        if (std::abs(traceA) == 0) {
            throw std::overflow_error("qpp::normalize(): Division by zero!");
        }
        result = rA / trace(rA);
    } else
        throw exception::MatrixNotSquareNorVector("qpp::normalize()", "A");

    return result;
}

/**
 * \brief Full eigen decomposition
 * \see qpp::heig()
 *
 * \param A Eigen expression
 * \return Pair of: 1. Eigenvalues of \a A, as a complex dynamic column vector,
 * and 2. Eigenvectors of \a A, as columns of a complex dynamic matrix
 */
template <typename Derived>
[[qpp::critical]] std::pair<dyn_col_vect<cplx>, cmat>
eig(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::eig()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::eig()", "A");
    // END EXCEPTION CHECKS

    Eigen::ComplexEigenSolver<cmat> es(rA.template cast<cplx>());

    return std::make_pair(es.eigenvalues(), es.eigenvectors());
}

/**
 * \brief Eigenvalues
 * \see qpp::hevals()
 *
 * \param A Eigen expression
 * \return Eigenvalues of \a A, as a complex dynamic column vector
 */
template <typename Derived>
dyn_col_vect<cplx> evals(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::evals()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::evals()", "A");
    // END EXCEPTION CHECKS

    return eig(rA).first;
}

/**
 * \brief Eigenvectors
 * \see qpp::hevects()
 *
 * \param A Eigen expression
 * \return Eigenvectors of \a A, as columns of a complex dynamic matrix
 */
template <typename Derived>
cmat evects(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::evects()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::evects()", "A");
    // END EXCEPTION CHECKS

    Eigen::ComplexEigenSolver<cmat> es(rA.template cast<cplx>());

    return eig(rA).second;
}

/**
 * \brief Full eigen decomposition of Hermitian expression
 * \see qpp::eig()
 *
 * \param A Eigen expression, assumed to be Hermitian
 * \return Pair of:  1. Eigenvalues of \a A, as a real dynamic column vector,
 * and 2. Eigenvectors of \a A, as columns of a complex dynamic matrix
 */
template <typename Derived>
[[qpp::critical]] std::pair<dyn_col_vect<double>, cmat>
heig(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::heig()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::heig()", "A");
    // END EXCEPTION CHECKS

    Eigen::SelfAdjointEigenSolver<cmat> es(rA.template cast<cplx>());

    return std::make_pair(es.eigenvalues(), es.eigenvectors());
}

/**
 * \brief Hermitian eigenvalues
 * \see qpp::evals()
 *
 * \param A Eigen expression, assumed to be Hermitian
 * \return Eigenvalues of Hermitian \a A, as a real dynamic column vector
 */
template <typename Derived>
dyn_col_vect<double> hevals(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::hevals()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::hevals()", "A");
    // END EXCEPTION CHECKS

    return heig(rA).first;
}

/**
 * \brief Eigenvectors of Hermitian matrix
 * \see qpp::evects()
 *
 * \param A Eigen expression, assumed to be Hermitian
 * \return Eigenvectors of Hermitian matrix \a A, as columns of a complex matrix
 */
template <typename Derived>
cmat hevects(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::hevects()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::hevects()", "A");
    // END EXCEPTION CHECKS

    return heig(rA).second;
}

/**
 * \brief Full singular value decomposition
 *
 * \param A Eigen expression
 * \return Tuple of: 1. Left sigular vectors of \a A, as columns of a complex
 * dynamic matrix, 2. Singular values of \a A, ordered in decreasing order,
 * as a real dynamic column vector, and 3. Right singular vectors of \a A,
 * as columns of a complex dynamic matrix
 */
template <typename Derived>
[[qpp::critical]] std::tuple<cmat, dyn_col_vect<double>, cmat>
svd(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::svd()", "A");
    // END EXCEPTION CHECKS

    auto const sv = rA.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

    return std::make_tuple(sv.matrixU(), sv.singularValues(), sv.matrixV());
}

/**
 * \brief Singular values
 *
 * \param A Eigen expression
 * \return Singular values of \a A, ordered in decreasing order, as a real
 * dynamic column vector
 */
template <typename Derived>
dyn_col_vect<double> svals(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::svals()", "A");
    // END EXCEPTION CHECKS

    return rA.bdcSvd().singularValues();
}

/**
 * \brief Left singular vectors
 *
 * \param A Eigen expression
 * \return Complex dynamic matrix, whose columns are the left singular vectors
 * of \a A
 */
template <typename Derived>
[[qpp::critical]] cmat svdU(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::svdU()", "A");
    // END EXCEPTION CHECKS

    return rA.bdcSvd(Eigen::ComputeFullU).matrixU();
}

/**
 * \brief Right singular vectors
 *
 * \param A Eigen expression
 * \return Complex dynamic matrix, whose columns are the right singular vectors
 * of \a A
 */
template <typename Derived>
[[qpp::critical]] cmat svdV(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::svdV()", "A");
    // END EXCEPTION CHECKS

    return rA.bdcSvd(Eigen::ComputeFullV).matrixV();
}

// Matrix functional calculus

/**
 * \brief Functional calculus f(A)
 *
 * \note Does not take into account issues such as numerical stability etc.
 *
 * \param A Diagonalizable Eigen expression
 * \param f Pointer-to-function from complex to complex
 * \return \a \f$f(A)\f$
 */
template <typename Derived>
[[qpp::critical]] cmat funm(const Eigen::MatrixBase<Derived>& A,
                            cplx (*f)(const cplx&)) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::funm()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::funm()", "A");
    // END EXCEPTION CHECKS

    Eigen::ComplexEigenSolver<cmat> es(rA.template cast<cplx>());
    const cmat& evects = es.eigenvectors();
    cmat evals = es.eigenvalues();
    for (idx i = 0; i < static_cast<idx>(evals.rows()); ++i)
        evals(i) = (*f)(evals(i)); // apply f(x) to each eigenvalue

    cmat evalsdiag = evals.asDiagonal();

    return evects * evalsdiag * evects.inverse();
}

/**
 * \brief Matrix square root
 *
 * \param A Eigen expression
 * \return Matrix square root of \a A
 */
template <typename Derived>
cmat sqrtm(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::sqrtm()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::sqrtm()", "A");
    // END EXCEPTION CHECKS

    return funm(rA, &std::sqrt);
}

/**
 * \brief Matrix absolute value
 *
 * \param A Eigen expression
 * \return Matrix absolute value of \a A
 */
template <typename Derived>
cmat absm(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::absm()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::absm()", "A");
    // END EXCEPTION CHECKS

    return sqrtm(adjoint(rA) * rA);
}

/**
 * \brief Matrix exponential
 *
 * \param A Eigen expression
 * \return Matrix exponential of \a A
 */
template <typename Derived>
cmat expm(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::expm()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::expm()", "A");
    // END EXCEPTION CHECKS

    return funm(rA, &std::exp);
}

/**
 * \brief Matrix logarithm
 *
 * \param A Eigen expression
 * \return Matrix logarithm of \a A
 */
template <typename Derived>
cmat logm(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::logm()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::logm()", "A");
    // END EXCEPTION CHECKS

    return funm(rA, &std::log);
}

/**
 * \brief Matrix sin
 *
 * \param A Eigen expression
 * \return Matrix sine of \a A
 */
template <typename Derived>
cmat sinm(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::sinm()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::sinm()", "A");
    // END EXCEPTION CHECKS

    return funm(rA, &std::sin);
}

/**
 * \brief Matrix cos
 *
 * \param A Eigen expression
 * \return Matrix cosine of \a A
 */
template <typename Derived>
cmat cosm(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::cosm()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::cosm()", "A");
    // END EXCEPTION CHECKS

    return funm(rA, &std::cos);
}

/**
 * \brief Matrix power
 * \see qpp::powm()
 *
 * Uses the spectral decomposition of \a A to compute the matrix power. By
 * convention \f$A^0 = I\f$.
 *
 * \param A Diagonalizable Eigen expression
 * \param z Complex number
 * \return Matrix power \f$A^z\f$
 */
template <typename Derived>
cmat spectralpowm(const Eigen::MatrixBase<Derived>& A, const cplx z) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::spectralpowm()", "A");

    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::spectralpowm()", "A");
    // END EXCEPTION CHECKS

    // Define A^0 = Id, for z IDENTICALLY zero
    if (real(z) == 0 && imag(z) == 0)
        return cmat::Identity(rA.rows(), rA.cols());

    Eigen::ComplexEigenSolver<cmat> es(rA.template cast<cplx>());
    const cmat& evects = es.eigenvectors();
    cmat evals = es.eigenvalues();
    for (idx i = 0; i < static_cast<idx>(evals.rows()); ++i)
        evals(i) = std::pow(evals(i), z);

    cmat evalsdiag = evals.asDiagonal();

    return evects * evalsdiag * evects.inverse();
}

/**
 * \brief Fast matrix power based on the SQUARE-AND-MULTIPLY algorithm
 * \see qpp::spectralpowm()
 *
 * Explicitly multiplies the matrix \a A with itself \a n times. By convention
 * \f$A^0 = I\f$.
 *
 * \param A Eigen expression
 * \param n Non-negative integer
 * \return Matrix power \f$A^n\f$, as a dynamic matrix over the same scalar
 * field as \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> powm(const Eigen::MatrixBase<Derived>& A,
                                       idx n) {
    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(A))
        throw exception::ZeroSize("qpp::powm()", "A");

    // check square matrix
    if (!internal::check_square_mat(A))
        throw exception::MatrixNotSquare("qpp::powm()", "A");
    // END EXCEPTION CHECKS

    // if n = 1, return the matrix unchanged
    if (n == 1)
        return A;

    dyn_mat<typename Derived::Scalar> result =
        dyn_mat<typename Derived::Scalar>::Identity(A.rows(), A.cols());

    // if n = 0, return the identity (as just prepared in result)
    if (n == 0)
        return result;

    dyn_mat<typename Derived::Scalar> cA = A.derived(); // copy

    // fast matrix power
    for (; n > 0; n /= 2) {
        if (n % 2)
            result = (result * cA).eval();
        cA = (cA * cA).eval();
    }

    return result;
}

/**
 * \brief Schatten matrix norm
 *
 * \param A Eigen expression
 * \param p Real number, greater or equal to 1, use qpp::infty for
 * \f$p = \infty\f$
 * \return Schatten-\a p matrix norm of \a A
 */
template <typename Derived>
double schatten(const Eigen::MatrixBase<Derived>& A, double p) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::schatten()", "A");
    if (p < 1)
        throw exception::OutOfRange("qpp::schatten()", "p");
    // END EXCEPTION CHECKS

    if (p == infty) // infinity norm (largest singular value)
        return svals(rA)(0);

    const dyn_col_vect<double> sv = svals(rA);
    double result = 0;
    for (idx i = 0; i < static_cast<idx>(sv.rows()); ++i)
        result += std::pow(sv[i], p);

    return std::pow(result, 1. / p);
}

// other functions

/**
 * \brief Functor
 *
 * \param A Eigen expression
 * \param f Pointer-to-function from scalars of \a A to \a OutputScalar
 * \return Component-wise \f$f(A)\f$, as a dynamic matrix over the
 * \a OutputScalar scalar field
 */
template <typename OutputScalar, typename Derived>
[[qpp::parallel]] dyn_mat<OutputScalar>
cwise(const Eigen::MatrixBase<Derived>& A,
      OutputScalar (*f)(typename Derived::Scalar)) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::cwise()", "A");
    // END EXCEPTION CHECKS

    dyn_mat<OutputScalar> result(rA.rows(), rA.cols());

#ifdef HAS_OPENMP
// NOLINTNEXTLINE
#pragma omp parallel for collapse(2)
#endif // HAS_OPENMP
    // column major order for speed
    for (idx j = 0; j < static_cast<idx>(rA.cols()); ++j)
        for (idx i = 0; i < static_cast<idx>(rA.rows()); ++i)
            result(i, j) = (*f)(rA(i, j));

    return result;
}

// Kronecker product of multiple matrices, preserve return type
// variadic template
/**
 * \brief Kronecker product
 * \see qpp::kronpow()
 *
 * Used to stop the recursion for the variadic template version of qpp::kron()
 *
 * \param head Eigen expression
 * \return Its argument \a head
 */
template <typename T>
dyn_mat<typename T::Scalar> kron(const T& head) {
    return head;
}

/**
 * \brief Kronecker product
 * \see qpp::kronpow()
 *
 * \param head Eigen expression
 * \param tail Variadic Eigen expression (zero or more parameters)
 * \return Kronecker product of all input parameters, evaluated from left to
 * right, as a dynamic matrix over the same scalar field as its arguments
 */
template <typename T, typename... Args>
[[qpp::critical]] dyn_mat<typename T::Scalar> kron(const T& head,
                                                   const Args&... tail) {
    return internal::kron2(head, kron(tail...));
}

/**
 * \brief Kronecker product
 * \see qpp::kronpow()
 *
 * \param As std::vector of Eigen expressions
 * \return Kronecker product of all elements in \a As, evaluated from left to
 * right, as a dynamic matrix over the same scalar field as its arguments
 */
template <typename Derived>
[[qpp::critical]] dyn_mat<typename Derived::Scalar>
kron(const std::vector<Derived>& As) {
    // EXCEPTION CHECKS

    if (As.empty())
        throw exception::ZeroSize("qpp::kron()", "As");

    for (auto&& elem : As)
        if (!internal::check_nonzero_size(elem))
            throw exception::ZeroSize("qpp::kron()", "A");
    // END EXCEPTION CHECKS

    dyn_mat<typename Derived::Scalar> result = As[0].derived();
    for (idx i = 1; i < As.size(); ++i) {
        result = kron(result, As[i]);
    }

    return result;
}

// Kronecker product of a list of matrices, preserve return type
// deduce the template parameters from initializer_list
/**
 * \brief Kronecker product
 * \see qpp::kronpow()
 *
 * \param As std::initializer_list of Eigen expressions, such as
 * \a {A1, A2, ... ,Ak}
 * \return Kronecker product of all elements in \a As, evaluated from left to
 * right, as a dynamic matrix over the same scalar field as its arguments
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar>
kron(const std::initializer_list<Derived>& As) {
    return kron(std::vector<Derived>(As));
}

/**
 * \brief Kronecker power
 * \see qpp::kron()
 *
 * \param A Eigen expression
 * \param n Positive integer
 * \return Kronecker product of \a A with itself \a n times \f$A^{\otimes n}\f$,
 * as a dynamic matrix over the same scalar field as \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> kronpow(const Eigen::MatrixBase<Derived>& A,
                                          idx n) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::kronpow()", "A");

    // check out of range
    if (n == 0)
        throw exception::OutOfRange("qpp::kronpow()", "n");
    // END EXCEPTION CHECKS

    std::vector<dyn_mat<typename Derived::Scalar>> As(n, rA);

    return kron(As);
}

// Direct sum of multiple matrices, preserve return type
// variadic template
/**
 * \brief Direct sum
 * \see qpp::dirsumpow()
 *
 * Used to stop the recursion for the variadic template version of qpp::dirsum()
 *
 * \param head Eigen expression
 * \return Its argument \a head
 */
template <typename T>
dyn_mat<typename T::Scalar> dirsum(const T& head) {
    return head;
}

/**
 * \brief Direct sum
 * \see qpp::dirsumpow()
 *
 * \param head Eigen expression
 * \param tail Variadic Eigen expression (zero or more parameters)
 * \return Direct sum of all input parameters, evaluated from left to right, as
 * a dynamic matrix over the same scalar field as its arguments
 */
template <typename T, typename... Args>
dyn_mat<typename T::Scalar> dirsum(const T& head, const Args&... tail) {
    return internal::dirsum2(head, dirsum(tail...));
}

/**
 * \brief Direct sum
 * \see qpp::dirsumpow()
 *
 * \param As std::vector of Eigen expressions
 * \return Direct sum of all elements in \a As, evaluated from left to right, as
 * a dynamic matrix over the same scalar field as its arguments
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> dirsum(const std::vector<Derived>& As) {
    // EXCEPTION CHECKS

    if (As.empty())
        throw exception::ZeroSize("qpp::dirsum()", "As");

    for (auto&& elem : As)
        if (!internal::check_nonzero_size(elem))
            throw exception::ZeroSize("qpp::dirsum()", "A");
    // END EXCEPTION CHECKS

    idx total_rows = 0, total_cols = 0;
    for (idx i = 0; i < As.size(); ++i) {
        total_rows += static_cast<idx>(As[i].rows());
        total_cols += static_cast<idx>(As[i].cols());
    }
    dyn_mat<typename Derived::Scalar> result =
        dyn_mat<typename Derived::Scalar>::Zero(total_rows, total_cols);

    idx cur_row = 0, cur_col = 0;
    for (idx i = 0; i < As.size(); ++i) {
        result.block(cur_row, cur_col, As[i].rows(), As[i].cols()) = As[i];
        cur_row += static_cast<idx>(As[i].rows());
        cur_col += static_cast<idx>(As[i].cols());
    }

    return result;
}

// Direct sum of a list of matrices, preserve return type
// deduce the template parameters from initializer_list
/**
 * \brief Direct sum
 * \see qpp::dirsumpow()
 *
 * \param As std::initializer_list of Eigen expressions,
 * such as \a {A1, A2, ... ,Ak}
 * \return Direct sum of all elements in \a As, evaluated from left to right, as
 * a dynamic matrix over the same scalar field as its arguments
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar>
dirsum(const std::initializer_list<Derived>& As) {
    return dirsum(std::vector<Derived>(As));
}

/**
 * \brief Direct sum power
 * \see qpp::dirsum()
 *
 * \param A Eigen expression
 * \param n Positive integer
 * \return Direct sum of \a A with itself \a n times \f$A^{\oplus n}\f$, as a
 * dynamic matrix over the same scalar field as \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> dirsumpow(const Eigen::MatrixBase<Derived>& A,
                                            idx n) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::dirsumpow()", "A");

    // check out of range
    if (n == 0)
        throw exception::OutOfRange("qpp::dirsumpow()", "n");
    // END EXCEPTION CHECKS

    std::vector<dyn_mat<typename Derived::Scalar>> As(n, rA);

    return dirsum(As);
}

/**
 * \brief Reshape
 *
 * Uses column-major order when reshaping (same as MATLAB)
 *
 * \param A Eigen expression
 * \param rows Number of rows of the reshaped matrix
 * \param cols Number of columns of the reshaped matrix
 * \return Reshaped matrix with \a rows rows and \a cols columns, as a dynamic
 * matrix over the same scalar field as \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> reshape(const Eigen::MatrixBase<Derived>& A,
                                          idx rows, idx cols) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    idx Arows = static_cast<idx>(rA.rows());
    idx Acols = static_cast<idx>(rA.cols());

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::reshape()");

    if (Arows * Acols != rows * cols)
        throw exception::SizeMismatch("qpp::reshape()", "A");
    // END EXCEPTION CHECKS

    return Eigen::Map<dyn_mat<typename Derived::Scalar>>(
        const_cast<typename Derived::Scalar*>(rA.data()), rows, cols);
}

/**
 * \brief Commutator
 * \see qpp::anticomm()
 *
 * Commutator \f$[A,B] = AB - BA\f$. Both \a A and \a B must be Eigen
 * expressions over the same scalar field.
 *
 * \param A Eigen expression
 * \param B Eigen expression
 * \return Commutator \f$AB -BA\f$, as a dynamic matrix over the same scalar
 * field as \a A
 */
template <typename Derived1, typename Derived2>
dyn_mat<typename Derived1::Scalar> comm(const Eigen::MatrixBase<Derived1>& A,
                                        const Eigen::MatrixBase<Derived2>& B) {
    const dyn_mat<typename Derived1::Scalar>& rA = A.derived();
    const dyn_mat<typename Derived2::Scalar>& rB = B.derived();

    // EXCEPTION CHECKS

    // check types
    if (!std::is_same<typename Derived1::Scalar,
                      typename Derived2::Scalar>::value)
        throw exception::TypeMismatch("qpp::comm()", "A/B");

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::comm()", "A");
    if (!internal::check_nonzero_size(rB))
        throw exception::ZeroSize("qpp::comm()", "B");

    // check square matrices
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::comm()", "A");
    if (!internal::check_square_mat(rB))
        throw exception::MatrixNotSquare("qpp::comm()", "B");

    // check equal dimensions
    if (rA.rows() != rB.rows())
        throw exception::DimsNotEqual("qpp::comm()", "A/B");
    // END EXCEPTION CHECKS

    return rA * rB - rB * rA;
}

/**
 * \brief Anti-commutator
 * \see qpp::comm()
 *
 * Anti-commutator \f$\{A,B\} = AB + BA\f$.
 * Both \a A and \a B must be Eigen expressions over the same scalar field.
 *
 * \param A Eigen expression
 * \param B Eigen expression
 * \return Anti-commutator \f$AB +BA\f$, as a dynamic matrix over the same
 * scalar field as \a A
 */
template <typename Derived1, typename Derived2>
dyn_mat<typename Derived1::Scalar>
anticomm(const Eigen::MatrixBase<Derived1>& A,
         const Eigen::MatrixBase<Derived2>& B) {
    const dyn_mat<typename Derived1::Scalar>& rA = A.derived();
    const dyn_mat<typename Derived2::Scalar>& rB = B.derived();

    // EXCEPTION CHECKS

    // check types
    if (!std::is_same<typename Derived1::Scalar,
                      typename Derived2::Scalar>::value)
        throw exception::TypeMismatch("qpp::anticomm()", "A/B");

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::anticomm()", "A");
    if (!internal::check_nonzero_size(rB))
        throw exception::ZeroSize("qpp::anticomm()", "B");

    // check square matrices
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::anticomm()", "A");
    if (!internal::check_square_mat(rB))
        throw exception::MatrixNotSquare("qpp::anticomm()", "B");

    // check equal dimensions
    if (rA.rows() != rB.rows())
        throw exception::DimsNotEqual("qpp::anticomm()", "A/B");
    // END EXCEPTION CHECKS

    return rA * rB + rB * rA;
}

/**
 * \brief Projector
 *
 * Normalized projector onto state vector
 *
 * \param A Eigen expression
 * \return Projector onto the state vector \a A, or the matrix \a Zero if \a A
 * has norm zero, as a dynamic matrix over the same scalar field as \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> prj(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::prj()", "A");

    // check column vector
    if (!internal::check_cvector(rA))
        throw exception::MatrixNotCvector("qpp::prj()", "A");
    // END EXCEPTION CHECKS

    double normA = norm(rA);
    if (normA > 0)
        return rA * adjoint(rA) / (normA * normA);
    else
        return dyn_mat<typename Derived::Scalar>::Zero(rA.rows(), rA.rows());
}

/**
 * \brief Gram-Schmidt orthogonalization
 *
 * \param As std::vector of Eigen expressions as column vectors
 * \return Gram-Schmidt vectors of \a As as columns of a dynamic matrix over the
 * same scalar field as its arguments
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> grams(const std::vector<Derived>& As) {
    // EXCEPTION CHECKS

    // check empty list
    if (!internal::check_nonzero_size(As))
        throw exception::ZeroSize("qpp::grams()", "As");

    for (auto&& elem : As)
        if (!internal::check_nonzero_size(elem))
            throw exception::ZeroSize("qpp::grams()", "A");

    // check that As[0] is a column vector
    if (!internal::check_cvector(As[0]))
        throw exception::MatrixNotCvector("qpp::grams()", "As[0]");

    // now check that all the rest match the size of the first vector
    for (auto&& elem : As)
        if (elem.rows() != As[0].rows() || elem.cols() != 1)
            throw exception::DimsNotEqual("qpp::grams()", "A");
    // END EXCEPTION CHECKS

    dyn_mat<typename Derived::Scalar> cut =
        dyn_mat<typename Derived::Scalar>::Identity(As[0].rows(), As[0].rows());

    dyn_mat<typename Derived::Scalar> vi =
        dyn_mat<typename Derived::Scalar>::Zero(As[0].rows(), 1);

    std::vector<dyn_mat<typename Derived::Scalar>> outvecs;
    // find the first non-zero vector in the list
    idx pos = 0;
    for (pos = 0; pos < As.size(); ++pos) {
        if (norm(As[pos]) > 0) // add it as the first element
        {
            outvecs.emplace_back(As[pos]);
            break;
        }
    }

    // start the process
    for (idx i = pos + 1; i < As.size(); ++i) {
        cut -= prj(outvecs[i - 1 - pos]);
        vi = cut * As[i];
        outvecs.emplace_back(vi);
    }

    dyn_mat<typename Derived::Scalar> result(As[0].rows(), outvecs.size());

    idx tmp = 0;
    for (auto&& elem : outvecs) {
        double normA = norm(elem);
        if (normA > 0) // we add only the non-zero vectors
        {
            result.col(tmp++) = elem / normA;
        }
    }

    return result.block(0, 0, As[0].rows(), tmp);
}

// deduce the template parameters from initializer_list
/**
 * \brief Gram-Schmidt orthogonalization
 *
 * \param As std::initializer_list of Eigen expressions as column vectors
 * \return Gram-Schmidt vectors of \a As as columns of a dynamic matrix over the
 * same scalar field as its arguments
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar>
grams(const std::initializer_list<Derived>& As) {
    return grams(std::vector<Derived>(As));
}

/**
 * \brief Gram-Schmidt orthogonalization
 *
 * \param A Eigen expression, the input vectors are the columns of \a A
 * \return Gram-Schmidt vectors of the columns of \a A, as columns of a dynamic
 * matrix over the same scalar field as \a A
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> grams(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::grams()", "A");
    // END EXCEPTION CHECKS

    std::vector<dyn_mat<typename Derived::Scalar>> input;

    for (idx i = 0; i < static_cast<idx>(rA.cols()); ++i)
        input.emplace_back(rA.col(i));

    return grams<dyn_mat<typename Derived::Scalar>>(input);
}

// TODO check why 2 * internal::maxn
/**
 * \brief Non-negative integer index to multi-index
 * \see qpp::multiidx2n()
 *
 * Uses standard lexicographical order, i.e., 00...0, 00...1 etc.
 *
 * \param n Non-negative integer index
 * \param dims Dimensions of the multi-partite system
 * \return Multi-index of the same size as \a dims
 */
[[qpp::critical]] inline std::vector<idx>
n2multiidx(idx n, const std::vector<idx>& dims) {
    // EXCEPTION CHECKS

    if (dims.size() > internal::maxn)
        throw exception::OutOfRange("qpp::n2multiidx()", "dims/maxn");

    if (!internal::check_dims(dims)) {
        throw exception::DimsInvalid("qpp::n2multiidx()", "dims");
    }

    if (n >= std::accumulate(std::begin(dims), std::end(dims),
                             static_cast<idx>(1), std::multiplies<>())) {
        throw exception::OutOfRange("qpp::n2multiidx()", "n");
    }
    // END EXCEPTION CHECKS

    // double the size for matrices reshaped as vectors
    idx result[2 * internal::maxn];
    internal::n2multiidx(n, dims.size(), dims.data(), result);

    return std::vector<idx>(result, result + dims.size());
}

/**
 * \brief Multi-index to non-negative integer index
 * \see qpp::n2multiidx()
 *
 * Uses standard lexicographical order, i.e., 00...0, 00...1 etc.
 *
 * \param midx Multi-index
 * \param dims Dimensions of the multi-partite system
 * \return Non-negative integer index
 */
[[qpp::critical]] inline idx multiidx2n(const std::vector<idx>& midx,
                                        const std::vector<idx>& dims) {
    // EXCEPTION CHECKS
    if (midx.size() != dims.size())
        throw exception::SizeMismatch("qpp::multiidx2n()", "dims/midx");

    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::multiidx2n()", "dims");

    if (dims.size() > internal::maxn)
        throw exception::OutOfRange("qpp::multiidx2n()", "dims/maxn");

    for (idx i = 0; i < dims.size(); ++i)
        if (midx[i] >= dims[i]) {
            throw exception::OutOfRange("qpp::multiidx2n()", "dims/midx");
        }
    // END EXCEPTION CHECKS

    return internal::multiidx2n(midx.data(), dims.size(), dims.data());
}

/**
 * \brief Multi-partite qudit ket
 * \see qpp::operator "" _ket()
 *
 * Constructs the multi-partite qudit ket \f$|\mathrm{mask}\rangle\f$,
 * where \a mask is a std::vector of non-negative integers. Each element in
 * \a mask has to be smaller than the corresponding element in \a dims.
 *
 * \param mask std::vector of non-negative integers
 * \param dims Dimensions of the multi-partite system
 * \return Multi-partite qudit state vector, as a complex dynamic column vector
 */
inline ket mket(const std::vector<idx>& mask, const std::vector<idx>& dims) {
    idx n = mask.size();

    idx D = std::accumulate(std::begin(dims), std::end(dims),
                            static_cast<idx>(1), std::multiplies<>());

    // EXCEPTION CHECKS

    // check zero size
    if (n == 0)
        throw exception::ZeroSize("qpp::mket()", "mask");
    // check valid dims
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::mket()", "dims");
    // check mask and dims have the same size
    if (mask.size() != dims.size())
        throw exception::SizeMismatch("qpp::mket()", "dims/mask");
    // check mask is a valid vector
    for (idx i = 0; i < n; ++i)
        if (mask[i] >= dims[i])
            throw exception::SubsysMismatchDims("qpp::mket()", "dims/mask");
    // END EXCEPTION CHECKS

    ket result = ket::Zero(D);
    idx pos = multiidx2n(mask, dims);
    result(pos) = 1;

    return result;
}

/**
 * \brief Multi-partite qudit ket
 * \see qpp::operator "" _ket()
 *
 * Constructs the multi-partite qudit ket \f$|\mathrm{mask}\rangle\f$, all
 * subsystem having equal dimension \a d. \a mask is a std::vector of
 * non-negative integers, and each element in \a mask has to be strictly smaller
 * than \a d.
 *
 * \param mask std::vector of non-negative integers
 * \param d Subsystem dimensions
 * \return Multi-partite qudit state vector, as a complex dynamic column vector
 */
inline ket mket(const std::vector<idx>& mask, idx d = 2) {
    idx n = mask.size();
    idx D = static_cast<idx>(std::llround(std::pow(d, n)));

    // EXCEPTION CHECKS

    // check zero size
    if (n == 0)
        throw exception::ZeroSize("qpp::mket()", "mask");

    // check valid dims
    if (d == 0)
        throw exception::DimsInvalid("qpp::mket()", "d");

    // check mask is a valid vector
    for (idx i = 0; i < n; ++i)
        if (mask[i] >= d)
            throw exception::SubsysMismatchDims("qpp::mket()", "d/mask");
    // END EXCEPTION CHECKS

    ket result = ket::Zero(D);
    std::vector<idx> dims(n, d);
    idx pos = multiidx2n(mask, dims);
    result(pos) = 1;

    return result;
}

/**
 * \brief Projector onto multi-partite qudit ket
 * \see qpp::operator "" _prj()
 *
 * Constructs the projector onto the multi-partite qudit ket
 * \f$|\mathrm{mask}\rangle\f$, where \a mask is a std::vector of non-negative
 * integers. Each element in \a mask has to be smaller than the corresponding
 * element in \a dims.
 *
 * \param mask std::vector of non-negative integers
 * \param dims Dimensions of the multi-partite system
 * \return Projector onto multi-partite qudit state vector, as a complex dynamic
 * matrix
 */
inline cmat mprj(const std::vector<idx>& mask, const std::vector<idx>& dims) {
    idx n = mask.size();

    idx D = std::accumulate(std::begin(dims), std::end(dims),
                            static_cast<idx>(1), std::multiplies<>());

    // EXCEPTION CHECKS

    // check zero size
    if (n == 0)
        throw exception::ZeroSize("qpp::mprj()", "mask");
    // check valid dims
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::mprj()", "dims");
    // check mask and dims have the same size
    if (mask.size() != dims.size())
        throw exception::SizeMismatch("qpp::mprj()", "dims/mask");
    // check mask is a valid vector
    for (idx i = 0; i < n; ++i)
        if (mask[i] >= dims[i])
            throw exception::SubsysMismatchDims("qpp::mprj()", "dims/mask");
    // END EXCEPTION CHECKS

    cmat result = cmat::Zero(D, D);
    idx pos = multiidx2n(mask, dims);
    result(pos, pos) = 1;

    return result;
}

/**
 * \brief Projector onto multi-partite qudit ket
 * \see qpp::operator "" _prj()
 *
 * Constructs the projector onto the multi-partite qudit ket
 * \f$|\mathrm{mask}\rangle\f$, all subsystem having equal dimension \a d.
 * \a mask is a std::vector of non-negative integers, and each element in
 * \a mask has to be strictly smaller than \a d.
 *
 * \param mask std::vector of non-negative integers
 * \param d Subsystem dimensions
 * \return Projector onto multi-partite qudit state vector, as a complex dynamic
 * matrix
 */
inline cmat mprj(const std::vector<idx>& mask, idx d = 2) {
    idx n = mask.size();
    idx D = static_cast<idx>(std::llround(std::pow(d, n)));

    // EXCEPTION CHECKS

    // check zero size
    if (n == 0)
        throw exception::ZeroSize("qpp::mprj()", "mask");

    // check valid dims
    if (d == 0)
        throw exception::DimsInvalid("qpp::mprj()", "d");

    // check mask is a valid vector
    for (idx i = 0; i < n; ++i)
        if (mask[i] >= d)
            throw exception::SubsysMismatchDims("qpp::mprj()", "d/mask");
    // END EXCEPTION CHECKS

    cmat result = cmat::Zero(D, D);
    std::vector<idx> dims(n, d);
    idx pos = multiidx2n(mask, dims);
    result(pos, pos) = 1;

    return result;
}

/**
* \brief Computes the absolute values squared of an STL-like range of complex
* numbers

* \param first Iterator to the first element of the range
* \param last  Iterator to the last element of the range
* \return Real vector consisting of the range absolute values squared
*/
template <typename InputIterator>
std::vector<double> abssq(InputIterator first, InputIterator last) {
    std::vector<double> weights(std::distance(first, last));
    std::transform(first, last, std::begin(weights),
                   [](cplx z) -> double { return std::norm(z); });

    return weights;
}

/**
 * \brief Computes the absolute values squared of an STL-like container
 *
 * \param c STL-like container
 * \return Real vector consisting of the container's absolute values squared
 */
template <typename Container>
std::vector<double>
abssq(const Container& c,
      typename std::enable_if<is_iterable<Container>::value>::type* = nullptr)
// we need the std::enable_if to SFINAE out Eigen expressions
// that will otherwise match, instead of matching
// the overload below:

// template<typename Derived>
// abssq(const Eigen::MatrixBase<Derived>& A)
{
    return abssq(std::begin(c), std::end(c));
}

/**
* \brief Computes the absolute values squared of an Eigen expression

* \param A Eigen expression
* \return Real vector consisting of the absolute values squared
*/
template <typename Derived>
std::vector<double> abssq(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::abssq()", "A");
    // END EXCEPTION CHECKS

    return abssq(rA.data(), rA.data() + rA.size());
}

/**
 * \brief Sum of the elements of an STL-like range
 *
 * \note If the range is empty, returns the zero-initialized value_type
 *
 * \param first Iterator to the first element of the range
 * \param last Iterator to the last element of the range
 * \return Sum of the elements of the range
 */
template <typename InputIterator, typename value_type = std::decay_t<
                                      decltype(*std::declval<InputIterator>())>>
value_type sum(InputIterator first, InputIterator last) {
    if (first == last)
        return {};

    value_type result = *first;
    while (++first != last)
        result += *first;

    return result;
}

/**
 * \brief Sum of the elements of an STL-like container
 *
 * \param c STL-like container
 * \return Sum of the elements of the container
 */
template <typename Container>
typename Container::value_type
sum(const Container& c,
    typename std::enable_if<is_iterable<Container>::value>::type* = nullptr) {
    return sum(std::begin(c), std::end(c));
}

/**
 * \brief Sum of the elements of an initializer list
 *
 * \param Ts Initializer list
 * \return Sum of the elements of the list
 */
template <typename T>
T sum(const std::initializer_list<T>& Ts) {
    return sum(std::vector<T>(Ts));
}

/**
 * \brief Product of the elements of an STL-like range
 *
 * \note If the range is empty, returns the zero-initialized value_type
 *
 * \param first Iterator to the first element of the range
 * \param last Iterator to the last element of the range
 * \return Product of the elements of the range
 */
template <typename InputIterator, typename value_type = std::decay_t<
                                      decltype(*std::declval<InputIterator>())>>
value_type prod(InputIterator first, InputIterator last) {
    if (first == last)
        return {};

    value_type result = *first;
    while (++first != last)
        result *= *first;

    return result;
}

/**
 * \brief Product of the elements of an STL-like container
 *
 * \param c STL-like container
 * \return Product of the elements of the container
 */
template <typename Container>
typename Container::value_type
prod(const Container& c,
     typename std::enable_if<is_iterable<Container>::value>::type* = nullptr) {
    return prod(std::begin(c), std::end(c));
}

/**
 * \brief Product of the elements of an initializer list
 *
 * \param Ts Initializer list
 * \return Product of the elements of the list
 */
template <typename T>
T prod(const std::initializer_list<T>& Ts) {
    return prod(std::vector<T>(Ts));
}

/**
 * \brief Finds the pure state representation of a matrix proportional to a
 * projector onto a pure state
 *
 * \note No purity check is done, the input state \a A must have rank one,
 * otherwise the function returns the first non-zero eigenvector of \a A
 *
 * \param A Eigen expression, assumed to be proportional to a projector onto a
 * pure state, i.e., \a A is assumed to have rank one
 * \return The unique non-zero eigenvector of \a A (up to a phase), as a
 * dynamic column vector over the same scalar field as \a A
 */
template <typename Derived>
dyn_col_vect<typename Derived::Scalar>
rho2pure(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS
    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::rho2pure()", "A");
    // check square matrix
    if (!internal::check_square_mat(rA))
        throw exception::MatrixNotSquare("qpp::rho2pure()", "A");
    // END EXCEPTION CHECKS

    dyn_col_vect<double> tmp_evals = hevals(rA);
    cmat tmp_evects = hevects(rA);
    dyn_col_vect<typename Derived::Scalar> result =
        dyn_col_vect<typename Derived::Scalar>::Zero(rA.rows());
    // find the non-zero eigenvector
    // there is only one, assuming the state is pure
    for (idx k = 0; k < static_cast<idx>(rA.rows()); ++k) {
        if (std::abs(tmp_evals(k)) > 0) {
            result = tmp_evects.col(k);
            break;
        }
    }

    return result;
}

/**
 * \brief Constructs the complement (in sorted order) of a subsystem vector
 *
 * \param subsys Subsystem vector
 * \param n Total number of systems
 * \return Complement of \a subsys with respect to the set
 * \f$\{0, 1, \ldots, n - 1\}\f$
 */
inline std::vector<idx> complement(std::vector<idx> subsys, idx n) {
    // EXCEPTION CHECKS

    if (n < subsys.size())
        throw exception::OutOfRange("qpp::complement()", "n");
    for (idx s : subsys)
        if (s >= n)
            throw exception::OutOfRange("qpp::complement()", "n/subsys");
    // END EXCEPTION CHECKS

    std::vector<idx> all(n);
    std::vector<idx> subsys_bar(n - subsys.size());

    std::iota(std::begin(all), std::end(all), 0);
    std::sort(std::begin(subsys), std::end(subsys));
    std::set_difference(std::begin(all), std::end(all), std::begin(subsys),
                        std::end(subsys), std::begin(subsys_bar));

    return subsys_bar;
}

/**
 * \brief Computes the 3-dimensional real Bloch vector corresponding to the
 * qubit density matrix \a A
 * \see qpp::bloch2rho()
 *
 * \note It is implicitly assumed that the density matrix is Hermitian
 *
 * \param A Eigen expression
 * \return 3-dimensional Bloch vector
 */
template <typename Derived>
std::vector<double> rho2bloch(const Eigen::MatrixBase<Derived>& A) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check qubit matrix
    if (!internal::check_qubit_matrix(rA))
        throw exception::NotQubitMatrix("qpp::rho2bloch()", "A");
    // END EXCEPTION CHECKS

    std::vector<double> result(3);
    cmat X(2, 2), Y(2, 2), Z(2, 2);
    X << 0, 1, 1, 0;
    Y << 0, -1_i, 1_i, 0;
    Z << 1, 0, 0, -1;
    result[0] = std::real(trace(rA * X));
    result[1] = std::real(trace(rA * Y));
    result[2] = std::real(trace(rA * Z));

    return result;
}

/**
 * \brief Computes the density matrix corresponding to the 3-dimensional real
 * Bloch vector \a r
 * \see qpp::rho2bloch()
 *
 * \param r 3-dimensional real vector
 * \return Qubit density matrix
 */
inline cmat bloch2rho(const std::vector<double>& r) {
    // EXCEPTION CHECKS

    // check 3-dimensional vector
    if (r.size() != 3)
        throw exception::CustomException("qpp::bloch2rho",
                                         "r is not a 3-dimensional vector");
    // END EXCEPTION CHECKS

    cmat X(2, 2), Y(2, 2), Z(2, 2), Id2(2, 2);
    X << 0, 1, 1, 0;
    Y << 0, -1_i, 1_i, 0;
    Z << 1, 0, 0, -1;
    Id2 << 1, 0, 0, 1;

    return (Id2 + r[0] * X + r[1] * Y + r[2] * Z) / 2.;
}

/**
 * \brief Extracts the dits from a normalized multi-partite pure state in the
 * computational basis. Behaves like the "inverse" of qpp::mket().
 * \see qpp::mket()
 *
 * \note Assumes \a psi is a normalized state vector (ket) in the computational
 * basis, up to a phase; finds the first coefficient of \a psi close to 1 up to
 * \a precision, and returns the digit representation of the corresponding state
 * with a single 1 in that position. If there's no such coefficient (i.e., the
 * state is not a computational basis state up to a phase), returns the empty
 * vector.
 *
 * \param psi Column vector Eigen expression
 * \param dims Dimensions of the multi-partite system
 * \param precision Numerical precision (how close to 1 should the nonzero
 * coefficient be)
 * \return Vector containing the digit representation of \a psi, or the empty
 * vector if \a psi is not a  computational basis state up to a phase.
 */
template <typename Derived>
std::vector<idx> zket2dits(const Eigen::MatrixBase<Derived>& psi,
                           const std::vector<idx>& dims,
                           double precision = 1e-12) {
    const dyn_col_vect<typename Derived::Scalar>& rpsi = psi.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rpsi))
        throw exception::ZeroSize("qpp::zket2dits()", "psi");

    // check column vector
    if (!internal::check_cvector(rpsi))
        throw exception::MatrixNotCvector("qpp::zket2dits()", "psi");

    // check that dims is a valid dimension vector
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::zket2dits()", "dims");

    // check that dims match psi column vector
    if (!internal::check_dims_match_cvect(dims, rpsi))
        throw exception::DimsMismatchCvector("qpp::zket2dits()", "dims/psi");
    // END EXCEPTION CHECKS

    auto N = static_cast<idx>(rpsi.size());
    for (idx i = 0; i < N; ++i) {
        if (std::abs(std::abs(rpsi[i]) - 1.0) < precision) {
            return n2multiidx(i, dims);
        }
    }

    return {};
}

/**
 * \brief Extracts the dits from a normalized multi-partite pure state in the
 * computational basis, all subsystem having equal dimension \a d. Behaves like
 * the "inverse" of qpp::mket().
 * \see qpp::mket()
 *
 * \note Assumes \a psi is a normalized state vector (ket) in the computational
 * basis, up to a phase; finds the first coefficient of \a psi close to 1 up to
 * \a precision, and returns the digit representation of the corresponding state
 * with a single 1 in that position. If there's no such coefficient (i.e., the
 * state is not a computational basis state up to a phase), returns the empty
 * vector.
 *
 * \param psi Column vector Eigen expression
 * \param d Subsystem dimensions
 * \param precision Numerical precision (how close to 1 should the nonzero
 * coefficient be)
 * \return Vector containing the digit representation of \a psi, or the empty
 * vector if \a psi is not a  computational basis state up to a phase.
 */
template <typename Derived>
std::vector<idx> zket2dits(const Eigen::MatrixBase<Derived>& psi, idx d = 2,
                           double precision = 1e-12) {
    const dyn_col_vect<typename Derived::Scalar>& rpsi = psi.derived();

    // EXCEPTION CHECKS
    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::zket2dits()", "d");

    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rpsi.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return zket2dits(psi, dims, precision);
}

inline namespace literals {
// Idea borrowed from https://techblog.altplus.co.jp/entry/2017/11/08/130921
/**
 * \brief Multi-partite qubit ket user-defined literal
 * \see qpp::mket()
 *
 * Constructs the multi-partite qubit ket \f$|\mathrm{Bits}\rangle\f$
 *
 * \tparam Bits String of binary numbers representing the qubit ket
 * \return Multi-partite qubit ket, as a complex dynamic column vector
 */
template <char... Bits>
ket operator"" _ket() {
    constexpr idx n = sizeof...(Bits);
    constexpr char bits[n + 1] = {Bits..., '\0'};
    qpp::ket q = qpp::ket::Zero(static_cast<idx>(std::llround(std::pow(2, n))));

    // EXCEPTION CHECKS

    // check valid multi-partite qubit state
    for (idx i = 0; i < n; ++i) {
        if (bits[i] != '0' && bits[i] != '1')
            throw exception::OutOfRange(R"xxx(qpp::operator "" _ket())xxx",
                                        "Bits");
    }
    // END EXCEPTION CHECKS

    idx pos = std::stoi(bits, nullptr, 2);
    q(pos) = 1;

    return q;
}

/**
 * \brief Multi-partite qubit bra user-defined literal
 * \see qpp::mket() and qpp::adjoint()
 *
 * Constructs the multi-partite qubit bra \f$\langle\mathrm{Bits}|\f$
 *
 * \tparam Bits String of binary numbers representing the qubit bra
 * \return Multi-partite qubit bra, as a complex dynamic row vector
 */
template <char... Bits>
bra operator"" _bra() {
    constexpr idx n = sizeof...(Bits);
    constexpr char bits[n + 1] = {Bits..., '\0'};
    qpp::bra q = qpp::ket::Zero(static_cast<idx>(std::llround(std::pow(2, n))));

    // EXCEPTION CHECKS

    // check valid multi-partite qubit state
    for (idx i = 0; i < n; ++i) {
        if (bits[i] != '0' && bits[i] != '1')
            throw exception::OutOfRange(R"xxx(qpp::operator "" _bra())xxx",
                                        "Bits");
    }
    // END EXCEPTION CHECKS

    idx pos = std::stoi(bits, nullptr, 2);
    q(pos) = 1;

    return q;
}

/**
 * \brief Multi-partite qubit projector user-defined literal
 * \see qpp::mprj()
 *
 * Constructs the multi-partite qubit projector
 * \f$|\mathrm{Bits}\rangle\langle\mathrm{Bits}|\f$ (in the computational basis)
 *
 * \tparam Bits String of binary numbers representing the qubit state to project
 * on
 * \return Multi-partite qubit projector, as a complex dynamic matrix
 */
template <char... Bits>
cmat operator"" _prj() {
    constexpr idx n = sizeof...(Bits);
    constexpr char bits[n + 1] = {Bits..., '\0'};

    // EXCEPTION CHECKS

    // check valid multi-partite qubit state
    for (idx i = 0; i < n; ++i) {
        if (bits[i] != '0' && bits[i] != '1')
            throw exception::OutOfRange(R"xxx(qpp::operator "" _prj())xxx",
                                        "Bits");
    }
    // END EXCEPTION CHECKS

    return kron(operator""_ket<Bits...>(), operator""_bra<Bits...>());
}
} /* namespace literals */

namespace internal {
/**
 * \brief Hash combine
 * \note Code borrowed from boost::hash_combine(), see
 * https://www.boost.org/doc/libs/1_69_0/doc/html/hash/reference.html#boost.hash_combine
 *
 * \tparam T Type
 * \param seed Initial seed, will be updated by the function
 * \param v Value with which the hash is combined
 */
template <class T>
void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << static_cast<std::size_t>(6)) +
            (seed >> static_cast<std::size_t>(2));
}
} /* namespace internal */

/**
 * \brief Computes the hash of en Eigen matrix/vector/expression
 *
 * \param A Eigen expression
 * \param seed Seed, 0 by default
 * \return Hash of its argument
 */
template <typename Derived>
std::size_t hash_eigen(const Eigen::MatrixBase<Derived>& A,
                       std::size_t seed = 0) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::hash_eigen()", "A");
    // END EXCEPTION CHECKS

    auto* p = rA.data();
    idx sizeA = static_cast<idx>(rA.size());
    for (idx i = 0; i < sizeA; ++i) {
        internal::hash_combine(seed, std::real(p[i]));
        internal::hash_combine(seed, std::imag(p[i]));
    }

    return seed;
}

namespace internal {
/**
 * \class qpp::internal::HashEigen
 * \brief Functor for hashing Eigen expressions
 */
struct HashEigen {
    template <typename Derived>
    std::size_t operator()(const Eigen::MatrixBase<Derived>& A) const {
        const dyn_mat<typename Derived::Scalar>& rA = A.derived();
        return hash_eigen(rA);
    }
};

/**
 * \class qpp::internal::EqualEigen
 * \brief Functor for comparing Eigen expressions for equality
 * \note Works without assertion fails even if the dimensions of the arguments
 * are different (in which case it simply returns false)
 */
struct EqualEigen {
    template <typename Derived>
    bool operator()(const Eigen::MatrixBase<Derived>& A,
                    const Eigen::MatrixBase<Derived>& B) const {
        const dyn_mat<typename Derived::Scalar>& rA = A.derived();
        const dyn_mat<typename Derived::Scalar>& rB = B.derived();
        if (rA.rows() == rB.rows() && rA.cols() == rB.cols())
            return rA == rB;
        else
            return false;
    }
};

} /* namespace internal */
} /* namespace qpp */

#endif /* FUNCTIONS_HPP_ */
