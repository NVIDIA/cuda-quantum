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
 * \file MATLAB/matlab.hpp
 * \brief Input/output interfacing with MATLAB
 */

#ifndef MATLAB_MATLAB_HPP_
#define MATLAB_MATLAB_HPP_

#include "mat.h"
#include "mex.h"

namespace qpp {
/**
 * \brief Loads a complex Eigen dynamic matrix from a MATLAB .mat file
 * \see qpp::save_MATLAB()
 *
 * The template parameter cannot be automatically deduced and must be explicitly
 * provided
 *
 * Example:
 * \code
 * // loads a previously saved Eigen ket
 * // from the MATLAB file "input.mat"
 * ket psi = load_MATLAB<ket>("input.mat");
 * \endcode
 *
 * \tparam Derived Complex Eigen type
 * \param mat_file MATLAB .mat file
 * \param var_name Variable name in the .mat file representing the matrix to be
 * loaded
 * \return Eigen dynamic matrix
 */
template <typename Derived> // complex
typename std::enable_if<std::is_same<typename Derived::Scalar, cplx>::value,
                        dyn_mat<cplx>>::type
load_MATLAB(const std::string& mat_file, const std::string& var_name) {
    MATFile* pmat = matOpen(mat_file.c_str(), "r");

    // EXCEPTION CHECKS

    if (!pmat) {
        throw std::runtime_error(
            "qpp::load_MATLAB(): Can not open MATLAB file " + mat_file + "!");
    }

    mxArray* pa = matGetVariable(pmat, var_name.c_str());
    if (!pa)
        throw std::runtime_error(
            "qpp::load_MATLAB(): Can not load the variable " + var_name +
            " from MATLAB file " + mat_file + "!");

    if (mxGetNumberOfDimensions(pa) != 2) // not a matrix
        throw std::runtime_error("qpp::load_MATLAB(): Loaded variable " +
                                 var_name + " is not 2-dimensional!");

    if (!mxIsDouble(pa))
        throw std::runtime_error("qpp::load_MATLAB(): Loaded variable " +
                                 var_name +
                                 " is not in double-precision format!");
        // END EXCEPTION CHECKS

// populate the result
#if MX_HAS_INTERLEAVED_COMPLEX
    dyn_mat<cplx> result(mxGetM(pa), mxGetN(pa));
    std::memcpy((void*) result.data(), (void*) mxGetComplexDoubles(pa),
                sizeof(cplx) * result.size());
    mxDestroyArray(pa);
    matClose(pmat);

    return result;
#else
    dyn_mat<double> result_re(mxGetM(pa), mxGetN(pa));
    dyn_mat<double> result_im(mxGetM(pa), mxGetN(pa));

    // populate the real part of the created array
    std::memcpy((void*) result_re.data(), (void*) mxGetPr(pa),
                sizeof(double) * result_re.size());

    if (mxIsComplex(pa)) // populate the imaginary part if exists
        std::memcpy((void*) result_im.data(), (void*) mxGetPi(pa),
                    sizeof(double) * result_im.size());
    else // set to zero the imaginary part
        std::memset(result_im.data(), 0, sizeof(double) * result_im.size());

    mxDestroyArray(pa);
    matClose(pmat);

    return (result_re.cast<cplx>()) + 1_i * (result_im.cast<cplx>());
#endif // MX_HAS_INTERLEAVED_COMPLEX
}

/**
 * \brief Loads a non-complex (real, integer etc.) Eigen dynamic matrix from a
 * MATLAB .mat file
 * \see qpp::save_MATLAB()
 *
 * The template parameter cannot be automatically deduced and must be explicitly
 * provided
 *
 * Example:
 * \code
 * // loads a previously saved Eigen dynamic double matrix
 * // from the MATLAB file "input.mat"
 * dmat mat = load_MATLAB<dmat>("input.mat");
 * \endcode
 *
 * \tparam Derived Non-complex Eigen type
 * \param mat_file MATLAB .mat file
 * \param var_name Variable name in the .mat file representing the matrix to be
 * loaded
 * \return Eigen dynamic matrix
 */
template <typename Derived> // real
typename std::enable_if<!std::is_same<typename Derived::Scalar, cplx>::value,
                        dyn_mat<typename Derived::Scalar>>::type
load_MATLAB(const std::string& mat_file, const std::string& var_name) {
    MATFile* pmat = matOpen(mat_file.c_str(), "r");

    // EXCEPTION CHECKS

    if (!pmat) {
        throw std::runtime_error(
            "qpp::load_MATLAB(): Can not open MATLAB file " + mat_file + "!");
    }

    mxArray* pa = matGetVariable(pmat, var_name.c_str());
    if (!pa)
        throw std::runtime_error(
            "qpp::load_MATLAB(): Can not load the variable " + var_name +
            " from MATLAB file " + mat_file + "!");

    if (mxGetNumberOfDimensions(pa) != 2) // not a matrix
        throw std::runtime_error("qpp::load_MATLAB(): Loaded variable " +
                                 var_name + " is not 2-dimensional!");

    if (!mxIsDouble(pa))
        throw std::runtime_error("qpp::load_MATLAB(): Loaded variable " +
                                 var_name +
                                 " is not in double-precision format!");
    // END EXCEPTION CHECKS

    dyn_mat<double> result(mxGetM(pa), mxGetN(pa));

// populate the result
#if MX_HAS_INTERLEAVED_COMPLEX
    std::memcpy((void*) result.data(), (void*) mxGetDoubles(pa),
                sizeof(double) * result.size());
#else
    std::memcpy((void*) result.data(), (void*) mxGetPr(pa),
                sizeof(double) * result.size());
#endif // MX_HAS_INTERLEAVED_COMPLEX

    mxDestroyArray(pa);
    matClose(pmat);

    // cast back to the original type
    return result.cast<typename Derived::Scalar>();
}

/**
 * \brief Saves a complex Eigen dynamic matrix to a MATLAB .mat file
 * \see qpp::load_MATLAB()
 *
 * \tparam Complex Eigen type
 * \param A Eigen expression over the complex field
 * \param mat_file MATLAB .mat file
 * \param var_name Variable name in the .mat file representing the matrix to be
 * saved
 * \param mode Saving mode (append, overwrite etc.), see MATLAB \a matOpen()
 * documentation for details
 */
template <typename Derived> // complex
typename std::enable_if<
    std::is_same<typename Derived::Scalar, cplx>::value>::type
save_MATLAB(const Eigen::MatrixBase<Derived>& A, const std::string& mat_file,
            const std::string& var_name, const std::string& mode) {
    const dyn_mat<cplx>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::save_MATLAB()", "A");

    MATFile* pmat = matOpen(mat_file.c_str(), mode.c_str());
    if (!pmat)
        throw std::runtime_error(
            "qpp::save_MATLAB(): Can not open/create MATLAB file " + mat_file +
            "!");
    mxArray* pa = mxCreateDoubleMatrix(rA.rows(), rA.cols(), mxCOMPLEX);
    if (!pa) {
        throw std::runtime_error(
            "qpp::save_MATLAB(): mxCreateDoubleMatrix failed!");
    }
    // END EXCEPTION CHECKS

// populate the MATLAB structure
#if MX_HAS_INTERLEAVED_COMPLEX
    std::memcpy((void*) mxGetComplexDoubles(pa), (void*) rA.data(),
                sizeof(cplx) * rA.size());
#else
    // cast the input to a double (internal MATLAB format)
    dyn_mat<double> tmp_re = rA.real();
    dyn_mat<double> tmp_im = rA.imag();

    // populate the real part of the created array
    std::memcpy((void*) mxGetPr(pa), (void*) tmp_re.data(),
                sizeof(double) * tmp_re.size());

    // populate the imaginary part of the created array
    std::memcpy((void*) mxGetPi(pa), (void*) tmp_im.data(),
                sizeof(double) * tmp_im.size());
#endif // MX_HAS_INTERLEAVED_COMPLEX

    // write it as a MATLAB variable
    if (matPutVariable(pmat, var_name.c_str(), pa))
        throw std::runtime_error(
            "qpp::save_MATLAB(): Can not write the variable " + var_name +
            " to MATLAB file " + mat_file + "!");

    mxDestroyArray(pa);
    matClose(pmat);
}

/**
 * \brief Saves a non-complex (real, integer etc.) Eigen dynamic matrix to a
 * MATLAB .mat file
 * \see qpp::load_MATLAB()
 *
 * \tparam Non-complex Eigen type
 * \param A Non-complex Eigen expression
 * \param mat_file MATLAB .mat file
 * \param var_name Variable name in the .mat file representing the matrix to be
 * saved
 * \param mode Saving mode (append, overwrite etc.), see MATLAB \a matOpen()
 * documentation for details
 */
template <typename Derived> // real
typename std::enable_if<
    !std::is_same<typename Derived::Scalar, cplx>::value>::type
save_MATLAB(const Eigen::MatrixBase<Derived>& A, const std::string& mat_file,
            const std::string& var_name, const std::string& mode) {
    // cast to double, since MATLAB does not work with other types
    const dyn_mat<double>& rA = A.template cast<double>();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::save_MATLAB()", "A");

    MATFile* pmat = matOpen(mat_file.c_str(), mode.c_str());
    if (!pmat)
        throw std::runtime_error(
            "qpp::save_MATLAB(): Can not open/create MATLAB file " + mat_file +
            "!");
    mxArray* pa = mxCreateDoubleMatrix(rA.rows(), rA.cols(), mxREAL);
    if (!pa) {
        throw std::runtime_error(
            "qpp::save_MATLAB(): mxCreateDoubleMatrix failed!");
    }
    // END EXCEPTION CHECKS

// populate the MATLAB structure
#if MX_HAS_INTERLEAVED_COMPLEX
    std::memcpy((void*) mxGetDoubles(pa), (void*) rA.data(),
                sizeof(double) * rA.size());
#else
    std::memcpy((void*) mxGetPr(pa), (void*) rA.data(),
                sizeof(double) * rA.size());
#endif // MX_HAS_INTERLEAVED_COMPLEX

    // write it as a MATLAB variable
    if (matPutVariable(pmat, var_name.c_str(), pa))
        throw std::runtime_error(
            "qpp::save_MATLAB(): Can not write the variable " + var_name +
            " to MATLAB file " + mat_file + "!");

    mxDestroyArray(pa);
    matClose(pmat);
}

} /* namespace qpp */
#endif /* MATLAB_MATLAB_HPP_ */
