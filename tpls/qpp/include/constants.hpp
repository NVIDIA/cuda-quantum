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
 * \file constants.hpp
 * \brief Constants
 */

#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_

namespace qpp {
inline namespace literals {
/**
 * \brief User-defined literal for complex \f$i=\sqrt{-1}\f$ (integer overload)
 *
 * Example: \code cplx z = 4_i; // type of z is std::complex<double> \endcode
 */
inline constexpr cplx operator"" _i(unsigned long long int x) noexcept {
    return {0., static_cast<double>(x)};
}

/**
 * \brief User-defined literal for complex \f$i=\sqrt{-1}\f$ (real overload)
 *
 * Example: \code cplx z = 4.5_i; // type of z is std::complex<double> \endcode
 */
inline constexpr cplx operator"" _i(long double x) noexcept {
    return {0., static_cast<double>(x)};
}

/**
 * \brief User-defined literal for complex \f$i=\sqrt{-1}\f$ (integer overload)
 *
 * Example: \code auto z = 4_if; // type of z is std::complex<double> \endcode
 */
inline constexpr std::complex<float>
operator"" _if(unsigned long long int x) noexcept {
    return {0., static_cast<float>(x)};
}

/**
 * \brief User-defined literal for complex \f$i=\sqrt{-1}\f$ (real overload)
 *
 * Example: \code auto z = 4.5_if; // type of z is std::complex<float> \endcode
 */
inline constexpr std::complex<float> operator"" _if(long double x) noexcept {
    return {0., static_cast<float>(x)};
}

} /* namespace literals */

/**
 * \brief  Used in qpp::disp() for setting to zero  numbers that have their
 * absolute value smaller than qpp::chop
 */
constexpr double chop = 1e-14;

/**
 * \brief \f$\pi\f$
 */
constexpr double pi = 3.141592653589793238462643383279502884;
/**
 * \brief Base of natural logarithm, \f$e\f$
 */
constexpr double ee = 2.718281828459045235360287471352662497;

/**
 * \brief Used to denote infinity in double precision
 */
constexpr double infty = std::numeric_limits<double>::infinity();

/**
 * \brief D-th root of unity
 *
 * \param D Non-negative integer
 * \return D-th root of unity \f$\exp(2\pi i/D)\f$
 */
inline cplx omega(idx D) {
    if (D == 0)
        throw exception::OutOfRange("qpp::omega()", "D");
    return std::exp(2.0 * pi * 1_i / static_cast<double>(D));
}

/**
 * \brief Constants to be used by std::get<> on the result of qpp::measure(),
 * qpp::measure_seq() etc.
 */
enum {
    RES = 0,  ///< Measurement result(s)
    PROB = 1, ///< Probabilit(y)/(ies)
    ST = 2,   ///< Output state(s)
};

namespace internal {
/**
 * \brief Maximum number of allowed qubits/qudits (subsystems)
 *
 * Used internally to allocate arrays on the stack (for performance reasons)
 */
constexpr idx maxn = 64;
} /* namespace internal */

} /* namespace qpp */

#endif /* CONSTANTS_HPP_ */
