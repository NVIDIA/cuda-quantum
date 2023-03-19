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
 * \file input_output.hpp
 * \brief Input/output functions
 */

#ifndef INPUT_OUTPUT_HPP_
#define INPUT_OUTPUT_HPP_

namespace qpp {
/**
 * \brief Eigen expression ostream manipulator
 *
 * \param A Eigen expression
 * \param chop Set to zero the elements smaller in absolute value than \a chop
 * \return Instance of qpp::internal::IOManipEigen
 */
template <typename Derived>
internal::IOManipEigen disp(const Eigen::MatrixBase<Derived>& A,
                            double chop = qpp::chop) {
    return internal::IOManipEigen(A, chop);
}

/**
 * \brief Complex number ostream manipulator
 *
 * \param z Complex number (or any other type implicitly cast-able to
 * std::complex<double>)
 * \param chop Set to zero the elements smaller in absolute value than \a chop
 * \return Instance of qpp::internal::IOManipEigen
 */
inline internal::IOManipEigen disp(cplx z, double chop = qpp::chop) {
    return internal::IOManipEigen(z, chop);
}

/**
 * \brief Range ostream manipulator
 *
 * \param first Iterator to the first element of the range
 * \param last  Iterator to the last element of the range
 * \param separator Separator
 * \param start Left marking
 * \param end Right marking
 * \param chop Set to zero the elements smaller in absolute value than \a chop
 * \return Instance of qpp::internal::IOManipRange
 */
template <typename InputIterator>
internal::IOManipRange<InputIterator>
disp(InputIterator first, InputIterator last, const std::string& separator,
     const std::string& start = "[", const std::string& end = "]",
     double chop = qpp::chop) {
    return internal::IOManipRange<InputIterator>(first, last, separator, start,
                                                 end, chop);
}

/**
 * \brief Standard container ostream manipulator. The container must support
 * std::begin(), std::end() and forward iteration.
 *
 * \param c Container
 * \param separator Separator
 * \param start Left marking
 * \param end Right marking
 * \param chop Set to zero the elements smaller in absolute value than \a chop
 * \return Instance of qpp::internal::IOManipRange
 */
template <typename Container>
internal::IOManipRange<typename Container::const_iterator>
disp(const Container& c, const std::string& separator,
     const std::string& start = "[", const std::string& end = "]",
     double chop = qpp::chop,
     typename std::enable_if<is_iterable<Container>::value>::type* = nullptr) {
    return internal::IOManipRange<typename Container::const_iterator>(
        std::begin(c), std::end(c), separator, start, end, chop);
}

/**
 * \brief C-style pointer ostream manipulator
 *
 * \param p Pointer to the first element
 * \param N Number of elements to be displayed
 * \param separator Separator
 * \param start Left marking
 * \param end Right marking
 * \param chop Set to zero the elements smaller in absolute value than \a chop
 * \return Instance of qpp::internal::IOManipPointer
 */
template <typename PointerType>
internal::IOManipPointer<PointerType>
disp(const PointerType* p, idx N, const std::string& separator,
     const std::string& start = "[", const std::string& end = "]",
     double chop = qpp::chop) {
    return internal::IOManipPointer<PointerType>(p, N, separator, start, end,
                                                 chop);
}

/**
 * \brief Saves Eigen expression to a text stream in corresponding machine
 * precision \see qpp::load()
 *
 * Example:
 * \code
 * // saves an Eigen dynamic complex matrix to a text stream
 * std::ofstream fout("mat.txt");
 * cmat mat = rand<cmat>(2, 2); // a 2 x 2 random complex matrix
 * save(mat, fout);
 * \endcode
 *
 * \param A Eigen expression
 * \param os Output text stream
 */
template <typename Derived>
void save(const Eigen::MatrixBase<Derived>& A, std::ostream& os) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::save()", "A");

    if (!os.good()) {
        throw std::runtime_error("qpp::save(): Error writing output stream!");
    }
    // END EXCEPTION CHECKS

    idx rows = static_cast<idx>(rA.rows());
    idx cols = static_cast<idx>(rA.cols());
    os << rows << " " << cols << '\n';

    bool is_complex = qpp::is_complex<typename Derived::Scalar>::value;

    for (idx i = 0; i < rows; ++i) {
        std::string sep;
        for (idx j = 0; j < cols; ++j) {
            if (is_complex) {
                os << sep << '(' << internal::real2text(std::real(rA(i, j)));
                os << ',' << internal::real2text(std::imag(rA(i, j))) << ')';
            } else {
                os << sep << internal::real2text(rA(i, j));
            }
            sep = " ";
        }
        os << '\n';
    }
}

/**
 * \brief Loads a complex Eigen matrix from a text stream in corresponding
 * machine precision
 * \see qpp::save()
 *
 * The template parameter cannot be automatically deduced and must be explicitly
 * provided, depending on the scalar field of the matrix that is being loaded
 *
 * Example:
 * \code
 * // loads a complex Eigen dynamic complex matrix from a text stream
 * std::ifstream fin("mat.txt");
 * cmat mat = load<cmat>(fin);
 * \endcode
 *
 * \param is Input text stream
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar>
load(std::istream& is,
     typename std::enable_if<
         is_complex<typename Derived::Scalar>::value>::type* = nullptr) {
    // EXCEPTION CHECKS

    if (!is.good()) {
        throw std::runtime_error("qpp::load(): Error opening input stream!");
    }
    // END EXCEPTION CHECKS

    idx rows, cols;
    is >> rows >> cols;

    dyn_mat<typename Derived::Scalar> A(rows, cols);

    char skip;
    decltype(std::declval<typename Derived::Scalar>().real()) re, im;

    for (idx i = 0; i < rows; ++i) {
        for (idx j = 0; j < cols; ++j) {
            is >> skip >> re >> skip >> im >> skip; // (re,im)
            A(i, j) = typename Derived::Scalar{re, im};
        }
    }

    return A;
}

/**
 * \brief Loads a real Eigen matrix from a text stream in corresponding machine
 * precision
 * \see qpp::save()
 *
 * The template parameter cannot be automatically deduced and must be explicitly
 * provided, depending on the scalar field of the matrix that is being loaded
 *
 * Example:
 * \code
 * // loads a real Eigen dynamic complex matrix from a text stream
 * std::ifstream fin("mat.txt");
 * dmat mat = load<dmat>(fin);
 * \endcode
 *
 * \param is Input text stream
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar>
load(std::istream& is,
     typename std::enable_if<
         !is_complex<typename Derived::Scalar>::value>::type* = nullptr) {
    // EXCEPTION CHECKS

    if (!is.good()) {
        throw std::runtime_error("qpp::load(): Error opening input stream!");
    }
    // END EXCEPTION CHECKS

    idx rows, cols;
    is >> rows >> cols;

    dyn_mat<typename Derived::Scalar> A(rows, cols);

    for (idx i = 0; i < rows; ++i)
        for (idx j = 0; j < cols; ++j)
            is >> A(i, j);

    return A;
}

namespace obsolete {
/**
 * \brief Saves an Eigen expression to a binary stream (internal format) in
 * double precision
 * \see qpp::obsolete::load()
 *
 * Example:
 *
 * \code
 * // saves an Eigen dynamic complex matrix to a binary stream
 * std::ofstream fout("mat.dat", std::ios::out | std::ios::binary);
 * cmat mat = rand<cmat>(2, 2); // a 2 x 2 random complex matrix
 * save(mat, fout);
 * \endcode
 *
 * \param A Eigen expression
 * \param os Output binary stream
 */
template <typename Derived>
void save(const Eigen::MatrixBase<Derived>& A, std::ostream& os) {
    const dyn_mat<typename Derived::Scalar>& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::obsolete::save()", "A");

    if (!os.good()) {
        throw std::runtime_error(
            "qpp::obsolete::save(): Error writing output stream!");
    }
    // END EXCEPTION CHECKS

    // write the header to file
    const std::string header_ = "TYPE::Eigen::Matrix";
    os.write(header_.c_str(), static_cast<std::ptrdiff_t>(header_.length()));

    idx rows = static_cast<idx>(rA.rows());
    idx cols = static_cast<idx>(rA.cols());
    os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    os.write(reinterpret_cast<const char*>(rA.data()),
             sizeof(typename Derived::Scalar) * rows * cols);
}

/**
 * \brief Loads an Eigen matrix from a binary stream (internal format) in double
 * precision
 * \see qpp::obsolete::save()
 *
 * The template parameter cannot be automatically deduced and must be explicitly
 * provided, depending on the scalar field of the matrix that is being loaded
 *
 * Example:
 *
 * \code
 * // loads an Eigen dynamic complex matrix from a binary stream
 * std::ifstream fin("mat.dat", std::ios::in | std::ios::binary);
 * cmat mat = load<cmat>(fin);
 * \endcode
 *
 * \param is Input binary stream
 */
template <typename Derived>
dyn_mat<typename Derived::Scalar> load(std::istream& is) {
    // EXCEPTION CHECKS

    if (!is.good()) {
        throw std::runtime_error(
            "qpp::obsolete::load(): Error opening input stream!");
    }

    const std::string header_ = "TYPE::Eigen::Matrix";
    std::unique_ptr<char[]> fheader_{new char[header_.length()]};

    // read the header from file
    is.read(fheader_.get(), static_cast<std::ptrdiff_t>(header_.length()));
    if (std::string(fheader_.get(), header_.length()) != header_) {
        throw std::runtime_error(
            "qpp::obsolete::load(): Input stream is corrupted!");
    }
    // END EXCEPTION CHECKS

    idx rows, cols;
    is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    is.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    dyn_mat<typename Derived::Scalar> A(rows, cols);

    is.read(reinterpret_cast<char*>(A.data()),
            sizeof(typename Derived::Scalar) * rows * cols);

    return A;
}
} /* namespace obsolete */

} /* namespace qpp */

#endif /* INPUT_OUTPUT_HPP_ */
