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
 * \file internal/classes/iomanip.hpp
 * \brief Input/output manipulators
 */

#ifndef INTERNAL_CLASSES_IOMANIP_HPP_
#define INTERNAL_CLASSES_IOMANIP_HPP_

namespace qpp::internal {
// ostream manipulators for nice formatting of
// Eigen matrices and STL/C-style containers/vectors

template <typename InputIterator>
class IOManipRange : public IDisplay {
    InputIterator first_, last_;
    std::string separator_, start_, end_;
    double chop_;

  public:
    explicit IOManipRange(InputIterator first, InputIterator last,
                          std::string separator, std::string start = "[",
                          std::string end = "]", double chop = qpp::chop)
        : first_{first}, last_{last}, separator_{std::move(separator)},
          start_{std::move(start)}, end_{std::move(end)}, chop_{chop} {}

    // to silence -Weffc++ warnings for classes that have pointer members
    // (whenever we have a pointer instantiation, i.e., iterator is a raw
    // pointer)
    IOManipRange(const IOManipRange&) = default;

    IOManipRange& operator=(const IOManipRange&) = default;

  private:
    std::ostream& display(std::ostream& os) const override {
        os << start_;

        std::string sep;
        for (InputIterator it = first_; it != last_; ++it) {
            os << sep << abs_chop(*it, chop_);
            sep = separator_;
        }
        os << end_;

        return os;
    }
}; /* class IOManipRange */

template <typename PointerType>
class IOManipPointer : public IDisplay {
    const PointerType* p_;
    idx N_;
    std::string separator_, start_, end_;
    double chop_;

  public:
    explicit IOManipPointer(const PointerType* p, idx N, std::string separator,
                            std::string start = "[", std::string end = "]",
                            double chop = qpp::chop)
        : p_{p}, N_{N}, separator_{std::move(separator)},
          start_{std::move(start)}, end_{std::move(end)}, chop_{chop} {}

    // to silence -Weffc++ warnings for classes that have pointer members
    IOManipPointer(const IOManipPointer&) = default;

    IOManipPointer& operator=(const IOManipPointer&) = default;

  private:
    std::ostream& display(std::ostream& os) const override {
        os << start_;

        for (idx i = 0; i < N_ - 1; ++i)
            os << abs_chop(p_[i], chop_) << separator_;
        if (N_ > 0)
            os << abs_chop(p_[N_ - 1], chop_);

        os << end_;

        return os;
    }
}; /* class IOManipPointer */

// silence g++4.8.x bogus warning -Wnon-virtual-dtor for
// qpp::internal::Display_impl_ when class qpp::internal::IOManipEigen
// privately inherits from it
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) &&  \
    (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif
class IOManipEigen : public IDisplay, private Display_Impl_ {
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) &&  \
    (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
    cmat A_;
    double chop_;

  public:
    // Eigen matrices
    template <typename Derived>
    explicit IOManipEigen(const Eigen::MatrixBase<Derived>& A,
                          double chop = qpp::chop)
        : A_{A.template cast<cplx>()}, // copy, so we can bind rvalues safely
          chop_{chop} {}

    // Complex numbers
    explicit IOManipEigen(const cplx z, double chop = qpp::chop)
        : A_{cmat::Zero(1, 1)}, chop_{chop} {
        // put the complex number inside an Eigen matrix
        A_(0, 0) = z;
    }

  private:
    std::ostream& display(std::ostream& os) const override {
        return display_impl_(A_, os, chop_);
    }
}; /* class IOManipEigen */

} /* namespace qpp::internal */

#endif /* INTERNAL_CLASSES_IOMANIP_HPP_ */
