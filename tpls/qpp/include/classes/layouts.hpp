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
 * \file classes/layouts.hpp
 * \see qpp::ILayout
 * \brief Various qudit placement layouts, all must implement the interface
 * \a qpp::ILayout
 */

#ifndef CLASSES_LAYOUTS_HPP_
#define CLASSES_LAYOUTS_HPP_

namespace qpp {

/**
 * \class qpp::ILayout
 * \brief Mandatory interface for qudit placement layouts
 *
 * \note A layout is a bijection (surjection when there are periodic boundary
 * conditions) between the set of indexes and layout coordinates
 */
class ILayout {
  public:
    /**
     * \brief Computes the index of the point represented by \a xs in the
     * layout coordinate system (bijection)
     *
     * \param xs Vector of coordinates in the layout coordinate system
     * \return Index of the point represented by \a xs in the layout coordinate
     * system
     */
    virtual idx operator()(const std::vector<idx>& xs) const = 0;

    /**
     * \brief Converts index to coordinates (bijection)
     *
     * \param i Index
     * \return Coordinates of the point with index \a i
     */
    virtual std::vector<idx> to_coordinates(idx i) const = 0;

    /**
     * \brief Layout coordinate system dimensions
     *
     * \return Layout coordinate system dimensions
     */
    virtual std::vector<idx> get_dims() const = 0;

    /**
     * \brief Default virtual destructor
     */
    virtual ~ILayout() = default;
}; /* class ILayout */

/**
 * \class qpp::Lattice
 * \brief N-dimensional orthogonal lattice coordinate system
 * \see qpp::PeriodicBoundaryLattice
 */
class Lattice : public ILayout {
  protected:
    std::vector<idx> dims_; ///< lattice dimensions
  public:
    /**
     * \brief Constructor
     * \param dims Vector of lattice dimensions
     */
    explicit Lattice(const std::vector<idx>& dims) : dims_{dims} {
        // EXCEPTION CHECKS

        if (dims.empty())
            throw exception::ZeroSize("qpp::Lattice::Lattice()", "dims");
        // END EXCEPTION CHECKS
    }

    /**
     * \brief Variadic constructor
     *
     * \tparam Ts Variadic type list
     * \param ds Lattice dimensions
     */
    template <class... Ts>
    explicit Lattice(Ts... ds)
        : Lattice(std::vector<idx>{static_cast<idx>(ds)...}) {}

    /**
     * \brief Computes the index of the point represented by \a xs in the
     * lattice coordinate system
     *
     * \param xs Vector of coordinates in the lattice coordinate system
     * \return Index of the point represented by \a xs in the lattice coordinate
     * system
     */
    idx operator()(const std::vector<idx>& xs) const override {
        // EXCEPTION CHECKS

        if (xs.size() != dims_.size())
            throw exception::SizeMismatch("qpp::Lattice::operator()", "xs");
        for (idx i = 0; i < dims_.size(); ++i)
            if (xs[i] >= dims_[i])
                throw exception::OutOfRange("qpp::Lattice::operator()", "xs");
        // END EXCEPTION CHECKS

        return internal::multiidx2n(xs.data(), dims_.size(), dims_.data());
    }

    /**
     * \brief Computes the index of the point represented by \a xs in the
     * lattice coordinate system

     * \tparam Ts Variadic type list
     * \param xs Coordinates in the lattice coordinate system
     * \return Index of the point represented by \a xs in the lattice coordinate
     * system
     */
    template <class... Ts>
    idx operator()(Ts... xs) const {
        return operator()(std::vector<idx>{static_cast<idx>(xs)...});
    }

    /**
     * \brief Converts index to lattice coordinates
     *
     * \param i Index
     * \return Lattice coordinates of the point with index \a i
     */
    std::vector<idx> to_coordinates(idx i) const override {
        // EXCEPTION CHECKS

        if (i >= prod(dims_))
            throw exception::OutOfRange("qpp::Lattice::to_coordinates()", "i");
        // END EXCEPTION CHECKS

        std::vector<idx> result(dims_.size());
        internal::n2multiidx(i, dims_.size(), dims_.data(), result.data());

        return result;
    }

    /**
     * \brief Lattice dimensions
     *
     * \return Lattice dimensions
     */
    std::vector<idx> get_dims() const override { return dims_; }
}; /* class Lattice */

/**
 * \class qpp::PeriodicBoundaryLattice
 * \brief N-dimensional orthogonal lattice coordinate system with periodic
 * boundary conditions
 * \see qpp::Lattice
 */
class PeriodicBoundaryLattice : public Lattice {
  public:
    using Lattice::Lattice;
    using Lattice::operator();

    /**
     * \brief Computes the index of the point represented by \a xs in the
     * lattice coordinate system
     *
     * \param xs Vector of coordinates in the lattice coordinate system
     * \return Index of the point represented by \a xs in the lattice coordinate
     * system
     */
    idx operator()(const std::vector<idx>& xs) const override {
        // EXCEPTION CHECKS

        if (xs.size() != dims_.size())
            throw exception::SizeMismatch("qpp::Lattice::operator()", "xs");
        // END EXCEPTION CHECKS

        std::vector<idx> xs_copy = xs;
        for (idx i = 0; i < dims_.size(); ++i)
            xs_copy[i] = xs[i] % dims_[i];
        return Lattice::operator()(xs_copy);
    }
}; /* class PeriodicBoundaryLattice */

} /* namespace qpp */

#endif /* CLASSES_LAYOUTS_HPP_ */
